"""
FakeShield — Preprocessing Pipeline
=====================================
Handles the full image preprocessing chain:
    Raw Image → Face Detection (MTCNN) → Crop & Expand → Resize → Normalize → Tensor

Key design decisions:
    - Bounding box is expanded by 1.3× to capture surrounding context
      (hair, ears, jawline) which often contains manipulation artifacts.
    - Images are resized to 380×380 — EfficientNet-B4's native resolution.
    - Normalization uses ImageNet mean/std because the backbone was
      pretrained on ImageNet.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 380  # EfficientNet-B4 native resolution
BBOX_EXPAND = 1.3  # Expansion factor for face bounding box


# ─────────────────────────────────────────────
#  Transform Pipelines
# ─────────────────────────────────────────────

def get_transforms(mode: str = "test") -> transforms.Compose:
    """
    Get the image transform pipeline for a given mode.

    Args:
        mode: One of 'train', 'val', or 'test'.

    Returns:
        A torchvision Compose transform.
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        # Validation and test use the same deterministic pipeline
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─────────────────────────────────────────────
#  MTCNN Face Detector (cached singleton)
# ─────────────────────────────────────────────

_mtcnn_cache: dict[str, MTCNN] = {}


def _get_mtcnn(device: torch.device) -> MTCNN:
    """Get or create a cached MTCNN instance for the given device."""
    key = str(device)
    if key not in _mtcnn_cache:
        _mtcnn_cache[key] = MTCNN(
            keep_all=False,
            select_largest=True,
            device=device,
        )
    return _mtcnn_cache[key]


# ─────────────────────────────────────────────
#  Face Detection
# ─────────────────────────────────────────────

def detect_face(
    image_bgr: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, float]:
    """
    Detect the largest face in an image using MTCNN.

    Args:
        image_bgr: Input image in BGR format (OpenCV convention).
        device: PyTorch device for MTCNN. Auto-detects if None.

    Returns:
        Tuple of (boxes, best_box, best_confidence):
            - boxes: All detected bounding boxes or None.
            - best_box: (x1, y1, x2, y2) of the highest-confidence face, or None.
            - best_confidence: Detection confidence of the best face (0.0 if none).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = _get_mtcnn(device)

    # MTCNN expects RGB PIL image
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None, None, 0.0

    # Select the highest-confidence detection
    best_idx = probs.argmax()
    best_box = tuple(int(b) for b in boxes[best_idx])
    best_conf = float(probs[best_idx])

    return boxes, best_box, best_conf


# ─────────────────────────────────────────────
#  Face Cropping
# ─────────────────────────────────────────────

def crop_face(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    expand: float = BBOX_EXPAND,
) -> np.ndarray:
    """
    Crop a face region from an image with expanded bounding box.

    The bounding box is expanded by `expand` factor to include surrounding
    context (hair, ears, jawline) which often contains manipulation clues.

    Args:
        image_bgr: Full image in BGR format.
        bbox: Face bounding box as (x1, y1, x2, y2).
        expand: Expansion factor (1.3 = 30% larger than detected bbox).

    Returns:
        Cropped face region as BGR numpy array.

    Raises:
        ValueError: If the crop would be empty.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    # Compute expanded square crop centered on the face
    bw, bh = x2 - x1, y2 - y1
    size = int(max(bw, bh) * expand)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Clamp to image boundaries
    nx1 = max(cx - size // 2, 0)
    ny1 = max(cy - size // 2, 0)
    size = min(w - nx1, size)
    size = min(h - ny1, size)

    crop = image_bgr[ny1 : ny1 + size, nx1 : nx1 + size]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop from bbox={bbox} on image of shape {image_bgr.shape}"
        )

    return crop


# ─────────────────────────────────────────────
#  Full Preprocessing (Detection → Crop → Tensor)
# ─────────────────────────────────────────────

def preprocess_face(
    image_bgr: np.ndarray,
    device: torch.device | None = None,
    target_size: int = INPUT_SIZE,
) -> tuple[torch.Tensor | None, np.ndarray | None, tuple | None]:
    """
    Full preprocessing pipeline: detect → crop → resize → normalize → tensor.

    This is the primary entry point for inference preprocessing.

    Args:
        image_bgr: Raw input image in BGR format.
        device: PyTorch device. Auto-detects if None.
        target_size: Target resolution (default: 380 for EfficientNet-B4).

    Returns:
        Tuple of (tensor, crop_bgr, bbox):
            - tensor: Preprocessed tensor of shape (1, 3, 380, 380), or None if no face.
            - crop_bgr: The resized face crop in BGR (for visualization), or None.
            - bbox: Original face bounding box (x1, y1, x2, y2), or None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Detect face
    _, best_box, conf = detect_face(image_bgr, device)

    if best_box is None:
        return None, None, None

    # Step 2: Crop with expanded bbox
    crop = crop_face(image_bgr, best_box)

    # Step 3: Resize to model input size
    crop_resized = cv2.resize(crop, (target_size, target_size))

    # Step 4: Convert BGR → RGB → PIL → tensor with normalization
    rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    transform = get_transforms("test")
    tensor = transform(pil_img).unsqueeze(0).to(device)  # (1, 3, H, W)

    return tensor, crop_resized, best_box
