"""
FakeShield — Grad-CAM Explainability
======================================
Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explanations.

How it works:
    1. Forward pass through the model, hooking the last conv layer
    2. Backward pass from the target class score
    3. Global-average-pool the gradients → per-channel importance weights
    4. Weighted sum of activations → ReLU → normalize to [0, 1]
    5. Overlay as a JET colormap on the original face

The resulting heatmap highlights facial regions the model considers
most important for its REAL/FAKE decision:
    - FAKE faces: typically highlights face-swap boundaries (jawline, eyes, mouth)
    - REAL faces: typically highlights natural skin texture (nose, cheeks)
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks.

    Targets the last convolutional layer of EfficientNet-B4 by default.

    Args:
        model: The DeepfakeDetector model (must be in eval mode).
        target_layer: The layer to hook. If None, uses backbone.features[-1].
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.model.eval()

        # Internal storage for hooked values
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        # Default: last convolutional block in EfficientNet-B4
        if target_layer is None:
            target_layer = model.backbone.features[-1]

        # Register forward and backward hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: save activations from the target layer."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save gradients flowing into the target layer."""
        self._gradients = grad_output[0].detach()

    def generate(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> tuple[np.ndarray, int, float]:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
                          Must have requires_grad=True.
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            Tuple of (heatmap, predicted_class, confidence):
                - heatmap: numpy array of shape (h, w) with values in [0, 1].
                - predicted_class: int (0=REAL, 1=FAKE).
                - confidence: float in [0, 1].
        """
        # Ensure gradients can flow
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        # Use predicted class if no target specified
        if target_class is None:
            target_class = pred_class

        # Backward pass for the target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        # 1. Global average pool gradients → channel importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)

        # 2. Weighted combination of activations
        cam = (weights * self._activations).sum(dim=1, keepdim=True)

        # 3. ReLU — keep only positive contributions
        cam = F.relu(cam)

        # 4. Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred_class, confidence


def overlay_heatmap(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overlay a Grad-CAM heatmap on an image.

    Args:
        image_bgr: Original image in BGR format.
        heatmap: Grad-CAM heatmap of shape (h, w) with values in [0, 1].
        alpha: Blending factor (0=only image, 1=only heatmap).
        colormap: OpenCV colormap to use (default: JET).

    Returns:
        Tuple of (overlay, heatmap_colored):
            - overlay: Alpha-blended image + heatmap in BGR.
            - heatmap_colored: The colorized heatmap in BGR.
    """
    h, w = image_bgr.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap (converts [0,255] grayscale → BGR color)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )

    # Alpha-blend
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay, heatmap_colored
