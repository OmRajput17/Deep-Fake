"""
FakeShield — Inference Pipeline
=================================
High-level API that orchestrates the full detection flow:
    Input → Face Detection → Preprocessing → Model → Prediction → Explainability → Result

Usage:
    from inference import FakeShieldPipeline

    pipeline = FakeShieldPipeline()
    result = pipeline.analyze_image("photo.jpg")
    print(result)

    # Or for video:
    result = pipeline.analyze_video("video.mp4")
"""
import os
import sys
import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from PIL import Image

from models.detector import load_model, DEFAULT_MODEL_PATH
from utils.preprocessing import (
    detect_face,
    crop_face,
    preprocess_face,
    INPUT_SIZE,
)
from utils.gradcam import GradCAM, overlay_heatmap


# ─────────────────────────────────────────────
#  Result Data Classes
# ─────────────────────────────────────────────

LABEL_MAP = {0: "REAL", 1: "FAKE"}


@dataclass
class ImageResult:
    """Result of analyzing a single image."""

    label: str                          # "REAL" or "FAKE"
    confidence: float                   # 0.0 – 1.0
    face_detected: bool                 # Whether a face was found
    bbox: tuple | None = None           # (x1, y1, x2, y2) of detected face
    face_crop: np.ndarray | None = None # Resized face crop (BGR, 380×380)
    heatmap: np.ndarray | None = None   # Grad-CAM heatmap (H×W, 0–1)
    overlay: np.ndarray | None = None   # Heatmap overlaid on face (BGR)

    def __repr__(self) -> str:
        if not self.face_detected:
            return "ImageResult(face_detected=False)"
        return (
            f"ImageResult(label={self.label!r}, "
            f"confidence={self.confidence:.1%}, "
            f"bbox={self.bbox})"
        )


@dataclass
class FrameResult:
    """Result for a single video frame."""

    frame_index: int
    timestamp: float       # Seconds into the video
    label: str
    confidence: float


@dataclass
class VideoResult:
    """Result of analyzing a video."""

    verdict: str                         # Overall "REAL" or "FAKE"
    fake_percentage: float               # % of frames classified as fake
    total_frames_analyzed: int
    real_count: int
    fake_count: int
    duration_seconds: float
    fps: float
    frame_results: list[FrameResult] = field(default_factory=list)
    sample_overlays: list[np.ndarray] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"VideoResult(verdict={self.verdict!r}, "
            f"fake={self.fake_percentage:.1%}, "
            f"frames_analyzed={self.total_frames_analyzed}, "
            f"duration={self.duration_seconds:.1f}s)"
        )


# ─────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────

class FakeShieldPipeline:
    """
    End-to-end deepfake detection pipeline.

    Handles model loading, face detection, preprocessing, inference,
    and Grad-CAM explainability in a single, easy-to-use class.

    Args:
        model_path: Path to the trained .pth checkpoint.
        device: Target device. Auto-detects GPU if None.
        enable_gradcam: Whether to generate Grad-CAM heatmaps (default: True).

    Example:
        >>> pipeline = FakeShieldPipeline()
        >>> result = pipeline.analyze_image("suspect_photo.jpg")
        >>> print(f"{result.label} — {result.confidence:.1%}")
        FAKE — 97.3%
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: torch.device | None = None,
        enable_gradcam: bool = True,
    ):
        print(f"[*] FakeShield -- Loading model from {model_path}")
        self.model, self.device = load_model(model_path, device=device)
        self.enable_gradcam = enable_gradcam

        # Initialize Grad-CAM if enabled
        self.gradcam = GradCAM(self.model) if enable_gradcam else None

        device_name = (
            torch.cuda.get_device_name(0)
            if self.device.type == "cuda"
            else "CPU"
        )
        print(f"[+] Model loaded on {device_name}")

    # ── Image Analysis ──────────────────────

    def analyze_image(self, image_input) -> ImageResult:
        """
        Analyze a single image for deepfake content.

        Args:
            image_input: One of:
                - str: file path to an image
                - np.ndarray: image in BGR format (OpenCV)
                - PIL.Image: PIL image object

        Returns:
            ImageResult with prediction, confidence, face crop, and heatmap.
        """
        # Normalize input to BGR numpy array
        image_bgr = self._load_image(image_input)

        # Step 1: Detect face and preprocess
        tensor, crop_bgr, bbox = preprocess_face(image_bgr, self.device)

        if tensor is None:
            return ImageResult(
                label="UNKNOWN",
                confidence=0.0,
                face_detected=False,
            )

        # Step 2: Run inference (with or without Grad-CAM)
        if self.enable_gradcam and self.gradcam is not None:
            heatmap, pred_class, confidence = self.gradcam.generate(tensor)
            overlay, _ = overlay_heatmap(crop_bgr, heatmap)
        else:
            # Inference without Grad-CAM (faster)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            heatmap = None
            overlay = None

        label = LABEL_MAP.get(pred_class, "UNKNOWN")

        return ImageResult(
            label=label,
            confidence=confidence,
            face_detected=True,
            bbox=bbox,
            face_crop=crop_bgr,
            heatmap=heatmap,
            overlay=overlay,
        )

    # ── Video Analysis ──────────────────────

    def analyze_video(
        self,
        video_path: str,
        frames_per_second: int = 3,
        max_samples: int = 6,
    ) -> VideoResult:
        """
        Analyze a video for deepfake content by sampling frames.

        Args:
            video_path: Path to the video file.
            frames_per_second: How many frames to analyze per second of video.
            max_samples: Maximum number of sample overlays to store.

        Returns:
            VideoResult with overall verdict and per-frame results.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Calculate frame skip interval
        frame_skip = max(1, int(fps / frames_per_second))

        real_count = 0
        fake_count = 0
        frame_results: list[FrameResult] = []
        sample_overlays: list[np.ndarray] = []
        frame_idx = 0

        print(f"[*] Analyzing {total_frames} frames ({duration:.1f}s) "
              f"at ~{frames_per_second} fps sampling...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                result = self.analyze_image(frame)

                if result.face_detected:
                    if result.label == "FAKE":
                        fake_count += 1
                    else:
                        real_count += 1

                    frame_results.append(FrameResult(
                        frame_index=frame_idx,
                        timestamp=frame_idx / fps,
                        label=result.label,
                        confidence=result.confidence,
                    ))

                    # Store sample overlays for visualization
                    if len(sample_overlays) < max_samples and result.overlay is not None:
                        sample_overlays.append(result.overlay)

            frame_idx += 1

        cap.release()

        # Determine overall verdict
        total_analyzed = real_count + fake_count
        if total_analyzed == 0:
            verdict = "UNKNOWN"
            fake_pct = 0.0
        else:
            fake_pct = fake_count / total_analyzed
            verdict = "FAKE" if fake_count > real_count else "REAL"

        print(f"[!] Verdict: {verdict} "
              f"({fake_pct:.1%} fake, {total_analyzed} frames analyzed)")

        return VideoResult(
            verdict=verdict,
            fake_percentage=fake_pct,
            total_frames_analyzed=total_analyzed,
            real_count=real_count,
            fake_count=fake_count,
            duration_seconds=duration,
            fps=fps,
            frame_results=frame_results,
            sample_overlays=sample_overlays,
        )

    # ── Utilities ───────────────────────────

    @staticmethod
    def _load_image(image_input) -> np.ndarray:
        """
        Normalize various image input types to BGR numpy array.

        Accepts: file path (str), numpy array (BGR), or PIL Image.
        """
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image_bgr = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Failed to decode image: {image_input}")
            return image_bgr

        elif isinstance(image_input, np.ndarray):
            return image_input

        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        else:
            raise TypeError(
                f"Unsupported image type: {type(image_input)}. "
                f"Expected str, np.ndarray, or PIL.Image."
            )

    def save_result(
        self, result: ImageResult, output_dir: str = "./results"
    ) -> dict[str, str]:
        """
        Save analysis results (face crop, heatmap, overlay) to disk.

        Args:
            result: An ImageResult from analyze_image().
            output_dir: Directory to save output images.

        Returns:
            Dict mapping image type to file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved = {}

        if result.face_crop is not None:
            path = os.path.join(output_dir, "face_crop.png")
            cv2.imwrite(path, result.face_crop)
            saved["face_crop"] = path

        if result.heatmap is not None:
            heatmap_vis = (result.heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(
                cv2.resize(heatmap_vis, (INPUT_SIZE, INPUT_SIZE)),
                cv2.COLORMAP_JET,
            )
            path = os.path.join(output_dir, "heatmap.png")
            cv2.imwrite(path, heatmap_colored)
            saved["heatmap"] = path

        if result.overlay is not None:
            path = os.path.join(output_dir, "overlay.png")
            cv2.imwrite(path, result.overlay)
            saved["overlay"] = path

        return saved
