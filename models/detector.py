"""
FakeShield — DeepfakeDetector Model
====================================
EfficientNet-B4 backbone with ImageNet pretrained weights,
custom 2-class classification head for REAL/FAKE detection.

Architecture:
    EfficientNet-B4 (frozen/unfrozen) → AdaptiveAvgPool → Dropout(0.5) → Linear(1792, 2)

Compatibility:
    Loads weights from the existing trained checkpoint at ./output/ff_effnet_v1/best.pth
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B4 based deepfake detector.

    Args:
        num_classes: Number of output classes (default: 2 for REAL/FAKE).
        dropout: Dropout rate before the final linear layer.
        freeze_backbone: If True, freeze all backbone parameters
                         (used during Phase 1 of 2-phase training).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 freeze_backbone: bool = False):
        super().__init__()

        # Load EfficientNet-B4 with ImageNet-1K pretrained weights
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b4(weights=weights)

        # Optionally freeze backbone for transfer learning warm-up
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the default classifier head with our custom head
        # EfficientNet-B4 outputs 1792-dim features from the pooling layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 380, 380).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)


# ─────────────────────────────────────────────
#  Model Loading Utility
# ─────────────────────────────────────────────

# Default path to the trained model checkpoint
DEFAULT_MODEL_PATH = os.path.join(".", "output", "ff_effnet_v1", "best.pth")


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    num_classes: int = 2,
    dropout: float = 0.5,
    device: torch.device | None = None,
) -> tuple[DeepfakeDetector, torch.device]:
    """
    Load a trained DeepfakeDetector from a checkpoint file.

    Args:
        model_path: Path to the .pth weights file.
        num_classes: Number of output classes the model was trained with.
        dropout: Dropout rate (must match training config).
        device: Target device. Auto-detects GPU if None.

    Returns:
        Tuple of (model, device).

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}\n"
            f"Train a model first or provide a valid --model_path."
        )

    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate architecture (no backbone freeze at inference time)
    model = DeepfakeDetector(
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=False,
    )

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    return model, device
