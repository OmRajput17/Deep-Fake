"""
EfficientNet-B4 with ImageNet pretrained weights.
Replaces XceptionNet for better feature extraction via transfer learning.
"""
import torch.nn as nn
import torchvision.models as models


class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5, freeze_backbone=False):
        super().__init__()
        # Load EfficientNet-B4 with ImageNet pretrained weights
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b4(weights=weights)

        # Optionally freeze backbone for first few epochs
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def unfreeze_backbone(self):
        """Call after initial epochs to fine-tune full network."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def model_selection(modelname='efficientnet_b4', num_out_classes=2,
                    dropout=0.5, freeze_backbone=False):
    if modelname == 'efficientnet_b4':
        return DeepfakeDetector(
            num_classes=num_out_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    else:
        raise NotImplementedError(modelname)
