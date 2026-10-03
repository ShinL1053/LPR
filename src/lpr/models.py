"""Small educational CNN and a compact single-class plate detector."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

from .ccpd import VOCABS


class PlateCNN(nn.Module):
    """Shared convolutional features with seven position-specific classifiers."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 12)),
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(128 * 2 * 12, 256), nn.ReLU(), nn.Dropout(0.2))
        self.heads = nn.ModuleList(nn.Linear(256, len(vocab)) for vocab in VOCABS)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = self.shared(self.features(images))
        return [head(features) for head in self.heads]


def make_detector(pretrained_backbone: bool = False):
    backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
    return fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=2,  # background and plate
    )


def recognition_loss(logits: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    return sum(nn.functional.cross_entropy(logit, targets[:, position]) for position, logit in enumerate(logits)) / 7


def predicted_indices(logits: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([logit.argmax(dim=1) for logit in logits], dim=1)
