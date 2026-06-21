"""Binary classifier built on EfficientNet-B0 with transfer learning."""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, freeze_backbone=True, dropout=0.3):
    """EfficientNet-B0 with a replaced classifier head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )
    return model


def unfreeze_backbone(model):
    """Call this for fine-tuning phase after the head has converged."""
    for p in model.features.parameters():
        p.requires_grad = True