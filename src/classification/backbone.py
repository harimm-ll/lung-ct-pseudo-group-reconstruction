"""Shared ResNet50 backbone used by the model."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class FineTunedResNet50(nn.Module):
    """
    Shared ResNet50 backbone.

    - All backbone parameters are frozen except layer4.
    - The final residual stage (layer4) is trainable.
    - All BatchNorm layers remain frozen.
    - The same backbone instance is used for target and context images.
    """

    output_dim = 2048

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)

        # Remove ImageNet classification head.
        self.model.fc = nn.Identity()

        # Freeze the complete backbone.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        # Fine-tune only the final residual stage.
        for parameter in self.model.layer4.parameters():
            parameter.requires_grad_(True)

        # Keep all BatchNorm layers frozen.
        self._freeze_batch_norm()

    def _freeze_batch_norm(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FineTunedResNet50":
        super().train(mode)


        self._freeze_batch_norm()

        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images:
                Tensor with shape [N, 3, H, W].

        Returns:
            Image-level features with shape [N, 2048].
        """
        return self.model(images)
