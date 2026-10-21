"""ResNet50 feature extractors used by the model."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class FineTunedResNet50(nn.Module):
    """ResNet50 with only layer4 trainable and all BatchNorm layers frozen."""

    output_dim = 2048

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Identity()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.layer4.parameters():
            parameter.requires_grad_(True)
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
        return self.model(images)


class FrozenResNet50(nn.Module):
    """Completely frozen ResNet50 used to build fixed pseudo-group context."""

    output_dim = 2048

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Identity()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenResNet50":
        # Context encoder must remain frozen/eval throughout training.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def frozen_copy(backbone: FineTunedResNet50) -> FrozenResNet50:
    """Create a frozen encoder snapshot from a FineTunedResNet50 instance."""
    frozen = FrozenResNet50(pretrained=False)
    frozen.model.load_state_dict(copy.deepcopy(backbone.model.state_dict()))
    frozen.eval()
    return frozen
