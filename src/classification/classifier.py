"""Complete pseudo-group context-aware classifier aligned with the paper."""

from __future__ import annotations

import torch
from torch import nn

from .backbone import FineTunedResNet50
from .context_head import GatedContextHead


class PseudoGroupContextClassifier(nn.Module):
    """Fine-tuned target encoder + fixed precomputed pseudo-group context."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = FineTunedResNet50(pretrained=pretrained)
        self.head = GatedContextHead(
            feature_dim=self.backbone.output_dim,
            hidden_dim=256,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict from trainable target features and fixed cached context features."""
        target_features = self.backbone(images)
        return self.head(target_features, context_features)
