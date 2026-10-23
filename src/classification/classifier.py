"""Pseudo-group context-aware classifier."""

from __future__ import annotations

import torch
from torch import nn

from .backbone import FineTunedResNet50
from .context_head import GatedContextHead


class PseudoGroupContextClassifier(nn.Module):
    """
    Context-aware classifier with a shared ResNet50 backbone.

    Target and context images are encoded by the same backbone.
    Context image features are mean-aggregated before gated fusion.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.backbone = FineTunedResNet50(
            pretrained=pretrained
        )

        self.head = GatedContextHead(
            feature_dim=self.backbone.output_dim,
            hidden_dim=256,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(
        self,
        target_images: torch.Tensor,
        context_images: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
       

        # Target features: [B, D]
        target_features = self.backbone(
            target_images
        )

        if context_images.ndim != 5:
            raise ValueError(
                "context_images must have shape "
                "[B, K, C, H, W]"
            )

        B, K, C, H, W = context_images.shape

       
        flat_context = context_images.reshape(
            B * K,
            C,
            H,
            W,
        )

        context_features = self.backbone(
            flat_context
        )


        context_features = context_features.reshape(
            B,
            K,
            -1,
        )

        # Mean aggregation.
        if context_mask is None:
            context_feature = context_features.mean(
                dim=1
            )
        else:
            if context_mask.shape != (B, K):
                raise ValueError(
                    "context_mask must have shape [B, K]"
                )

            mask = context_mask.unsqueeze(-1).to(
                dtype=context_features.dtype
            )

            context_sum = (
                context_features * mask
            ).sum(dim=1)

            context_count = mask.sum(
                dim=1
            ).clamp_min(1.0)

            context_feature = (
                context_sum / context_count
            )

        # Gated context fusion and classification.
        return self.head(
            target_features,
            context_feature,
        )
