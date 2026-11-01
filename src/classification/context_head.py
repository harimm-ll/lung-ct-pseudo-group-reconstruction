"""Pseudo-group context gated classification head."""

from __future__ import annotations

import torch
from torch import nn


class GatedContextHead(nn.Module):
    """
    Sample-level gated context fusion.

    alpha_i = sigmoid(W_g [z_i || g_i] + b_g)
    gated_context = alpha_i * g_i
    prediction = C([z_i || gated_context])
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        fused_dim = feature_dim * 2

        self.gate = nn.Linear(fused_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        target_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:

        if target_features.shape != context_features.shape:
            raise ValueError(
                "target_features and context_features must have the same shape"
            )

        # [z_i || g_i]
        gate_input = torch.cat(
            (target_features, context_features),
            dim=1,
        )

        # alpha_i = sigmoid(W_g [z_i || g_i] + b_g)
        alpha = torch.sigmoid(
            self.gate(gate_input)
        )

        # g_tilde_i = alpha_i * g_i
        gated_context = (
            alpha * context_features
        )

        # [z_i || g_tilde_i]
        fused = torch.cat(
            (target_features, gated_context),
            dim=1,
        )

        return self.classifier(fused)
