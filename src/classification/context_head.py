from __future__ import annotations

import torch
from torch import nn


class GatedContextHead(nn.Module):
    #拼接逻辑

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
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if target_features.shape != context_features.shape:
            raise ValueError("target_features and context_features must have equal shape")
        gate_input = torch.cat((target_features, context_features), dim=1)
        alpha = torch.sigmoid(self.gate(gate_input))
        fused = torch.cat((target_features, alpha * context_features), dim=1)
        logits = self.classifier(fused)
        return (logits, alpha) if return_gate else logits

