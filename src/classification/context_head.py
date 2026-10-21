""" pseudo-group context construction and gated classification head."""

from __future__ import annotations

import torch
from torch import nn


def leave_one_out_context(
    features: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if group_ids.ndim != 1 or group_ids.shape[0] != features.shape[0]:
        raise ValueError("group_ids must have shape [N]")

    context = torch.zeros_like(features)
    for group_id in torch.unique(group_ids):
        indices = torch.where(group_ids == group_id)[0]
        if indices.numel() > 1:
            group_features = features[indices]
            group_sum = group_features.sum(dim=0, keepdim=True)
            context[indices] = (group_sum - group_features) / (indices.numel() - 1)
    return context


class GatedContextHead(nn.Module):
 

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 256,
        dropout: float = 0.3,
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
            raise ValueError("target_features and context_features must have the same shape")

       
        context_features = context_features.detach()
        gate_input = torch.cat((target_features, context_features), dim=1)
        alpha = torch.sigmoid(self.gate(gate_input))
        gated_context = alpha * context_features
        fused = torch.cat((target_features, gated_context), dim=1)
        return self.classifier(fused)
