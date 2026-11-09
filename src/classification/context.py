"""
对每一个target slice，选择同一个split，同一个pseudo-group的其他slices
排除target本身
将剩余的所有slices特征均值作为context
确保不会垮train/val/test获取内容

分类模型需要调用该模块
返回target特征和上下文特征-> 融合分类头

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LeaveOneOutContext:
    target_features: np.ndarray
    context_features: np.ndarray
    context_available: np.ndarray
    context_indices: tuple[np.ndarray, ...]


def select_leave_one_out_context_indices(
    pseudo_group_ids: Sequence[str],
    split_ids: Sequence[str],
) -> tuple[np.ndarray, ...]:
    """Select same-split, same-group context slices excluding each target."""
    if len(pseudo_group_ids) != len(split_ids):
        raise ValueError("pseudo_group_ids and split_ids must have equal length")

    group_members: dict[tuple[str, str], list[int]] = defaultdict(list)
    for slice_index, (split_id, pseudo_group_id) in enumerate(
        zip(split_ids, pseudo_group_ids)
    ):
        group_members[(str(split_id), str(pseudo_group_id))].append(slice_index)

    selected: list[np.ndarray] = []
    for target_index, (split_id, pseudo_group_id) in enumerate(
        zip(split_ids, pseudo_group_ids)
    ):
        same_group = group_members[(str(split_id), str(pseudo_group_id))]
        context_for_target = [
            slice_index
            for slice_index in same_group
            if slice_index != target_index  # leave the target slice out
        ]
        selected.append(np.asarray(context_for_target, dtype=np.int64))

    return tuple(selected)


def build_leave_one_out_context(
    target_features: np.ndarray,
    pseudo_group_ids: Sequence[str],
    split_ids: Sequence[str],
) -> LeaveOneOutContext:
    """Mean-pool the leave-one-out context selected for every target slice."""
    if target_features.ndim != 2:
        raise ValueError("target_features must have shape [num_slices, feature_dim]")
    if len(target_features) != len(pseudo_group_ids):
        raise ValueError("one pseudo-group ID is required per target slice")

    context_indices = select_leave_one_out_context_indices(
        pseudo_group_ids=pseudo_group_ids,
        split_ids=split_ids,
    )
    context_features = np.zeros_like(target_features, dtype=np.float32)
    context_available = np.zeros((len(target_features), 1), dtype=np.float32)

    for target_index, selected_indices in enumerate(context_indices):
        if selected_indices.size > 0:
            context_features[target_index] = target_features[selected_indices].mean(axis=0)
            context_available[target_index, 0] = 1.0

    return LeaveOneOutContext(
        target_features=target_features.astype(np.float32, copy=False),
        context_features=context_features,
        context_available=context_available,
        context_indices=context_indices,
    )
