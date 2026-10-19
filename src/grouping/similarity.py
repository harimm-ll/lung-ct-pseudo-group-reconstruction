"""Stage 1: same-position DINOv2 patch similarity."""

from __future__ import annotations

import numpy as np
from .common import GroupingConfig, PairSimilarity, average_pool_grid, l2_normalize, _topk_neighbors, _pair_key

def prepare_similarity(features, image_ids, config):
    """Return pair metrics and top-K neighborhoods, without metadata or labels."""
    if len(image_ids) != len(features) or len(set(image_ids)) != len(image_ids):
        raise ValueError("Features and unique image_ids must have matching lengths")
    if len(image_ids) < 2:
        raise ValueError("At least two images required")
    thresholds = np.asarray(config.auc_thresholds, dtype=np.float64)
    if len(thresholds) < 2 or not np.all(np.diff(thresholds) > 0):
        raise ValueError("Invalid AUC thresholds")
    grid = average_pool_grid(np.asarray(features, dtype=np.float32), config.window_size)
    pooled = l2_normalize(grid.reshape(len(image_ids), -1, grid.shape[-1]))
    pairs = PairSimilarity(pooled, grid.shape[1], thresholds)
    neighbors = _topk_neighbors(pooled, config.neighbor_topk, config.chunk_size, image_ids)
    retrieved = {_pair_key(i, j) for i, ns in enumerate(neighbors) for j, _ in ns}
    for i, j in sorted(retrieved):
        pairs.get(i, j)
    return pairs, neighbors

