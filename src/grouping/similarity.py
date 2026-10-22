""" DINOv2 patch similarity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReconstructionConfig:
    # Local feature similarity
    pooling_window: int = 3

    auc_thresholds: tuple[float, ...] = (
        0.90,
        0.92,
        0.94,
        0.95,
        0.96,
        0.965,
        0.97,
        0.98,
    )

    # Neighborhoods
    neighbor_topk: int = 30
    seed_mutual_topk: int = 3
    adaptive_topk: int = 10
    singleton_rescue_topk: int = 3

    # Strict relation
    strict_median_threshold: float = 0.967
    strict_auc_threshold: float = 0.65

    # Adaptive calibration
    global_floor_quantile: float = 10.0
    local_primary_quantile: float = 75.0

    # Preliminary singleton expansion
    expansion_candidate_median: float = 0.965
    expansion_candidate_auc: float = 0.60
    expansion_best_median: float = 0.970
    expansion_best_auc: float = 0.75
    expansion_min_candidate: int = 3
    expansion_min_strong: int = 2
    expansion_min_coverage: float = 0.30

    # Group refinement
    bridge_threshold_quantile: float = 5.0
    max_merged_size: int = 45
    min_cross_pair_count: int = 1
    min_bidirectional_coverage: float = 0.20


def percentile(values, q):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        raise ValueError(
            "Cannot calculate percentile from an empty sequence."
        )

    return float(np.percentile(values, q))


def _pair_key(i: int, j: int):
    return (i, j) if i < j else (j, i)


def average_pool_grid(
    features: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """
    Average pooling with stride 1 and no padding.

    [N, 37, 37, C] -> [N, 35, 35, C]
    for a 3x3 window.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    n, h, w, c = features.shape

    out_h = h - window_size + 1
    out_w = w - window_size + 1

    output = np.zeros(
        (n, out_h, out_w, c),
        dtype=np.float32,
    )

    for a in range(window_size):
        for b in range(window_size):
            output += features[
                :,
                a:a + out_h,
                b:b + out_w,
                :,
            ]

    output /= float(
        window_size * window_size
    )

    return output


def l2_normalize(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:

    norm = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )

    return x / np.maximum(norm, eps)


def _largest_two_component_area(
    binary_map: np.ndarray,
) -> int:

    h, w = binary_map.shape
    visited = np.zeros_like(
        binary_map,
        dtype=bool,
    )

    component_sizes = []

    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for y in range(h):
        for x in range(w):

            if (
                not binary_map[y, x]
                or visited[y, x]
            ):
                continue

            stack = [(y, x)]
            visited[y, x] = True
            size = 0

            while stack:
                cy, cx = stack.pop()
                size += 1

                for dy, dx in directions:
                    ny = cy + dy
                    nx = cx + dx

                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and binary_map[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            component_sizes.append(size)

    component_sizes.sort(reverse=True)

    return sum(component_sizes[:2])


def spatial_auc(
    similarity_map: np.ndarray,
    thresholds,
) -> float:

    thresholds = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    h, w = similarity_map.shape
    denominator = float(h * w)

    coverage = []

    for threshold in thresholds:
        binary_map = (
            similarity_map >= threshold
        )

        area = _largest_two_component_area(
            binary_map
        )

        coverage.append(
            area / denominator
        )

    auc = np.trapezoid(
        coverage,
        thresholds,
    )

    return float(
        auc
        / (
            thresholds[-1]
            - thresholds[0]
        )
    )


class PairSimilarity:
    def __init__(
        self,
        pooled_features: np.ndarray,
        grid_size: int,
        thresholds,
    ):
        self.features = pooled_features
        self.grid_size = grid_size
        self.thresholds = thresholds
        self.cache = {}

    def get(self, i: int, j: int):
        key = _pair_key(i, j)

        if key in self.cache:
            return self.cache[key]

        sim = np.sum(
            self.features[i]
            * self.features[j],
            axis=-1,
        )

        similarity_map = sim.reshape(
            self.grid_size,
            self.grid_size,
        )

        patch_median = float(
            np.median(similarity_map)
        )

        auc = spatial_auc(
            similarity_map,
            self.thresholds,
        )

        pair_score = (
            0.75 * patch_median
            + 0.25 * auc
        )

        result = {
            "patch_median": patch_median,
            "spatial_auc": auc,
            "pair_score": pair_score,
        }

        self.cache[key] = result
        return result


def _topk_neighbors(
    pooled_features: np.ndarray,
    pairs: PairSimilarity,
    topk: int,
):
    """
    Retrieve nearest neighbors according to
    median same-position patch similarity m_ij.
    """

    n = len(pooled_features)
    neighbors = []

    for i in range(n):
        row = []

        for j in range(n):
            if i == j:
                continue

            pair = pairs.get(i, j)

            row.append(
                (
                    j,
                    pair["patch_median"],
                )
            )

        row.sort(
            key=lambda item: (
                -item[1],
                item[0],
            )
        )

        neighbors.append(
            row[:topk]
        )

    return neighbors


def prepare_similarity(
    features,
    image_ids,
    config: ReconstructionConfig,
):
    """
    Compute pooled DINOv2 patch representations,
    nearest-neighbor relationships, and retained
    pairwise similarity statistics.
    """

    if len(image_ids) != len(features):
        raise ValueError(
            "image_ids and features must have matching lengths."
        )

    if len(set(image_ids)) != len(image_ids):
        raise ValueError(
            "image_ids must be unique."
        )

    if len(image_ids) < 2:
        raise ValueError(
            "At least two images are required."
        )

    pooled_grid = average_pool_grid(
        features,
        config.pooling_window,
    )

    n, h, w, c = pooled_grid.shape

    pooled = pooled_grid.reshape(
        n,
        h * w,
        c,
    )

    pooled = l2_normalize(pooled)

    pairs = PairSimilarity(
        pooled,
        h,
        config.auc_thresholds,
    )

    neighbors = _topk_neighbors(
        pooled,
        pairs,
        config.neighbor_topk,
    )

    # Keep only the union of the Top-30 neighborhoods.
    retrieved = {
        _pair_key(i, j)
        for i, row in enumerate(neighbors)
        for j, _ in row
    }

    retained_cache = {
        pair: pairs.cache[pair]
        for pair in retrieved
    }

    pairs.cache = retained_cache

    return pairs, neighbors
