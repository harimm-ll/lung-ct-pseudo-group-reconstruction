""" adaptive candidate relation selection """

from __future__ import annotations

import numpy as np


def _percentile(values, q):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot calculate percentile from empty values."
        )

    return float(
        np.percentile(values, q)
    )


def compute_adaptive_edges(
    image_ids,
    neighbors,
    ranks,
    pairs,
    remaining_singletons,
    config,
):
    """
    Construct strict, adaptive, and
    singleton-rescue relations.
    """

    index = {
        image_id: i
        for i, image_id
        in enumerate(image_ids)
    }

    remaining = {
        index[image_id]
        for image_id
        in remaining_singletons
    }

    calibration_indices = (
        sorted(remaining)
        if remaining
        else list(range(len(image_ids)))
    )

    top1 = [
        row[0][0]
        for row in neighbors
    ]

    primary_floor = _percentile(
        [
            pairs.get(
                i,
                top1[i],
            )["patch_median"]
            for i in calibration_indices
        ],
        config.global_floor_quantile,
    )

    auc_floor = _percentile(
        [
            pairs.get(
                i,
                top1[i],
            )["spatial_auc"]
            for i in calibration_indices
        ],
        config.global_floor_quantile,
    )

    local_thresholds = []
    local_pass = {}

    for i, row in enumerate(neighbors):

        local_median_threshold = _percentile(
            [
                similarity
                for _, similarity
                in row
            ],
            config.local_primary_quantile,
        )

        threshold = max(
            primary_floor,
            local_median_threshold,
        )

        local_thresholds.append(
            threshold
        )

        for j, _ in row:

            pair = pairs.get(i, j)

            local_pass[(i, j)] = (
                ranks[(i, j)]
                <= config.adaptive_topk
                and pair["patch_median"]
                >= threshold
                and pair["spatial_auc"]
                >= auc_floor
            )

    eligible = []

    for i, j in sorted(
        pairs.cache.keys()
    ):

        pair = pairs.get(i, j)

        strict = (
            pair["patch_median"]
            >= config.strict_median_threshold
            and pair["spatial_auc"]
            >= config.strict_auc_threshold
        )

        adaptive = (
            local_pass.get(
                (i, j),
                False,
            )
            and local_pass.get(
                (j, i),
                False,
            )
        )

        rescue_i = (
            i in remaining
            and ranks.get(
                (i, j),
                10**9,
            )
            <= config.singleton_rescue_topk
            and local_pass.get(
                (i, j),
                False,
            )
        )

        rescue_j = (
            j in remaining
            and ranks.get(
                (j, i),
                10**9,
            )
            <= config.singleton_rescue_topk
            and local_pass.get(
                (j, i),
                False,
            )
        )

        if strict:
            edge_type = "strict"

        elif adaptive:
            edge_type = "adaptive"

        elif rescue_i or rescue_j:
            edge_type = "singleton_rescue"

        else:
            continue

        eligible.append(
            {
                "source": image_ids[i],
                "target": image_ids[j],
                "edge_type": edge_type,
                **pair,
            }
        )

    calibration = {
        "global_median_floor": (
            primary_floor
        ),
        "global_spatial_auc_floor": (
            auc_floor
        ),
        "anchor_threshold_min": float(
            np.min(local_thresholds)
        ),
        "anchor_threshold_max": float(
            np.max(local_thresholds)
        ),
        "anchor_threshold_median": float(
            np.median(local_thresholds)
        ),
    }

    return eligible, calibration
