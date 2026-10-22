""" preliminary singleton expansion """

from __future__ import annotations

import numpy as np


def expand_singletons(
    seed_groups,
    image_ids,
    pairs,
    config,
):
    """
    Attach eligible singleton seeds to
    existing non-singleton strict seed groups.
    """

    index = {
        image_id: i
        for i, image_id
        in enumerate(image_ids)
    }

    target_groups = [
        (group[0], list(group))
        for group in seed_groups
        if len(group) > 1
    ]

    singletons = sorted(
        group[0]
        for group in seed_groups
        if len(group) == 1
    )

    members = {
        group[0]: list(group)
        for group in seed_groups
    }

    accepted = []

    for singleton in singletons:

        i = index[singleton]
        eligible = []

        for group_id, original_group in target_groups:

            connections = [
                (
                    member,
                    pairs.get(
                        i,
                        index[member],
                    ),
                )
                for member in original_group
            ]

            candidate_support = [
                (member, pair)
                for member, pair in connections
                if (
                    pair["patch_median"]
                    >= config.expansion_candidate_median
                    and pair["spatial_auc"]
                    >= config.expansion_candidate_auc
                )
            ]

            strong_support = sum(
                (
                    pair["patch_median"]
                    >= config.strict_median_threshold
                    and pair["spatial_auc"]
                    >= config.strict_auc_threshold
                )
                for _, pair in connections
            )

            ordered = sorted(
                connections,
                key=lambda item: (
                    -item[1]["pair_score"],
                    item[0],
                ),
            )

            best = ordered[0][1]

            coverage = (
                len(candidate_support)
                / len(original_group)
            )

            valid = (
                best["patch_median"]
                >= config.expansion_best_median
                and best["spatial_auc"]
                >= config.expansion_best_auc
                and len(candidate_support)
                >= config.expansion_min_candidate
                and strong_support
                >= config.expansion_min_strong
                and coverage
                >= config.expansion_min_coverage
            )

            if valid:

                mean_top3 = float(
                    np.mean(
                        [
                            pair["pair_score"]
                            for _, pair
                            in ordered[:3]
                        ]
                    )
                )

                key = (
                    -best["pair_score"],
                    -strong_support,
                    -len(candidate_support),
                    -mean_top3,
                    group_id,
                )

                eligible.append(
                    (key, group_id)
                )

        if eligible:

            _, target_group = min(
                eligible
            )

            members[target_group].append(
                singleton
            )

            del members[singleton]

            accepted.append(
                (singleton, target_group)
            )

    groups = sorted(
        (
            sorted(group)
            for group
            in members.values()
        ),
        key=lambda group: (
            -len(group),
            group[0],
        ),
    )

    remaining_singletons = {
        group[0]
        for group in groups
        if len(group) == 1
    }

    return (
        groups,
        remaining_singletons,
        accepted,
    )
