""" conservative group-level refinement."""

from __future__ import annotations

from collections import defaultdict

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


def _pair_key(a, b):
    return (
        (a, b)
        if a < b
        else (b, a)
    )


class UnionFind:
    def __init__(self, values):
        self.parent = {
            value: value
            for value in values
        }

    def find(self, value):
        parent = self.parent[value]

        if parent != value:
            self.parent[value] = self.find(
                parent
            )

        return self.parent[value]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb

    def components(self):
        groups = defaultdict(list)

        for value in self.parent:
            groups[
                self.find(value)
            ].append(value)

        return groups


def _pair_stats(edges):
    medians = [
        edge["patch_median"]
        for edge in edges
    ]

    aucs = [
        edge["spatial_auc"]
        for edge in edges
    ]

    scores = [
        edge["pair_score"]
        for edge in edges
    ]

    return {
        "score_max": max(scores),
        "median_max": max(medians),
        "auc_max": max(aucs),
        "median_p25": _percentile(
            medians,
            25,
        ),
        "auc_p10": _percentile(
            aucs,
            10,
        ),
    }


def _cross_edges(
    eligible_edges,
    group_a,
    group_b,
):
    group_a = set(group_a)
    group_b = set(group_b)

    cross = []

    for edge in eligible_edges:

        source = edge["source"]
        target = edge["target"]

        if (
            source in group_a
            and target in group_b
        ) or (
            source in group_b
            and target in group_a
        ):
            cross.append(edge)

    return cross


def _coverage(
    cross_edges,
    group_a,
    group_b,
):
    a_connected = set()
    b_connected = set()

    group_a = set(group_a)
    group_b = set(group_b)

    for edge in cross_edges:
        source = edge["source"]
        target = edge["target"]

        if (
            source in group_a
            and target in group_b
        ):
            a_connected.add(source)
            b_connected.add(target)

        elif (
            source in group_b
            and target in group_a
        ):
            b_connected.add(source)
            a_connected.add(target)

    coverage_a = (
        len(a_connected)
        / len(group_a)
    )

    coverage_b = (
        len(b_connected)
        / len(group_b)
    )

    return min(
        coverage_a,
        coverage_b,
    )


def build_candidate_bridges(
    groups,
    eligible_edges,
    config,
):
    """
    Build initial candidate group pairs
    and calibrate merge thresholds.
    """

    group_lookup = {
        image_id: group[0]
        for group in groups
        for image_id in group
    }

    by_group_pair = defaultdict(list)

    for edge in eligible_edges:

        group_a = group_lookup[
            edge["source"]
        ]

        group_b = group_lookup[
            edge["target"]
        ]

        if group_a == group_b:
            continue

        by_group_pair[
            _pair_key(
                group_a,
                group_b,
            )
        ].append(edge)

    original_groups = {
        group[0]: set(group)
        for group in groups
    }

    bridges = []

    for (
        group_a,
        group_b,
    ), edges in by_group_pair.items():

        stats = _pair_stats(edges)

        bridges.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "cross_score_max": (
                    stats["score_max"]
                ),
                "cross_median_max": (
                    stats["median_max"]
                ),
                "cross_auc_max": (
                    stats["auc_max"]
                ),
                "cross_median_p25": (
                    stats["median_p25"]
                ),
                "cross_auc_p10": (
                    stats["auc_p10"]
                ),
                "strict_count": sum(
                    edge["edge_type"]
                    == "strict"
                    for edge in edges
                ),
                "adaptive_count": sum(
                    edge["edge_type"]
                    == "adaptive"
                    for edge in edges
                ),
            }
        )

    if not bridges:
        return (
            original_groups,
            [],
            None,
            None,
        )

    theta_m = _percentile(
        [
            bridge[
                "cross_median_p25"
            ]
            for bridge in bridges
        ],
        config.bridge_threshold_quantile,
    )

    theta_a = _percentile(
        [
            bridge[
                "cross_auc_p10"
            ]
            for bridge in bridges
        ],
        config.bridge_threshold_quantile,
    )

    ordered = sorted(
        bridges,
        key=lambda bridge: (
            -bridge["cross_score_max"],
            -bridge["cross_median_max"],
            -bridge["cross_auc_max"],
            -bridge["strict_count"],
            -bridge["adaptive_count"],
            bridge["group_a"],
            bridge["group_b"],
        ),
    )

    return (
        original_groups,
        ordered,
        theta_m,
        theta_a,
    )


def conservative_group_refinement(
    original_groups,
    bridges,
    eligible_edges,
    theta_m,
    theta_a,
    config,
):
    """
    Greedy refinement using current
    union-find components.
    """

    if not bridges:
        groups = [
            sorted(group)
            for group
            in original_groups.values()
        ]

        return (
            sorted(
                groups,
                key=lambda group: (
                    -len(group),
                    group[0],
                ),
            ),
            0,
        )

    union_find = UnionFind(
        original_groups.keys()
    )

    accepted_merges = 0

    for bridge in bridges:

        root_a = union_find.find(
            bridge["group_a"]
        )

        root_b = union_find.find(
            bridge["group_b"]
        )

        if root_a == root_b:
            continue

        current_a = set().union(
            *[
                group
                for name, group
                in original_groups.items()
                if union_find.find(name)
                == root_a
            ]
        )

        current_b = set().union(
            *[
                group
                for name, group
                in original_groups.items()
                if union_find.find(name)
                == root_b
            ]
        )

        if (
            len(current_a | current_b)
            > config.max_merged_size
        ):
            continue

        cross = _cross_edges(
            eligible_edges,
            current_a,
            current_b,
        )

        if (
            len(cross)
            < config.min_cross_pair_count
        ):
            continue

        stats = _pair_stats(cross)

        if (
            stats["median_p25"]
            < theta_m
        ):
            continue

        if (
            stats["auc_p10"]
            < theta_a
        ):
            continue

        if (
            _coverage(
                cross,
                current_a,
                current_b,
            )
            < config.min_bidirectional_coverage
        ):
            continue

        union_find.union(
            root_a,
            root_b,
        )

        accepted_merges += 1

    final_groups = []

    for group_names in (
        union_find.components().values()
    ):

        merged = set().union(
            *[
                original_groups[name]
                for name
                in group_names
            ]
        )

        final_groups.append(
            sorted(merged)
        )

    final_groups = sorted(
        final_groups,
        key=lambda group: (
            -len(group),
            group[0],
        ),
    )

    return (
        final_groups,
        accepted_merges,
    )
