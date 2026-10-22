""" mutual Top-3 strict seed construction """

from __future__ import annotations


def _connected_components(
    image_ids,
    edges,
):
    adjacency = {
        image_id: set()
        for image_id in image_ids
    }

    for edge in edges:
        a = edge["source"]
        b = edge["target"]

        adjacency[a].add(b)
        adjacency[b].add(a)

    visited = set()
    components = []

    for image_id in sorted(image_ids):

        if image_id in visited:
            continue

        stack = [image_id]
        visited.add(image_id)
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in sorted(
                adjacency[current]
            ):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(
            sorted(component)
        )

    return sorted(
        components,
        key=lambda group: (
            -len(group),
            group[0],
        ),
    )


def construct_strict_seed_groups(
    image_ids,
    neighbors,
    pairs,
    config,
):
    """
    Construct connected components from
    mutual Top-3 strict relations.
    """

    ranks = {
        (i, j): rank
        for i, row in enumerate(neighbors)
        for rank, (j, _) in enumerate(
            row,
            start=1,
        )
    }

    strict_edges = []

    for (i, j), pair in pairs.cache.items():

        strict = (
            pair["patch_median"]
            >= config.strict_median_threshold
            and pair["spatial_auc"]
            >= config.strict_auc_threshold
        )

        mutual_topk = (
            ranks.get((i, j), 10**9)
            <= config.seed_mutual_topk
            and ranks.get((j, i), 10**9)
            <= config.seed_mutual_topk
        )

        if strict and mutual_topk:
            strict_edges.append(
                {
                    "source": image_ids[i],
                    "target": image_ids[j],
                }
            )

    groups = _connected_components(
        image_ids,
        strict_edges,
    )

    return groups, ranks
