"""Stage 2: mutual-top-3 strict seed groups."""

from __future__ import annotations

import numpy as np
from .common import _connected_components

def construct_strict_seed_groups(image_ids, neighbors, pairs, config):
    ranks = {(i,j): r for i, ns in enumerate(neighbors) for r,(j,_) in enumerate(ns,1)}
    strict_edges = []
    for (i,j), p in pairs.cache.items():
        if (p["patch_median"] >= config.strict_median_threshold
            and p["spatial_auc"] >= config.strict_auc_threshold
            and ranks.get((i,j), 10**9) <= config.seed_mutual_topk
            and ranks.get((j,i), 10**9) <= config.seed_mutual_topk):
            strict_edges.append({"source":image_ids[i],"target":image_ids[j]})
    return _connected_components(image_ids, strict_edges), ranks


