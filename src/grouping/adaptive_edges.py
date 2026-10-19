"""Stage 4: adaptive edge selection."""

from __future__ import annotations

import numpy as np
from .common import percentile

def compute_adaptive_edges(image_ids,neighbors,ranks,pairs,seed_singletons,config):
    indices={id_:i for i,id_ in enumerate(image_ids)}
    remaining={indices[i] for i in seed_singletons}
    floor_indices=sorted(remaining) or list(range(len(image_ids)))
    top1=[row[0][0] for row in neighbors]
    primary_floor=percentile([pairs.get(i,top1[i])["patch_median"] for i in floor_indices],config.global_floor_quantile)
    auc_floor=percentile([pairs.get(i,top1[i])["spatial_auc"] for i in floor_indices],config.global_floor_quantile)
    local=[]
    local_pass={}
    for i,row in enumerate(neighbors):
        threshold=max(primary_floor,percentile([v for _,v in row],config.local_primary_quantile))
        local.append(threshold)
        for j,_ in row:
            p=pairs.get(i,j)
            local_pass[(i,j)]=(ranks[(i,j)]<=config.local_pass_topk and p["patch_median"]>=threshold and p["spatial_auc"]>=auc_floor)
    eligible=[]
    for i,j in sorted(pairs.cache):
        # Only pairs retrieved in at least one top-30 neighborhood are candidate relations.
        if (i,j) not in ranks and (j,i) not in ranks:
            continue
        p=pairs.get(i,j)
        strict=p["patch_median"]>=config.strict_median_threshold and p["spatial_auc"]>=config.strict_auc_threshold
        adaptive=local_pass.get((i,j),False) and local_pass.get((j,i),False)
        rescue=((i in remaining and ranks.get((i,j),10**9)<=config.singleton_rescue_topk and local_pass.get((i,j),False))
                or (j in remaining and ranks.get((j,i),10**9)<=config.singleton_rescue_topk and local_pass.get((j,i),False)))
        kind="strict_seed" if strict else "adaptive_near_miss" if adaptive else "singleton_rescue" if rescue else None
        if kind:
            eligible.append({"source":image_ids[i],"target":image_ids[j],"edge_type":kind,**p})
    return eligible, {"global_primary_floor":primary_floor,"global_secondary_floor":auc_floor}


