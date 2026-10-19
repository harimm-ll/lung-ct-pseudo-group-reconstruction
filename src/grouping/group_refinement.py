"""Stage 5: cohesion-constrained greedy refinement."""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from .common import UnionFind, percentile, _pair_stats, _coverage, _cross_edges, _pair_key

def build_candidate_bridges(groups,eligible,config):
    lookup={item:g[0] for g in groups for item in g}
    by_group=defaultdict(list)
    for edge in eligible:
        ga,gb=lookup[edge["source"]],lookup[edge["target"]]
        if ga!=gb: by_group[_pair_key(ga,gb)].append(edge)
    original={g[0]:set(g) for g in groups}
    bridges=[]
    for (ga,gb),edges in by_group.items():
        stats=_pair_stats(edges)
        bridges.append({"group_a":ga,"group_b":gb,"cross_score_max":stats["score_max"],
                        "cross_median_max":stats["median_max"],"cross_auc_max":stats["auc_max"],
                        "cross_median_p25":stats["median_p25"],"cross_auc_p10":stats["auc_p10"],
                        "strict_edge_count":sum(e["edge_type"]=="strict_seed" for e in edges),
                        "adaptive_edge_count":sum(e["edge_type"]=="adaptive_near_miss" for e in edges)})
    theta_m=percentile([r["cross_median_p25"] for r in bridges],config.bridge_threshold_quantile)
    theta_a=percentile([r["cross_auc_p10"] for r in bridges],config.bridge_threshold_quantile)
    ordered=sorted(bridges,key=lambda r:(-r["cross_score_max"],-r["cross_median_max"],-r["cross_auc_max"],
                                     -r["strict_edge_count"],-r["adaptive_edge_count"],r["group_a"],r["group_b"]))
    return original,ordered,theta_m,theta_a


def conservative_group_refinement(original,bridges,eligible,theta_m,theta_a,config):
    uf=UnionFind(set(original))
    accepted=0
    for row in bridges:
        ra,rb=uf.find(row["group_a"]),uf.find(row["group_b"])
        if ra==rb:continue
        ga=set().union(*(g for name,g in original.items() if uf.find(name)==ra))
        gb=set().union(*(g for name,g in original.items() if uf.find(name)==rb))
        if len(ga|gb)>config.max_merged_size:continue
        cross=_cross_edges(eligible,ga,gb)
        if len(cross)<config.min_cross_pair_count:continue
        stats=_pair_stats(cross)
        if (stats["median_p25"]<theta_m or stats["auc_p10"]<theta_a
            or _coverage(cross,ga,gb)<config.min_bidirectional_coverage):continue
        uf.union(ra,rb)
        accepted+=1
    result=[]
    for names in uf.components().values():
        result.append(sorted(set().union(*(original[n] for n in names))))
    return sorted(result,key=lambda g:(-len(g),g[0])),accepted


