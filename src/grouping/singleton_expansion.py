"""Stage 3: label-free preliminary singleton expansion."""

from __future__ import annotations

import numpy as np
from .common import PairSimilarity

def expand_singletons(seed_groups, image_ids, pairs, config):
    """Preliminary singleton expansion; no diagnostic labels or class metadata.

    Candidate support is computed against every member of the *original*
    non-singleton seed group. Singleton attachments do not create targets.
    """
    index = {id_:i for i,id_ in enumerate(image_ids)}
    targets = [(g[0], list(g)) for g in seed_groups if len(g)>1]
    singletons = sorted(g[0] for g in seed_groups if len(g)==1)
    members = {g[0]:list(g) for g in seed_groups}
    accepted=[]
    for s in singletons:
        si=index[s]
        eligible=[]
        for gid, original in targets:
            connections = [(j,pairs.get(si,index[j])) for j in original]
            supporting=[(j,p) for j,p in connections if p["patch_median"]>=config.expansion_candidate_median and p["spatial_auc"]>=config.expansion_candidate_auc]
            strong=sum(p["patch_median"]>=config.strict_median_threshold and p["spatial_auc"]>=config.strict_auc_threshold for _,p in connections)
            # Best connecting pair is selected by the composite pair score.
            ordered=sorted(connections, key=lambda e:(-e[1]["pair_score"],e[0]))
            best=ordered[0][1]
            if (best["patch_median"]>=config.expansion_best_median
                and best["spatial_auc"]>=config.expansion_best_auc
                and len(supporting)>=config.expansion_min_candidate
                and strong>=config.expansion_min_strong
                and len(supporting)/len(original)>=config.expansion_min_coverage):
                mean_top3=float(np.mean([p["pair_score"] for _,p in ordered[:3]]))
                eligible.append(((-best["pair_score"],-strong,-len(supporting),-mean_top3,gid),gid))
        if eligible:
            _, gid=min(eligible)
            members[gid].append(s)
            del members[s]
            accepted.append((s,gid))
    groups=sorted((sorted(g) for g in members.values()), key=lambda g:(-len(g),g[0]))
    remaining={g[0] for g in groups if len(g)==1}
    return groups,remaining,accepted


