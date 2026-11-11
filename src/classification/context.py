from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .schema import ImageRecord


def build_group_membership(
    records: Iterable[ImageRecord],
) -> dict[str, tuple[ImageRecord, ...]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    return {group_id: tuple(members) for group_id, members in groups.items()}


def leave_one_out_members(
    target: ImageRecord,
    group_members: dict[str, tuple[ImageRecord, ...]],
) -> tuple[ImageRecord, ...]:
    #返回组内其他，除去自身
    if target.group_id not in group_members:
        raise KeyError(f"Unknown pseudo-group: {target.group_id}")
    return tuple(
        record
        for record in group_members[target.group_id]
        if record.image_id != target.image_id
    )

