"""Prefix-conditioned, target-safe graph input for policy pseudo-episodes.

This is not a global temporal replay: InstructRec Books has no shared clock.
Only source ``train_data`` is accepted; validation/test interactions, original
instructions, and candidate lists are deliberately absent from the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Collection, Mapping, Sequence

from .policy_data import pseudo_target_from_train


@dataclass(frozen=True)
class PrefixSnapshot:
    """Minimal dataset interface consumed by graph/pruner/packer/agent.

    ``train_data`` contains immutable tuples behind a read-only mapping.
    The pseudo-target labels are never stored on this object.
    """

    train_data: Mapping[int, tuple[int, ...]]
    item_metadata: Mapping[int, Mapping[str, Any]] | None = None
    name: str = "instructrec-books"


def make_prefix_snapshot(
    source_train_data: Mapping[int, Sequence[int]],
    eligible_query_user_ids: Collection[int],
    *,
    min_prefix_length: int,
    item_metadata: Mapping[int, Mapping[str, Any]] | None = None,
    name: str = "instructrec-books",
) -> tuple[PrefixSnapshot, dict[int, int], dict[int, int]]:
    """Separate graph history, Stage-W event, and pseudo-target per query user.

    For a query history ``[..., warmup, target]``, graph input is ``[:-2]``,
    warm-up feedback is ``[-2]``, and reward-only pseudo-target is ``[-1]``.
    This mirrors the Books baseline's train/validation/test roles without
    claiming a cross-user clock. Non-query users retain their permitted train
    histories. Caller must start Stage-W from fresh empty storage; importing
    an earlier warm-up/cache can reintroduce an excluded interaction.
    """
    if min_prefix_length < 2:
        raise ValueError("Stage-W parity requires at least one graph-history item")
    query_ids = tuple(int(uid) for uid in eligible_query_user_ids)
    if len(query_ids) != len(set(query_ids)) or not query_ids:
        raise ValueError("Need nonempty, unique query user IDs")
    missing = set(query_ids) - set(source_train_data)
    if missing:
        raise ValueError(f"Query users absent from train_data: {sorted(missing)}")

    histories = {int(uid): tuple(int(item) for item in items)
                 for uid, items in source_train_data.items()}
    if len(histories) != len(source_train_data):
        raise ValueError("Source user IDs collide after integer conversion")
    warmup_events: dict[int, int] = {}
    pseudo_targets: dict[int, int] = {}
    for user_id in query_ids:
        selected = pseudo_target_from_train(
            histories[user_id], min_prefix_length=min_prefix_length,
        )
        if selected is None:
            raise ValueError(f"Query user {user_id} fails locked pseudo-target eligibility")
        prefix, target = selected
        warmup_events[user_id] = prefix[-1]
        histories[user_id] = prefix[:-1]
        pseudo_targets[user_id] = target

    snapshot = PrefixSnapshot(
        train_data=MappingProxyType(histories),
        item_metadata=item_metadata,
        name=name,
    )
    for user_id, target in pseudo_targets.items():
        if target in snapshot.train_data[user_id]:
            raise AssertionError("Pseudo-target survived in its user's snapshot history")
    return snapshot, warmup_events, pseudo_targets


def snapshot_sha256(snapshot: PrefixSnapshot) -> str:
    """Digest all graph-building user histories, not reward labels/metadata."""
    digest = hashlib.sha256()
    for user_id in sorted(snapshot.train_data):
        row = [user_id, list(snapshot.train_data[user_id])]
        digest.update(json.dumps(row, separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
