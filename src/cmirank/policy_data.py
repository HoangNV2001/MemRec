"""Target-safe draft policy split and pre-test pseudo-target eligibility.

The functions in this module never read ``test_data`` or original test
candidate lists.  They do not build a memory snapshot; eligibility alone is
not proof that an RL episode is leakage-free.
"""

from __future__ import annotations

import hashlib
from typing import Collection, Sequence


NEUTRAL_PSEUDO_INSTRUCTION = (
    "Rank the candidate books by how likely they are to match the user's "
    "current preferences."
)


def draft_policy_split(
    dev_user_ids: Collection[int], exposed_integration_ids: Collection[int],
    *, seed: str, train_size: int = 1500,
) -> dict[str, list[int]]:
    """Hash-partition development users, excluding the exposed integration set.

    A returned split is a *draft* until the researcher approves its manifest.
    No final test target, candidate list, or outcome enters the hash ordering.
    """
    if not seed:
        raise ValueError("An explicit seed is required")
    dev = set(int(uid) for uid in dev_user_ids)
    exposed = set(int(uid) for uid in exposed_integration_ids)
    if len(dev) != len(dev_user_ids) or len(exposed) != len(exposed_integration_ids):
        raise ValueError("Duplicate user ID")
    if not exposed or not exposed.issubset(dev):
        raise ValueError("Exposed integration users must be a nonempty dev subset")
    available = dev - exposed
    if not 0 < train_size < len(available):
        raise ValueError("Train size must leave policy-validation users")
    ordered = sorted(
        available,
        key=lambda uid: (hashlib.sha256(f"{seed}:{uid}".encode("ascii")).digest(), uid),
    )
    return {
        "policy_train": sorted(ordered[:train_size]),
        "policy_val": sorted(ordered[train_size:]),
        "integration_exposed": sorted(exposed),
    }


def pseudo_target_from_train(
    train_sequence: Sequence[int], *, min_prefix_length: int,
) -> tuple[tuple[int, ...], int] | None:
    """Choose last *train* item as pseudo-target, never validation/test item.

    Require the pseudo-target to be novel to this user's prefix, matching the
    original Books evaluation's no-history-collision candidate contract.
    """
    if min_prefix_length < 1:
        raise ValueError("min_prefix_length must be positive")
    if len(train_sequence) < min_prefix_length + 1:
        return None
    prefix = tuple(int(item) for item in train_sequence[:-1])
    target = int(train_sequence[-1])
    if target in prefix:
        return None
    return prefix, target
