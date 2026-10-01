"""One-positive iterative ranking rewards; no model or dataset dependencies."""

from __future__ import annotations

from math import log2
from typing import Sequence


def _gain(rank: int) -> float:
    if rank < 1:
        raise ValueError("Rank must be positive")
    return 1.0 / log2(rank + 1)


def ndcg_at_k(rank: int, k: int = 5) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    return _gain(rank) if rank <= k else 0.0


def _validate_trajectory(
    candidate_labels: Sequence[str], exclusions: Sequence[str],
    survivor: str, target_label: str,
) -> None:
    labels = tuple(candidate_labels)
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("Need at least two unique candidate labels")
    if target_label not in labels or survivor not in labels:
        raise ValueError("Target and survivor must be candidates")
    if len(exclusions) != len(labels) - 1:
        raise ValueError("A valid trajectory requires N-1 exclusions")
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("Duplicate exclusion")
    if set(exclusions) | {survivor} != set(labels) or survivor in exclusions:
        raise ValueError("Exclusions and survivor must partition candidates")


def target_rank(
    candidate_labels: Sequence[str], exclusions: Sequence[str],
    survivor: str, target_label: str,
) -> int:
    _validate_trajectory(candidate_labels, exclusions, survivor, target_label)
    return 1 if survivor == target_label else len(candidate_labels) - exclusions.index(target_label)


def mpss_rewards(
    candidate_labels: Sequence[str], exclusions: Sequence[str],
    survivor: str, target_label: str, k: int = 5,
) -> tuple[tuple[float, ...], int]:
    """Dense survival shaping with exact *undiscounted* NDCG@k return.

    The final metric correction is essential.  PPO training must use gamma=1
    (and verify its token/step reward placement) if it claims that the RL
    objective matches the undiscounted episode metric exactly.  This shaping
    does not change the optimal expected-return objective by itself.
    """
    _validate_trajectory(candidate_labels, exclusions, survivor, target_label)
    active = set(candidate_labels)
    phi = _gain(len(active))
    rewards: list[float] = []
    eliminated_rank: int | None = None
    for excluded in exclusions:
        size = len(active)
        if excluded not in active:
            raise ValueError("Action is not in active set")
        if eliminated_rank is not None:
            next_phi = phi
        elif excluded == target_label:
            eliminated_rank = size
            next_phi = _gain(size)
        else:
            next_phi = _gain(size - 1)
        rewards.append(next_phi - phi)
        phi = next_phi
        active.remove(excluded)
    if active != {survivor}:
        raise ValueError("Final survivor does not match trajectory")
    rank = 1 if eliminated_rank is None else eliminated_rank
    correction = ndcg_at_k(rank, k) - sum(rewards)
    rewards[-1] += correction
    if abs(sum(rewards) - ndcg_at_k(rank, k)) >= 1e-12:
        raise AssertionError("MPSS return differs from NDCG")
    return tuple(rewards), rank


def r1_binary_rewards(
    candidate_labels: Sequence[str], exclusions: Sequence[str],
    survivor: str, target_label: str,
) -> tuple[float, ...]:
    """R1-style negative-exclusion baseline, not the project reward."""
    _validate_trajectory(candidate_labels, exclusions, survivor, target_label)
    return tuple(0.0 if label == target_label else 1.0 for label in exclusions)


def terminal_ndcg_rewards(
    candidate_labels: Sequence[str], exclusions: Sequence[str],
    survivor: str, target_label: str, k: int = 5,
) -> tuple[float, ...]:
    """Terminal-only control with the same undiscounted episode return."""
    rank = target_rank(candidate_labels, exclusions, survivor, target_label)
    return (0.0,) * (len(exclusions) - 1) + (ndcg_at_k(rank, k),)
