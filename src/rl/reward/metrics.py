"""
Ranking metrics for the reward (RL_PLAN.md §5.1).

Single-relevant-item setting throughout: each user has exactly one gold item
among ``n`` candidates, so these reduce to functions of the gold's rank. They are
written to match ``src/train/metrics.py`` in the original pipeline
(``1 / log2(pos + 2)`` for a hit, 0 beyond k) so RL-side numbers are directly
comparable with the M0 baseline table.
"""
from typing import Optional, Sequence

import math


def rank_of(ranking: Sequence[int], gold_item_id: int) -> Optional[int]:
    """
    0-indexed position of the gold item, or None if it is absent.

    Absent means the ranker dropped it -- a bug or a malformed generation, never
    a legitimate "ranked last". Callers must distinguish the two.
    """
    for i, item in enumerate(ranking):
        if item == gold_item_id:
            return i
    return None


def hit_at_k(ranking: Sequence[int], gold_item_id: int, k: int) -> float:
    """1.0 if the gold item is in the top-k, else 0.0."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    pos = rank_of(ranking, gold_item_id)
    return 1.0 if pos is not None and pos < k else 0.0


def ndcg_at_k(ranking: Sequence[int], gold_item_id: int, k: int) -> float:
    """
    NDCG@k with one relevant item of gain 1.

    IDCG is 1 (the ideal puts the gold first), so this is just the discount at
    the gold's position, or 0 past the cutoff.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    pos = rank_of(ranking, gold_item_id)
    if pos is None or pos >= k:
        return 0.0
    return 1.0 / math.log2(pos + 2)


def ndcg_from_rank(pos: Optional[int], k: int) -> float:
    """``ndcg_at_k`` when the rank is already known. Avoids re-scanning."""
    if pos is None or pos >= k:
        return 0.0
    return 1.0 / math.log2(pos + 2)
