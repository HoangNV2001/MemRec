"""Pure ranking, fusion, and paired-evaluation helpers."""
from __future__ import annotations

import math
import random
from typing import Mapping, Sequence


def event_ndcg_at_5(ranking: Sequence[str], gold_item_id: str) -> float:
    try:
        rank = list(ranking).index(gold_item_id) + 1
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 1) if rank <= 5 else 0.0


def event_hit_at_5(ranking: Sequence[str], gold_item_id: str) -> float:
    return float(gold_item_id in ranking[:5])


def paired_bootstrap_ci(values: Sequence[float], *, resamples: int, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("paired bootstrap requires at least one value")
    if resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def rank_by_scores(candidates: Sequence[str], scores: Mapping[str, float]) -> list[str]:
    order = {item_id: index for index, item_id in enumerate(candidates)}
    return sorted(candidates, key=lambda item_id: (-float(scores.get(item_id, 0.0)), order[item_id]))


def residual_ranking(
    candidates: Sequence[str],
    local: Sequence[str],
    graph_scores: Mapping[str, float],
    *,
    alpha: float,
) -> list[str]:
    values = [float(graph_scores[item_id]) for item_id in candidates]
    minimum, maximum = min(values, default=0.0), max(values, default=0.0)
    if alpha <= 0 or maximum <= minimum:
        return list(local)
    local_scores = {item_id: 1.0 / math.log2(index + 2) for index, item_id in enumerate(local)}
    combined = {
        item_id: local_scores[item_id]
        + alpha * (float(graph_scores[item_id]) - minimum) / (maximum - minimum)
        for item_id in candidates
    }
    return rank_by_scores(candidates, combined)
