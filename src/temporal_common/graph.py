"""Pure temporal transition-graph and Monte Carlo scoring primitives."""
from __future__ import annotations

import bisect
import itertools
import random
from collections import defaultdict
from typing import DefaultDict, Dict, Mapping, Sequence


TransitionEvent = tuple[int, str]
TransitionGraph = Dict[str, list[tuple[str, ...]]]
TemporalGroup = tuple[int, tuple[str, ...]]


def timestamp_batches(
    events: Sequence[TransitionEvent], *, cutoff: int | None = None
) -> list[TemporalGroup]:
    """Group equal timestamps without imposing an order inside a batch."""
    limit = len(events) if cutoff is None else bisect.bisect_left(events, (cutoff, ""))
    batches: list[TemporalGroup] = []
    for timestamp, group in itertools.groupby(events[:limit], key=lambda event: event[0]):
        items = tuple(sorted({item_id for _, item_id in group}))
        if items:
            batches.append((timestamp, items))
    return batches


def temporal_groups(
    events: Sequence[TransitionEvent], *, cutoff: int, session_gap_seconds: int = 0
) -> list[TemporalGroup]:
    """Build exact batches or complete gap-bounded sessions before a cutoff.

    For session views, grouping happens before cutoff filtering. A session that
    crosses the cutoff is therefore excluded in full instead of being leaked as
    a truncated pre-cutoff session.
    """
    if session_gap_seconds < 0:
        raise ValueError("session gap cannot be negative")
    if session_gap_seconds == 0:
        return timestamp_batches(events, cutoff=cutoff)
    batches = timestamp_batches(events)
    if not batches:
        return []
    groups: list[TemporalGroup] = []
    end_timestamp, item_values = batches[0]
    items = set(item_values)
    for timestamp, values in batches[1:]:
        if timestamp - end_timestamp <= session_gap_seconds:
            items.update(values)
            end_timestamp = timestamp
        else:
            groups.append((end_timestamp, tuple(sorted(items))))
            end_timestamp, items = timestamp, set(values)
    groups.append((end_timestamp, tuple(sorted(items))))
    return [group for group in groups if group[0] < cutoff]


def build_transition_graph(
    histories: Mapping[str, Sequence[TransitionEvent]],
    *,
    cutoff: int,
    session_gap_seconds: int = 0,
) -> tuple[TransitionGraph, Dict[str, int]]:
    adjacency: DefaultDict[str, list[tuple[str, ...]]] = defaultdict(list)
    temporal_pairs = 0
    source_group_links = 0
    for events in histories.values():
        groups = temporal_groups(events, cutoff=cutoff, session_gap_seconds=session_gap_seconds)
        for (_, sources), (_, destinations) in zip(groups, groups[1:]):
            temporal_pairs += 1
            for source in sources:
                adjacency[source].append(destinations)
                source_group_links += 1
    return dict(adjacency), {
        "users": len(histories),
        "source_items": len(adjacency),
        "temporal_batch_pairs": temporal_pairs,
        "source_to_group_links": source_group_links,
    }


def recent_seed_items(events: Sequence[TransitionEvent], target_time: int, *, limit: int) -> list[str]:
    end = bisect.bisect_left(events, (target_time, ""))
    seeds: list[str] = []
    seen: set[str] = set()
    for _, item_id in reversed(events[:end]):
        if item_id not in seen:
            seen.add(item_id)
            seeds.append(item_id)
        if len(seeds) >= limit:
            break
    return seeds


def one_step_scores(
    seeds: Sequence[str],
    candidates: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, ...]]],
) -> Dict[str, float]:
    scores = {item_id: 0.0 for item_id in candidates}
    candidate_set = set(candidates)
    if not seeds:
        return scores
    seed_weight = 1.0 / len(seeds)
    for source in seeds:
        groups = adjacency.get(source, ())
        if not groups:
            continue
        group_weight = seed_weight / len(groups)
        for destinations in groups:
            destination_weight = group_weight / len(destinations)
            for item_id in destinations:
                if item_id in candidate_set:
                    scores[item_id] += destination_weight
    return scores


def ppr_monte_carlo_scores(
    seeds: Sequence[str],
    candidates: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, ...]]],
    *,
    walks: int,
    restart_probability: float,
    max_steps: int,
    seed: int,
) -> Dict[str, float]:
    if walks < 1 or max_steps < 1 or not 0 < restart_probability < 1:
        raise ValueError("invalid PPR Monte Carlo contract")
    counts = {item_id: 0 for item_id in candidates}
    candidate_set = set(candidates)
    if not seeds:
        return {item_id: 0.0 for item_id in candidates}
    rng = random.Random(seed)
    for _ in range(walks):
        current = seeds[rng.randrange(len(seeds))]
        for _ in range(max_steps):
            if rng.random() < restart_probability:
                break
            groups = adjacency.get(current, ())
            if not groups:
                break
            destinations = groups[rng.randrange(len(groups))]
            current = destinations[rng.randrange(len(destinations))]
        if current in candidate_set:
            counts[current] += 1
    return {item_id: count / walks for item_id, count in counts.items()}
