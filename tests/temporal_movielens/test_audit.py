from collections import Counter

import numpy as np

from src.temporal_movielens.audit import (
    counter_quantiles,
    eligible_events_for_user,
    session_sizes,
    timestamp_at_quantile,
)


def test_timestamp_quantile_returns_boundary_and_keeps_ties_later():
    values = np.asarray([1, 1, 2, 3, 3, 3, 4, 5, 6, 7], dtype=np.int64)
    cutoff = timestamp_at_quantile(values, 0.8)
    assert cutoff == 5
    assert int(np.sum(values < cutoff)) == 7


def test_session_sizes_group_consecutive_gaps():
    events = [(0, 1, 4.0), (10, 2, 3.0), (310, 3, 5.0), (611, 4, 4.0)]
    assert session_sizes(events, 300) == [3, 1]
    assert session_sizes(events, 0) == [1, 1, 1, 1]


def test_singleton_session_eligibility_is_strict_past_and_partitioned():
    history = [(index, index, 4.0) for index in range(1, 6)]
    events = [
        *history,
        (1000, 10, 4.5),
        (1001, 11, 5.0),
        (2000, 20, 4.0),
        (2000, 21, 4.0),
        (3000, 30, 4.0),
    ]
    counts = eligible_events_for_user(
        events,
        candidate_pool={10, 11, 20, 21, 30},
        train_cutoff=900,
        validation_cutoff=2500,
        session_gap=300,
        positive_rating_min=4.0,
        minimum_positive_history=5,
    )
    # 1000 and 1001 share a five-minute session, while the 2000 target is an
    # exact-timestamp multi-item batch. Only the isolated 3000 target survives.
    assert counts == {"development": 0, "test": 1}


def test_counter_quantiles_do_not_expand_session_histogram():
    values = counter_quantiles(Counter({1: 90, 10: 10}))
    assert values["0.5"] == 1
    assert values["0.95"] == 10
