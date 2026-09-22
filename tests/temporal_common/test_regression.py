"""Golden behavior locked before extracting the shared temporal core."""

from src.temporal_books.p7_transition_ppr import (
    build_transition_graph,
    one_step_scores,
    ppr_monte_carlo_scores,
    recent_seed_items,
    residual_ranking,
)
from src.temporal_common.graph import build_transition_graph_from_users, temporal_groups


HISTORIES = {
    "u1": [(10, "b"), (10, "a"), (20, "c"), (30, "e"), (30, "d")],
    "u2": [(5, "a"), (15, "d"), (25, "c")],
}


def test_amazon_exact_timestamp_graph_golden_contract():
    graph, stats = build_transition_graph(HISTORIES, cutoff=30)
    assert graph == {"a": [("c",), ("d",)], "b": [("c",)], "d": [("c",)]}
    assert stats == {
        "users": 2,
        "source_items": 3,
        "temporal_batch_pairs": 3,
        "source_to_group_links": 4,
    }
    assert recent_seed_items(HISTORIES["u1"], 30, limit=3) == ["c", "a", "b"]


def test_amazon_scores_and_fusion_golden_contract():
    graph, _ = build_transition_graph(HISTORIES, cutoff=30)
    candidates = ["a", "b", "c", "d", "x"]
    assert one_step_scores(["a", "d"], candidates, graph) == {
        "a": 0.0,
        "b": 0.0,
        "c": 0.75,
        "d": 0.25,
        "x": 0.0,
    }
    assert ppr_monte_carlo_scores(
        ["a", "d"],
        candidates,
        graph,
        walks=1000,
        restart_probability=0.15,
        max_steps=64,
        seed=7,
    ) == {"a": 0.077, "b": 0.0, "c": 0.828, "d": 0.095, "x": 0.0}
    assert residual_ranking(
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d"],
        {"a": 0.0, "b": 0.2, "c": 1.0, "d": 0.1},
        alpha=0.8,
    ) == ["c", "a", "b", "d"]


def test_session_groups_merge_by_gap_and_exclude_cutoff_crossing_session():
    events = [(0, "a"), (10, "b"), (310, "c"), (611, "d")]
    assert temporal_groups(events, cutoff=1000, session_gap_seconds=300) == [
        (310, ("a", "b", "c")),
        (611, ("d",)),
    ]
    assert temporal_groups([(500, "a"), (700, "b")], cutoff=600, session_gap_seconds=300) == []


def test_streamed_graph_builder_matches_mapping_builder():
    expected = build_transition_graph(HISTORIES, cutoff=30)
    assert build_transition_graph_from_users(iter(HISTORIES.values()), cutoff=30) == expected
