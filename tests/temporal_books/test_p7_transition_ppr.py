from src.temporal_books.p7_transition_ppr import (
    build_transition_graph,
    one_step_scores,
    ppr_monte_carlo_scores,
    recent_seed_items,
    residual_ranking,
    timestamp_batches,
)


def test_timestamp_batches_do_not_order_same_timestamp_items():
    events = [(10, "b"), (10, "a"), (20, "c")]
    assert timestamp_batches(events) == [(10, ("a", "b")), (20, ("c",))]
    graph, stats = build_transition_graph({"u": events}, cutoff=30)
    assert graph["a"] == [("c",)] and graph["b"] == [("c",)]
    assert "c" not in graph
    assert stats["temporal_batch_pairs"] == 1


def test_recent_seeds_are_strict_past_and_unique():
    events = [(1, "a"), (2, "b"), (3, "a"), (4, "future")]
    assert recent_seed_items(events, 4, limit=3) == ["a", "b"]


def test_ppr_reaches_multihop_candidate_deterministically():
    graph = {"a": [("b",)], "b": [("c",)]}
    candidates = ["c", "x"]
    assert one_step_scores(["a"], candidates, graph)["c"] == 0
    first = ppr_monte_carlo_scores(
        ["a"], candidates, graph, walks=5000, restart_probability=0.15, max_steps=64, seed=7
    )
    second = ppr_monte_carlo_scores(
        ["a"], candidates, graph, walks=5000, restart_probability=0.15, max_steps=64, seed=7
    )
    assert first == second
    assert first["c"] > 0 and first["x"] == 0


def test_residual_transition_score_can_promote_candidate():
    ranking = residual_ranking(["a", "b", "c"], ["a", "b", "c"], {"a": 0.0, "b": 1.0, "c": 0.0}, alpha=1.0)
    assert ranking[0] == "b"
