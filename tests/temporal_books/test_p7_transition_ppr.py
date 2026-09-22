import json

import pytest

from src.temporal_books.p7_selfhost_local import (
    complete_permutation,
    configured_smoke_events,
    permutation_repairs,
)
from src.temporal_books.current_support import smoke_events
from src.temporal_books.p7_transition_ppr import (
    build_transition_graph,
    one_step_scores,
    ppr_monte_carlo_scores,
    recent_seed_items,
    residual_ranking,
    smoke_subsets,
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


def test_p7v2_duplicate_label_is_completed_deterministically():
    candidates = [f"item-{label}" for label in "ABCDEFGHIJ"]
    ranking = complete_permutation(
        {"ranking": ["A", "A", "B", "C", "D", "E", "F", "G", "H", "I"]},
        candidates,
    )
    assert ranking == candidates
    assert len(ranking) == len(set(ranking)) == 10


@pytest.mark.parametrize(
    "response",
    [
        {"ranking": list("ABCDEFGHI")},
        {"ranking": list("ABCDEFGHIK")},
        {"ranking": "ABCDEFGHIJ"},
    ],
)
def test_p7v2_rejects_outputs_outside_locked_repair_scope(response):
    with pytest.raises(ValueError, match="ten labels from A-J"):
        complete_permutation(response, list("0123456789"))


def test_p7v2_requires_ten_distinct_candidates():
    with pytest.raises(ValueError, match="ten distinct candidates"):
        complete_permutation({"ranking": list("ABCDEFGHIJ")}, list("0000000000"))


def test_p7v2_smoke_includes_regression_event():
    prepared = {
        "test_events": [
            {"user_id": f"user-{index:02d}", "timestamp": index}
            for index in range(100)
        ]
    }
    base_keys = {f"{event['user_id']}:{event['timestamp']}" for event in smoke_events(prepared, 20)}
    regression_event = next(
        event
        for event in prepared["test_events"]
        if f"{event['user_id']}:{event['timestamp']}" not in base_keys
    )
    regression_key = f"{regression_event['user_id']}:{regression_event['timestamp']}"
    smoke = configured_smoke_events(
        prepared,
        {"smoke_events": 20, "regression_event_key": regression_key},
    )
    assert len(smoke) == 21
    assert any(f"{event['user_id']}:{event['timestamp']}" == regression_key for event in smoke)


def test_p7v2_permutation_repairs_counts_only_successful_reranks(tmp_path, monkeypatch):
    calls = tmp_path / "calls.jsonl"
    rows = [
        {"key": "rerank:local:fixed", "kind": "rerank", "status": "success", "response": {"ranking": list("AABCDEFGHI")}},
        {"key": "rerank:local:valid", "kind": "rerank", "status": "success", "response": {"ranking": list("ABCDEFGHIJ")}},
        {"key": "stage_r:x", "kind": "stage_r", "status": "success", "response": {"ranking": list("AABCDEFGHI")}},
        {"key": "rerank:local:error", "kind": "rerank", "status": "error", "response": {"ranking": list("AABCDEFGHI")}},
    ]
    calls.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr("src.temporal_books.p7_selfhost_local.project_path", lambda value: calls)
    assert permutation_repairs("ignored.jsonl") == ["rerank:local:fixed"]


def test_replication_graph_smoke_uses_only_fresh_targets():
    prepared = {
        "test_events": [
            {"user_id": f"u{index}", "timestamp": index}
            for index in range(30)
        ]
    }
    calibration, test = smoke_subsets(
        {"p7": {"replication_mode": True, "smoke_test_events": 20}},
        prepared,
    )
    assert calibration == []
    assert len(test) == 20
    assert len({f"{event['user_id']}:{event['timestamp']}" for event in test}) == 20
