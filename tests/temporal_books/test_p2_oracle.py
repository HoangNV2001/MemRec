from src.temporal_books.p2_oracle import (
    event_ndcg_at_5,
    overlay_memories,
    rerank_prompt,
    smoke_subset,
    stable_sample_candidates,
    validate_ranking,
)
from src.temporal_books.p2_analysis import paired_bootstrap_ci


def test_candidate_sampling_is_fixed_and_excludes_history():
    pool = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "gold"]
    first = stable_sample_candidates(pool, {"a", "b"}, "gold", n_candidates=10, seed_key="same")
    second = stable_sample_candidates(pool, {"a", "b"}, "gold", n_candidates=10, seed_key="same")
    assert first == second
    assert len(first) == 10
    assert "gold" in first
    assert not ({"a", "b"} & set(first))


def test_oracle_overlay_only_changes_gold_endpoint():
    event = {"candidate_item_ids": ["gold", "other"], "gold_item_id": "gold"}
    item_info = {"gold": {"base_memory": "gold base"}, "other": {"base_memory": "other base"}}
    values = overlay_memories(
        event,
        item_info,
        {"source": {"memory": "semantic packet"}},
        {},
        {"gold": ["source"], "other": ["source"]},
        arm="oracle_two_hop",
        packet_cap=2,
    )
    assert "semantic packet" in values["gold"]
    assert values["other"] == "other base"


def test_ranking_validation_and_ndcg():
    assert validate_ranking({"ranking": list("ABCDEFGHIJ")}, [str(i) for i in range(10)]) == [str(i) for i in range(10)]
    assert event_ndcg_at_5(["a", "b", "c"], "b") > 0


def test_rerank_schema_excludes_rationale_to_bound_output():
    event = {"candidate_item_ids": [str(i) for i in range(10)]}
    _, schema = rerank_prompt(event, [], {str(i): "memory" for i in range(10)})
    assert set(schema) == {"ranking"}


def test_smoke_keeps_every_source_needed_by_smoke_rerank_prompt():
    prepared = {
        "events": [
            {"user_id": "u1", "timestamp": 1, "gold_item_id": "gold", "candidate_item_ids": ["anchor", "other"]},
        ],
        "source_packets": [{"source_user_id": "base"}, {"source_user_id": "direct"}, {"source_user_id": "remote"}],
        "anchors_by_item": {"anchor": ["direct"]},
        "origins_by_endpoint_support_gte_threshold": {"gold": ["remote"]},
    }
    packets, events = smoke_subset(prepared, {"smoke_packet_sources": 1, "smoke_evaluation_events": 1})
    assert len(events) == 1
    packet_ids = {packet["source_user_id"] for packet in packets}
    assert {"direct", "remote"} <= packet_ids


def test_paired_bootstrap_is_deterministic_and_contains_mean():
    values = [-0.2, 0.0, 0.1, 0.4]
    first = paired_bootstrap_ci(values, resamples=1000, seed=7)
    assert first == paired_bootstrap_ci(values, resamples=1000, seed=7)
    assert first[0] <= sum(values) / len(values) <= first[1]
