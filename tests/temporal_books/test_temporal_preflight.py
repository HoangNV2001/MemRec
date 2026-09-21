from src.temporal_books.p0_audit import timestamp_at_quantile
from src.temporal_books.p1_preflight import build_endpoint_ledger, coverage, history_before, validate_llm_budget
from src.temporal_books.p3_preflight import build_three_hop_ledger, serializable_three_hop_ledger
from src.temporal_books.p4_filtered_preflight import build_filtered_three_hop_ledger, serializable_filtered_ledger


def test_quantile_cutoff_and_strict_past_history():
    assert timestamp_at_quantile({10: 2, 20: 3, 30: 5}, 0.5) == 20
    events = [(10, "a", "", ""), (20, "b", "", ""), (20, "c", "", "")]
    assert history_before(events, 20) == [(10, "a", "", "")]


def test_causal_ledger_excludes_same_timestamp_events_and_keeps_origin_support():
    histories = {
        "source": [(10, "old", "", ""), (30, "x", "", "")],
        "peer_a": [(11, "x", "", ""), (12, "y", "", "")],
        "peer_same_day": [(20, "same_day_item", "", ""), (30, "x", "", "")],
    }
    ledger, _ = build_endpoint_ledger(
        histories,
        ["source"],
        max_anchors=1,
        max_peers=8,
        max_remote_items=8,
        max_witnesses=4,
    )

    assert ledger["y"]["origin_users"] == {"source"}
    # The peer's edge x at the packet timestamp must not be visible.
    assert "same_day_item" not in ledger
    audit = coverage(ledger, [{"gold_item_id": "y"}, {"gold_item_id": "missing"}], [1, 2])
    assert audit["1"]["gold_coverage"] == 0.5
    assert audit["2"]["gold_coverage"] == 0.0


def test_budget_is_capped_at_one_thousand_requests():
    budget = validate_llm_budget(
        {
            "max_total_requests": 1000,
            "semantic_packet_requests": 560,
            "evaluation_events": 100,
            "shared_stage_r_requests_per_event": 1,
            "rerank_arms": 3,
            "retry_reserve_requests": 40,
        }
    )
    assert budget["planned_total_requests"] == 1000


def test_three_hop_ledger_is_strict_past_and_ranks_origins_by_path_support():
    histories = {
        "source": [(5, "old", "", ""), (100, "anchor", "", "")],
        "peer1": [(10, "anchor", "", ""), (20, "bridge", "", "")],
        "peer2": [(30, "bridge", "", ""), (40, "gold", "", "")],
        "same_day_peer": [(50, "bridge", "", ""), (100, "leak", "", "")],
    }
    packets = [{"source_user_id": "source", "packet_timestamp": 100, "anchor_item_ids": ["anchor"]}]
    ledger = build_three_hop_ledger(
        histories,
        packets,
        peers1_per_anchor=4,
        bridge_items_per_peer1=4,
        peers2_per_bridge=4,
        endpoint_items_per_peer2=4,
        max_witnesses=4,
    )
    assert ledger["gold"]["origin_users"] == {"source"}
    assert "leak" not in ledger
    rows = serializable_three_hop_ledger(ledger)
    gold = next(row for row in rows if row["endpoint_item_id"] == "gold")
    assert gold["origin_users_by_path_support"] == ["source"]
    assert gold["origin_path_counts"]["source"] >= 1


def test_filtered_three_hop_keeps_positive_paths_and_drops_negative_edges():
    histories = {
        "source": [(100, "anchor", 5.0, "", "")],
        "peer1": [(10, "anchor", 5.0, "", ""), (20, "bridge", 4.0, "", "")],
        "peer2": [(30, "bridge", 5.0, "", ""), (40, "good", 5.0, "", ""), (50, "bad", 2.0, "", "")],
        "same_day": [(60, "bridge", 5.0, "", ""), (100, "leak", 5.0, "", "")],
    }
    packets = [{"source_user_id": "source", "packet_timestamp": 100, "anchor_item_ids": ["anchor"]}]
    ledger = build_filtered_three_hop_ledger(
        histories,
        packets,
        minimum_rating=4.0,
        half_life_days=730.0,
        diverse_paths_per_origin=3,
        peers1_per_anchor=4,
        bridge_items_per_peer1=4,
        peers2_per_bridge=4,
        endpoint_items_per_peer2=4,
        max_witnesses=4,
    )
    assert "good" in ledger
    assert "bad" not in ledger
    assert "leak" not in ledger
    row = serializable_filtered_ledger(ledger)[0]
    assert row["origin_users_by_quality"] == ["source"]
    assert row["origin_quality_scores"]["source"] > 0
