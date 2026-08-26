from src.temporal_books.p0_audit import timestamp_at_quantile
from src.temporal_books.p1_preflight import build_endpoint_ledger, coverage, history_before, validate_llm_budget


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
