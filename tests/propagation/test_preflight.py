from src.propagation.p0_temporal_audit import audit_events
from src.propagation.p1_source_route_audit import build_endpoint_ledger, coverage_audit, serializable_ledger


def test_temporal_audit_distinguishes_local_positions_from_global_clock():
    local_positions = [(1, 1.0), (2, 1.0), (1, 2.0), (2, 2.0)]
    report = audit_events(local_positions)

    assert report["per_user_monotonic_users"] == 2
    assert report["timestamp_values_reused_across_users"] == 2
    assert report["dynamic_cross_user_propagation_admissible"] is False

    strict = audit_events([(1, 1.0), (2, 2.0), (1, 3.0)])
    assert strict["dynamic_cross_user_propagation_admissible"] is True


def test_source_only_route_ledger_and_posthoc_coverage_are_candidate_blind():
    histories = {
        1: [10],
        2: [10, 20],
        3: [10, 30],
    }
    ledger = build_endpoint_ledger(histories, max_anchors=8, max_peers=8, max_remote_items=8, max_witnesses=4)

    # Every source user's single candidate-blind packet may use the shared
    # anchor.  The audit must retain independent-origin support, rather than
    # treating one hand-picked source as privileged.
    assert ledger[20]["origin_users"] == {1, 3}
    assert ledger[30]["origin_users"] == {1, 2}
    rows = serializable_ledger(ledger)
    assert {row["endpoint_item_id"] for row in rows} >= {20, 30}

    controls = [
        {
            "user_id": 99,
            "ranking_context": {"candidates": [20, 999], "gold_item_id": 20},
        }
    ]
    audit = coverage_audit(ledger, controls, support_thresholds=[1, 2])
    assert audit["1"]["candidate_slot_coverage"] == 0.5
    assert audit["1"]["gold_coverage"] == 1.0
    assert audit["2"]["candidate_slot_coverage"] == 0.5
