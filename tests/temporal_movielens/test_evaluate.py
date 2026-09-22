import pytest

from src.temporal_movielens.evaluate import (
    analyze,
    burst_bucket,
    event_rankings,
    preceding_gap_bucket,
)


def toy_config():
    return {
        "study": {"run_id": "toy-hnv"},
        "graph": {
            "fusion_alpha": 0.8,
            "bootstrap_resamples": 100,
            "bootstrap_seed": 7,
            "admission_min_delta": 0.03,
        },
    }


def toy_row():
    candidates = [str(value) for value in range(10)]
    zeros = {item_id: 0.0 for item_id in candidates}
    exact_ppr = dict(zeros)
    exact_ppr["9"] = 1.0
    session_ppr = dict(zeros)
    session_ppr["9"] = 2.0
    return {
        "event_key": "u:1000",
        "candidate_item_ids": candidates,
        "gold_label_used_for_scoring": False,
        "views": {
            "exact": {"one_step_scores": zeros, "ppr_scores": exact_ppr},
            "session_300s": {"one_step_scores": zeros, "ppr_scores": session_ppr},
        },
    }


def test_preregistered_bucket_boundaries():
    assert burst_bucket(0.79) == "lt_0.80"
    assert burst_bucket(0.80) == "0.80_to_0.95"
    assert burst_bucket(0.95) == "0.80_to_0.95"
    assert burst_bucket(0.951) == "gt_0.95"
    assert preceding_gap_bucket(301) == "5m_to_1h"
    assert preceding_gap_bucket(3600) == "5m_to_1h"
    assert preceding_gap_bucket(3601) == "1h_to_1d"
    assert preceding_gap_bucket(86401) == "gt_1d"
    with pytest.raises(ValueError):
        preceding_gap_bucket(300)


def test_event_rankings_materializes_all_frozen_arms():
    local = [str(value) for value in range(10)]
    rankings = event_rankings(toy_row(), local, alpha=0.8)
    assert len(rankings) == 9
    assert rankings["exact_ppr_graph_only"][0] == "9"
    assert rankings["exact_ppr_residual"][0] == "9"
    assert rankings["exact_one_step_residual"] == local


def test_analyze_applies_primary_and_secondary_gates_without_best_arm_selection():
    event = {
        "user_id": "u",
        "timestamp": 1000,
        "gold_item_id": "9",
        "candidate_item_ids": [str(value) for value in range(10)],
        "historical_burstiness_le_60s": 0.9,
        "previous_gap_seconds": 4000,
    }
    local = [str(value) for value in range(10)]
    calls = {"rerank:local:u:1000": {"value": local}}
    result = analyze([event], [toy_row()], calls, toy_config())
    assert result["arms"]["local"]["ndcg_at_5"] == 0.0
    assert result["arms"]["exact_ppr_residual"]["ndcg_at_5"] == 1.0
    assert result["gates"]["primary_exact_ppr"] == "pass"
    assert result["gates"]["secondary_session_300s_ppr"] == "pass"
    assert result["gates"]["conclusion"] == "robust_cross_domain_transfer"
    assert result["evidence_coverage"]["exact"]["gold_ppr_rate"] == 1.0
    assert result["descriptive_buckets"]["target_preceding_gap"]["1h_to_1d"]["n_events"] == 1
