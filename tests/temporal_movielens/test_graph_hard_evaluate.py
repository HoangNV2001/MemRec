import pytest

from src.temporal_movielens.graph_hard_evaluate import (
    PRIMARY_ARM,
    analyze,
    event_hit,
    event_ndcg,
    event_rankings,
    validate_blind_scores,
)


def toy_config():
    return {
        "study": {"run_id": "m7-toy-hnv"},
        "graph": {
            "fusion_alpha": 0.8,
            "bootstrap_resamples": 100,
            "bootstrap_seed": 7,
            "admission_min_delta": 0.03,
        },
    }


def toy_rows():
    candidates = [str(value) for value in range(10)]
    zeros = {item_id: 0.0 for item_id in candidates}
    one_step = dict(zeros)
    one_step["9"] = 1.0
    graph = {
        "event_key": "u:1000",
        "candidate_item_ids": candidates,
        "gold_label_used_for_scoring": False,
        "views": {
            "exact": {"one_step_scores": one_step, "ppr_scores": zeros},
            "session_300s": {"one_step_scores": one_step, "ppr_scores": zeros},
        },
    }
    baseline = {
        "event_key": "u:1000",
        "candidate_item_ids": candidates,
        "test_labels_used_for_scoring": False,
        "scores": {
            name: {item_id: float(item_id == "9") for item_id in candidates}
            for name in ("most_popular", "bpr_mf", "sasrec")
        },
    }
    return graph, baseline


def test_m7_metrics_cover_memrec_cutoffs():
    ranking = [str(value) for value in range(10)]
    assert event_hit(ranking, "0", 1) == 1.0
    assert event_hit(ranking, "3", 3) == 0.0
    assert event_ndcg(ranking, "2", 3) == pytest.approx(0.5)
    assert event_ndcg(ranking, "5", 5) == 0.0


def test_m7_rankings_make_exact_one_step_the_primary_treatment():
    graph, baseline = toy_rows()
    local = [str(value) for value in range(10)]
    rankings = event_rankings(graph, baseline, local, alpha=0.8)
    assert PRIMARY_ARM == "exact_one_step_residual"
    assert rankings[PRIMARY_ARM][0] == "9"
    assert rankings["exact_ppr_residual"] == local
    assert rankings["sasrec"][0] == "9"


def test_m7_blind_validation_rejects_bad_local_permutation():
    graph, baseline = toy_rows()
    calls = {
        "stage_r:u:1000": {"value": {"facets": []}},
        "rerank:local:u:1000": {"value": [str(value) for value in range(9)]},
    }
    with pytest.raises(RuntimeError, match="local permutation"):
        validate_blind_scores(
            [graph], [baseline], calls, expected_events=1, candidates_per_event=10
        )


def test_m7_analysis_applies_only_preregistered_primary_gate():
    graph, baseline = toy_rows()
    event = {
        "user_id": "u",
        "timestamp": 1000,
        "gold_item_id": "9",
        "candidate_item_ids": [str(value) for value in range(10)],
    }
    calls = {"rerank:local:u:1000": {"value": [str(value) for value in range(10)]}}
    result = analyze([event], [graph], [baseline], calls, toy_config())
    assert result["arms"]["local"]["ndcg_at_5"] == 0.0
    assert result["arms"][PRIMARY_ARM]["ndcg_at_5"] == 1.0
    assert result["arms"][PRIMARY_ARM]["hit_at_1"] == 1.0
    assert result["gate"]["decision"] == "pass"
    assert result["comparisons_vs_local"]["exact_ppr_residual"]["delta_ndcg_at_5"] == 0.0
