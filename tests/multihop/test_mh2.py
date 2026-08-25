import json

import pytest

from src.multihop import mh2


def test_metrics_and_score_validation_are_deterministic():
    assert mh2.metrics_for_ranking([2, 1, 3], 1)["ndcg_at_3"] == pytest.approx(1 / 1.5849625)
    assert mh2.rank_from_scores({1: 0.5, 2: 0.5, 3: 0.9}, [2, 1, 3]) == [3, 2, 1]
    assert mh2.validate_score_payload(
        {"scores": [{"item_id": 2, "score": 0.5}, {"item_id": 1, "score": 0.3}]}, [1, 2]
    ) == {2: 0.5, 1: 0.3}
    with pytest.raises(ValueError, match="candidate mismatch"):
        mh2.validate_score_payload({"scores": [{"item_id": 1, "score": 0.5}]}, [1, 2])


def test_oracle_selection_uses_ndcg_then_stable_stage_identifier():
    cache = {
        mh2.selection_key(7, "oracle_q4_v0"): {"metrics": {"ndcg_at_5": 0.5}},
        mh2.selection_key(7, "oracle_q2_v3"): {"metrics": {"ndcg_at_5": 0.5}},
        mh2.selection_key(7, "oracle_q6_v1"): {"metrics": {"ndcg_at_5": 0.2}},
    }
    assert mh2.choose_oracle(7, ["oracle_q4_v0", "oracle_q2_v3", "oracle_q6_v1"], cache) == "oracle_q2_v3"


def test_report_metrics_uses_only_independent_report_records():
    stage = {"prompt": "p", "prompt_sha256": "h", "selected_node_ids": [], "K_u": 0, "T_u": 0}
    oracle = [
        {"quota_requested": 2, "variant": index, "stage_r": stage, "remote_actual": 2, "remote_shortfall": 0}
        for index in range(12)
    ]
    user = {
        "user_id": 1,
        "control": {"user_id": 1, "stage_r_context": stage},
        "naive": {"quota_requested": 4, "stage_r": stage, "remote_actual": 4, "remote_shortfall": 0},
        "oracle": oracle,
    }
    cache = {
        mh2.selection_key(1, f"oracle_q2_v{index}"): {"metrics": {"ndcg_at_5": 1.0 if index == 3 else 0.0}}
        for index in range(12)
    }
    values = {"one_hop": 0.0, "naive_two_hop": 0.5, "oracle_two_hop": 1.0}
    for arm, value in values.items():
        for repeat in (1, 2):
            cache[mh2.report_key(1, arm, repeat)] = {
                "metrics": {metric: value for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "ndcg_at_3", "ndcg_at_5")}
            }

    report = mh2.report_metrics([user], cache, report_repeats=2, bootstrap_samples=20, seed=42)

    assert report["analysed_users"] == 1
    assert report["per_user"][0]["chosen_oracle_stage"] == "oracle_q2_v3"
    assert report["comparisons"]["naive_two_hop"]["delta_ndcg_at_5"] == 0.5
    assert report["comparisons"]["oracle_two_hop"]["delta_ndcg_at_5"] == 1.0


def test_load_yaml_substitutes_env_without_loading_training_dependencies(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("provider:\n  api_key: ${ENV:TEST_MH2_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("TEST_MH2_KEY", "value")
    assert mh2.load_yaml(config)["provider"]["api_key"] == "value"


def test_repair_cache_keeps_first_record_for_each_key(tmp_path):
    cache = tmp_path / "raw.jsonl"
    cache.write_text(
        "\n".join(
            [
                json.dumps({"key": "a", "value": 1}),
                json.dumps({"key": "b", "value": 2}),
                json.dumps({"key": "a", "value": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    repair = mh2.repair_cache(cache)

    assert repair == {"records_before": 3, "records_after": 2, "duplicates_removed": 1}
    assert [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()] == [
        {"key": "a", "value": 1},
        {"key": "b", "value": 2},
    ]
