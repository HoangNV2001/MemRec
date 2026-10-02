"""Synthetic CPU audits: ties, split isolation, provenance and feature separation."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.cmirank.candidates import shuffle_candidates
from src.cmirank.provenance import file_sha256
from src.cmirank.shortcut_audit import (
    PROBES, feature_rows, fit_probe_scores, null_pvalue, rank_credits,
    validate_audit_contract, validate_feature_row, verify_candidate_run,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shortcut_script", ROOT / "scripts/cmirank/09_audit_shortcuts_cpu.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_all_ties_have_exact_random_chance_not_first_item_bias():
    credit = rank_credits(np.zeros((3, 10)))
    assert np.allclose(credit[:, :, 0], 0.1)
    assert np.allclose(credit[:, :, 1], 0.5)
    assert np.allclose(credit[:, :, 2], sum(1 / np.log2(r + 1) for r in range(1, 6)) / 10)


def test_unique_scores_and_partial_ties_match_rank_averaging():
    credit = rank_credits(np.asarray([[9, 9, 8, 7, 6, 5, 4, 3, 2, 1]]))[0]
    assert np.allclose(credit[:2, 0], 0.5)
    assert np.allclose(credit[:2, 2], (1 + 1 / np.log2(3)) / 2)
    assert credit[5, 1] == credit[5, 2] == 0
    assert np.allclose(rank_credits(np.arange(10)[None, :])[0, -1], 1)


@pytest.mark.parametrize("scores", [np.zeros((2, 9)), np.arange(10), np.full((2, 10), np.nan)])
def test_bad_score_shapes_or_nonfinite_fail(scores):
    with pytest.raises(ValueError):
        rank_credits(scores)


def test_probe_fit_is_invariant_to_all_validation_labels():
    rng = np.random.default_rng(7)
    features = {name: rng.normal(size=(8, 10)) for name in PROBES}
    targets = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    mask = np.asarray([True] * 5 + [False] * 3)
    scores, fitted = fit_probe_scores(features, targets, mask)
    changed = targets.copy()
    changed[~mask] = 0
    other_scores, other_fitted = fit_probe_scores(features, changed, mask)
    assert fitted == other_fitted
    assert all(np.array_equal(scores[name], other_scores[name]) for name in PROBES)
    assert fitted["position"]["train_position_counts"] == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_null_is_repeatable_and_all_ties_cannot_be_significant():
    credits = rank_credits(np.zeros((30, 10)))[:, :, 0]
    assert null_pvalue(credits, 0.1, seed=5, permutations=999) == 1
    peaked = rank_credits(np.tile(np.arange(10), (30, 1)))[:, :, 0]
    assert null_pvalue(peaked, 1, seed=5, permutations=999) == 0.001
    assert null_pvalue(peaked, 0.2, seed=5, permutations=999) == null_pvalue(peaked, 0.2, seed=5, permutations=999)


def test_contract_is_fixed_before_outcome_and_not_training_ready():
    config = json.loads((ROOT / "configs/cmirank/shortcut_audit_v1.json").read_text())
    validate_audit_contract(config)
    config["material_hit1_delta"] = 0.01
    with pytest.raises(ValueError, match="contract"):
        validate_audit_contract(config)


def test_features_ignore_rewards_components_and_positive_position():
    ids = list(range(10))
    recipes = {"pseudo": SimpleNamespace(positions=dict(zip(ids, ids)),
               vectors=np.eye(10, dtype=np.float32), popularity={1: 4})}
    texts = {item: f"Book {item}" for item in ids}
    lengths = {item: (6, 0, False) for item in ids}
    row = {"policy_input": {"episode_id": "a", "candidate_ids": ids},
           "reward_audit": {"positive_item_id": 0, "positive_position": 0}}
    before = feature_rows([row], recipes, texts, lengths)
    row["reward_audit"] = {"positive_item_id": 9, "positive_position": 9,
                           "negative_components": {"FORBIDDEN_MARKER": [0]}}
    after = feature_rows([row], recipes, texts, lengths)
    assert before == after
    assert after[0]["features"]["semantic_centrality"] == [0] * 10
    validate_feature_row(after[0], row["policy_input"])
    after[0]["features"]["positive_position"] = [9] * 10
    with pytest.raises(ValueError):
        validate_feature_row(after[0], row["policy_input"])


def candidate_fixture(tmp_path):
    eligible = list(range(20))
    inputs = {"warmups": {uid: 100 + uid for uid in eligible},
              "targets": {uid: 200 + uid for uid in eligible},
              "snapshot": SimpleNamespace(train_data={uid: (500 + uid,) for uid in eligible}),
              "snapshot_sha256": "graph", "manifest_sha256": "split"}
    negatives = list(range(300, 309))
    catalog = set(negatives) | set(inputs["warmups"].values()) | set(inputs["targets"].values())
    recipes = {"config": {"seed": "pseudo"}, "episode_config": {
                "warmup": {"seed": "warmup", "composition": {"uniform": 9}},
                "pseudo": {"composition": {"uniform": 3, "popularity_matched": 3, "semantic_hard": 3}}},
               "catalog_ids": catalog, "pseudo": SimpleNamespace(positions=dict.fromkeys(catalog, 0)),
               "episode_contract_sha256": "episode", "candidate_contract_sha256": "contract",
               "index_manifest_sha256": "index"}
    positions = {kind: Counter() for kind in ("warmup", "pseudo")}
    rows = []
    for uid in eligible:
        for kind in ("warmup", "pseudo"):
            positive = inputs["warmups" if kind == "warmup" else "targets"][uid]
            episode = f"books-policy-v1-user-{uid}-{kind}"
            ids = list(shuffle_candidates([positive, *negatives], seed=kind, episode_id=episode))
            components = {"uniform": negatives} if kind == "warmup" else {
                "uniform": negatives[:3], "popularity_matched": negatives[3:6], "semantic_hard": negatives[6:]}
            row = {"policy_input": {"episode_id": episode, "candidate_ids": ids},
                   "reward_audit": {"positive_item_id": positive, "positive_position": ids.index(positive),
                                    "negative_components": components},
                   "provenance": {"user_id": uid, "kind": kind, "snapshot_sha256": "graph",
                                  "episode_contract_sha256": "episode"}}
            rows.append(row)
            positions[kind][str(ids.index(positive))] += 1
    common = {"source_commit": "candidate-source", "graph_snapshot_sha256": "graph",
              "policy_split_manifest_sha256": "split", "episode_contract_sha256": "episode",
              "candidate_contract_sha256": "contract", "index_manifest_sha256": "index",
              "original_test_candidates_accessed": False, "original_instruction_accessed": False,
              "original_suffix_item_ids_parsed_or_used": False, "negative_history_collisions": 0,
              "warmup_own_pseudo_target_negatives": 0, "real_llm_requests": 0, "training_ready": False,
              "users": 20, "query_user_ids": eligible, "candidate_sets": 40,
              "stage_counts": {"warmup": 20, "pseudo": 20}, "smoke_candidate_sets_reused": 40,
              "positive_position_counts": {kind: dict(counts) for kind, counts in positions.items()}}
    for kind, status in (("full", "COMPLETE_APPROVED_POLICY_CANDIDATES_NOT_G0_OR_PPO_PROMOTION"),
                         ("smoke", "CANDIDATE_INTEGRITY_SMOKE_PASS_NOT_G0_OR_RANKING_RESULT")):
        directory = tmp_path / f"{kind}-hnv"
        directory.mkdir()
        artifact = directory / "candidate-rows.jsonl"
        artifact.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / "report.json").write_text(json.dumps({**common, "status": status,
                                                         "candidate_manifest_sha256": file_sha256(artifact)}))
    (tmp_path / "source-commit.txt").write_text('candidate-source\n')
    (tmp_path / "cleanup.json").write_text(json.dumps({"process_exit_code": 0, "device": "cpu",
                                                       "gpu_requested": False, "child_exited": True}))
    config = {"full_report_sha256": file_sha256(tmp_path / "full-hnv/report.json"),
              "full_candidates_sha256": file_sha256(tmp_path / "full-hnv/candidate-rows.jsonl"),
              "candidate_source_commit": "candidate-source"}
    return config, inputs, recipes


def test_completed_candidate_rows_smoke_reuse_and_cleanup_verified(tmp_path):
    config, inputs, recipes = candidate_fixture(tmp_path)
    assert len(verify_candidate_run(tmp_path, config, inputs, recipes)) == 40
    (tmp_path / "cleanup.json").write_text('{}')
    with pytest.raises(ValueError, match="cleanup"):
        verify_candidate_run(tmp_path, config, inputs, recipes)


@pytest.mark.parametrize("tamper", ["extra_policy_field", "negative_history", "smoke_bytes", "candidate_hash"])
def test_full_audit_fails_closed_on_tampered_completed_artifacts(tmp_path, tamper):
    config, inputs, recipes = candidate_fixture(tmp_path)
    path = tmp_path / "full-hnv/candidate-rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if tamper == "extra_policy_field":
        rows[0]["policy_input"]["target_item_id"] = 100
    elif tamper == "negative_history":
        inputs["snapshot"].train_data[0] = (300,)
    elif tamper == "smoke_bytes":
        (tmp_path / "smoke-hnv/candidate-rows.jsonl").write_text(path.read_text().replace('"episode_id":', '"episode_id" :', 1))
        report_path = tmp_path / "smoke-hnv/report.json"
        report = json.loads(report_path.read_text())
        report["candidate_manifest_sha256"] = file_sha256(tmp_path / "smoke-hnv/candidate-rows.jsonl")
        report_path.write_text(json.dumps(report))
    if tamper in ("extra_policy_field", "candidate_hash"):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows) + ('\n' if tamper == "candidate_hash" else ''))
    if tamper == "extra_policy_field":
        config["full_candidates_sha256"] = file_sha256(path)  # Exercise schema check even with a new hash.
    with pytest.raises(ValueError):
        verify_candidate_run(tmp_path, config, inputs, recipes)


def test_feature_full_gate_requires_exact_successful_smoke(tmp_path):
    ids = list(range(20))
    rows = [{"episode_id": f"books-policy-v1-user-{uid}-pseudo", "candidate_ids": list(range(10)),
             "features": {name: [0] * 10 for name in PROBES}} for uid in ids]
    (tmp_path / "features.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {"status": "CPU_SHORTCUT_FEATURE_SMOKE_PASS_NOT_STATISTICAL_PROMOTION", "users": 20,
              "query_user_ids": ids, "provenance": {"source_commit": "new-audit-source"},
              "features_sha256": file_sha256(tmp_path / "features.jsonl")}
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert len(script.verify_smoke(tmp_path, report["provenance"], ids)) == 20
    with pytest.raises(ValueError, match="no matching"):
        script.verify_smoke(tmp_path, {"source_commit": "different"}, ids)


def test_validation_risk_flag_cannot_be_promoted_to_training():
    """Strong synthetic shortcut is detected with train-only fit."""
    config = json.loads((ROOT / "configs/cmirank/shortcut_audit_v1.json").read_text())
    inputs = {"groups": {"policy_train": list(range(1497)), "policy_val": list(range(1497, 1797))}}
    rows, features = [], []
    for uid in range(1797):
        rows.append({"reward_audit": {"positive_position": 0}, "provenance": {"user_id": uid}})
        features.append({"features": {name: [1] + [0] * 9 for name in PROBES}})
    result = script.evaluate(features, rows, inputs, config)
    assert set(result["risk_flags"]) == set(PROBES)
    assert result["validation_users"] == 300
    assert result["probes"]["popularity"]["bonferroni_p"] == 0.0007
