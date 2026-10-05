"""V2 is a sampler validity repair, not a new model or benchmark protocol."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.cmirank.candidate_artifacts import validate_prefix_contract
from src.cmirank.candidates import MixedCandidateSampler, PrefixAnchoredCandidateSampler, digest_key
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.shortcut_audit import bind_completed_prefix_run, validate_v2_audit_delta

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prefix_generation", ROOT / "scripts/cmirank/08_audit_candidates_cpu.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


def models():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(40, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    popularity = {item: item % 5 for item in range(40)}
    return (MixedCandidateSampler(range(40), vectors, popularity, seed="fixed"),
            PrefixAnchoredCandidateSampler(range(40), vectors.copy(), popularity, seed="fixed"))


def test_v2_preserves_uniform_popularity_seed_and_policy_allowlist():
    old, new = models()
    v1, v2 = [model.sample("same-event", 10, [1, 2, 3]) for model in (old, new)]
    for name in ("uniform", "popularity_matched"):
        assert v1.negative_components[name] == v2.negative_components[name]
    assert len(set(v2.candidate_ids)) == 10
    assert not set(v2.candidate_ids) & {1, 2, 3}
    assert set(v2.policy_input()) == {"episode_id", "candidate_ids"}
    assert v2 == new.sample("same-event", 10, [3, 1, 2])


def test_anchor_has_no_positive_vector_dependency_and_matches_prefix_mean():
    _, model = models()
    query = model.semantic_query(10, [1, 2, 3])
    expected = model.vectors[[1, 2, 3]].astype(np.float64).mean(axis=0)
    expected = (expected / np.linalg.norm(expected)).astype(np.float32)
    assert np.array_equal(query, expected)
    assert np.array_equal(query, model.semantic_query(20, [3, 2, 1]))
    before = model.sample("same-event", 10, [1, 2, 3])
    model.vectors[10] *= -1  # Change only positive metadata, still unit and finite.
    assert np.array_equal(query, model.semantic_query(10, [1, 2, 3]))
    after = model.sample("same-event", 10, [1, 2, 3])
    assert before == after


def test_semantic_negatives_are_top_three_by_prefix_not_target():
    _, model = models()
    row = model.sample("same-event", 10, [1, 2, 3])
    blocked = {1, 2, 3, 10, *row.negative_components["uniform"],
               *row.negative_components["popularity_matched"]}
    scores = model.vectors @ model.semantic_query(10, [1, 2, 3])
    expected = sorted((item for item in range(40) if item not in blocked), key=lambda item: (
        -float(scores[item]), digest_key("fixed", "same-event", "semantic_tie", item), item))[:3]
    assert row.negative_components["semantic_hard"] == tuple(expected)


def test_missing_prefix_embeddings_are_counted_without_imputation_or_user_drop():
    _, model = models()
    query, counts = model.prefix_anchor([1, 999, 1])
    assert np.allclose(query, model.vectors[1])
    assert counts == {"prefix_interactions": 3, "embedded_prefix_interactions": 2,
                      "unembedded_prefix_interactions": 1}
    with pytest.raises(ValueError, match="No frozen"):
        model.sample("same-event", 10, [999])


def test_interaction_multiplicity_is_preserved_in_anchor():
    _, model = models()
    query, _ = model.prefix_anchor([1, 2, 1])
    expected = model.vectors[[1, 1, 2]].astype(np.float64).mean(axis=0)
    expected /= np.linalg.norm(expected)
    assert np.allclose(query, expected)


def test_zero_mean_anchor_and_history_positive_fail_without_fallback():
    _, model = models()
    model.vectors[1] = -model.vectors[0]
    with pytest.raises(ValueError, match="Degenerate"):
        model.sample("same-event", 10, [0, 1])
    with pytest.raises(ValueError, match="not novel"):
        model.sample("same-event", 10, [10, 2])


@pytest.mark.parametrize("change", ["target_anchor", "seed", "cohort", "predecessor", "threshold"])
def test_repair_contract_cannot_silently_change_other_settings(change):
    config = json.loads((ROOT / "configs/cmirank/candidate_sampler_v2.json").read_text())
    kwargs = {"predecessor_candidate_sha": file_sha256(ROOT / "configs/cmirank/candidate_sampler_v1.json"),
              "predecessor_episode_sha": file_sha256(ROOT / "configs/cmirank/episode_candidates_v1.json"),
              "index_sha": config["index_manifest_sha256"]}
    validate_prefix_contract(config, **kwargs)
    if change == "target_anchor": config["semantic_anchor"]["pseudo_target_used"] = True
    if change == "seed": config["seed"] = "pick-another-seed"
    if change == "cohort": config["cohort_changes_allowed"] = True
    if change == "predecessor": config["predecessor_candidate_contract_sha256"] = "other"
    if change == "threshold": config["semantic_anchor"]["minimum_norm"] = 0.01
    with pytest.raises(ValueError, match="prefix-only"):
        validate_prefix_contract(config, **kwargs)


def test_v2_audit_cannot_change_the_original_threshold_or_probe_family():
    delta = json.loads((ROOT / "configs/cmirank/shortcut_audit_v2.json").read_text())
    base_sha = file_sha256(ROOT / "configs/cmirank/shortcut_audit_v1.json")
    validate_v2_audit_delta(delta, base_sha)
    delta["material_hit1_delta"] = 0.20
    with pytest.raises(ValueError, match="unchanged"):
        validate_v2_audit_delta(delta, base_sha)


def binding_fixture(tmp_path):
    inputs = {"warmups": {1: 7, 2: 8}, "snapshot": SimpleNamespace(train_data={1: (5,), 2: (6,)}),
              "snapshot_sha256": "graph", "manifest_sha256": "split"}
    model = SimpleNamespace(prefix_anchor=lambda prefix: (np.ones(1, dtype=np.float32),
                           {"prefix_interactions": len(prefix), "embedded_prefix_interactions": len(prefix),
                            "unembedded_prefix_interactions": 0}))
    recipes = {"pseudo": model, "recipe_version": 2, "candidate_contract_sha256": "candidate",
               "episode_contract_sha256": "episode", "index_manifest_sha256": "index"}
    coverage = [{"user_id": uid, "prefix_interactions": 2, "embedded_prefix_interactions": 2,
                 "unembedded_prefix_interactions": 0,
                 "anchor_float32_sha256": hashlib.sha256(np.ones(1, dtype=np.float32).tobytes()).hexdigest()}
                for uid in (1, 2)]
    for phase in ("smoke", "full"):
        directory = tmp_path / f"{phase}-hnv"
        directory.mkdir()
        (directory / "anchor-coverage.json").write_text(artifact_json_dumps(coverage))
        (directory / "candidate-rows.jsonl").write_text('{}\n')
        report = {"source_commit": "same-source", "recipe_version": 2,
                  "prefix_anchor_coverage_sha256": file_sha256(directory / "anchor-coverage.json"),
                  "prefix_anchor_users_verified": 2, "candidate_contract_sha256": "candidate",
                  "episode_contract_sha256": "episode", "index_manifest_sha256": "index",
                  "graph_snapshot_sha256": "graph", "policy_split_manifest_sha256": "split",
                  "candidate_manifest_sha256": file_sha256(directory / "candidate-rows.jsonl")}
        (directory / "report.json").write_text(json.dumps(report))
    (tmp_path / "source-commit.txt").write_text('same-source\n')
    return inputs, recipes


def test_new_candidate_binding_requires_same_source_and_full_anchor_coverage(tmp_path):
    inputs, recipes = binding_fixture(tmp_path)
    bound = bind_completed_prefix_run(tmp_path, source_commit="same-source", inputs=inputs, recipes=recipes)
    assert bound["candidate_source_commit"] == "same-source"
    assert bound["full_candidates_sha256"] == file_sha256(tmp_path / "full-hnv/candidate-rows.jsonl")
    with pytest.raises(ValueError, match="exact audit source"):
        bind_completed_prefix_run(tmp_path, source_commit="other", inputs=inputs, recipes=recipes)
    (tmp_path / "smoke-hnv/anchor-coverage.json").write_text('[]\n')
    with pytest.raises(ValueError, match="coverage"):
        bind_completed_prefix_run(tmp_path, source_commit="same-source", inputs=inputs, recipes=recipes)


def test_v2_full_promotion_requires_the_same_anchor_coverage(tmp_path):
    episode = {"smoke_include_user_ids": [18, 19], "smoke_users": 20}
    recipes = {"recipe_version": 2, "episode_config": episode, "episode_contract_sha256": "episode",
               "candidate_contract_sha256": "candidate", "index_manifest_sha256": "index"}
    inputs = {"warmups": dict.fromkeys(range(20), 100), "snapshot_sha256": "graph",
              "manifest_sha256": "split", "prefix_anchor_coverage_sha256": "coverage"}
    (tmp_path / "candidate-rows.jsonl").write_text('{}\n')
    report = {"status": "CANDIDATE_INTEGRITY_SMOKE_PASS_NOT_G0_OR_RANKING_RESULT",
              "source_commit": "source", "users": 20, "candidate_sets": 40, "query_user_ids": list(range(20)),
              "recipe_version": 2, "prefix_anchor_coverage_sha256": "coverage",
              "graph_snapshot_sha256": "graph", "policy_split_manifest_sha256": "split",
              "episode_contract_sha256": "episode", "candidate_contract_sha256": "candidate",
              "index_manifest_sha256": "index", "candidate_manifest_sha256": file_sha256(tmp_path / "candidate-rows.jsonl")}
    (tmp_path / "report.json").write_text(json.dumps(report))
    generation.verify_full_gate(tmp_path, source_commit="source", recipes=recipes, inputs=inputs)
    inputs["prefix_anchor_coverage_sha256"] = "changed"
    with pytest.raises(ValueError, match="no matching"):
        generation.verify_full_gate(tmp_path, source_commit="source", recipes=recipes, inputs=inputs)
