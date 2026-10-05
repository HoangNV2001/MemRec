"""Hash-verified shared catalog and episode recipes; no LLM dependencies."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .candidates import (
    MixedCandidateSampler, PrefixAnchoredCandidateSampler, UniformWarmupSampler, popularity_from_snapshot,
    validate_candidate_contract,
)
from .provenance import file_sha256


def validate_episode_contract(config: dict, pseudo_contract_sha256: str) -> None:
    if (config.get("schema_version") != 1
            or config.get("approval_status") != "researcher_approved_2026-10-02"
            or config.get("pseudo_contract_sha256") != pseudo_contract_sha256
            or config.get("pseudo", {}).get("composition") != {
                "uniform": 3, "popularity_matched": 3, "semantic_hard": 3}
            or config.get("warmup", {}).get("composition") != {"uniform": 9}
            or config["warmup"].get("negative_pool") != "same_nonempty_title_catalog_as_pseudo_sampler"
            or config["warmup"].get("positive") != "train_minus_two_metadata_row_required_title_may_be_empty"
            or config["warmup"].get("forbidden_negatives") != "graph_prefix_and_own_pseudo_target"
            or config["warmup"].get("missing_title_imputation") is not False
            or not config["warmup"].get("seed")
            or config.get("smoke_users") != 20
            or config.get("cohort_changes_allowed") is not False):
        raise ValueError("Episode candidate contract is not the approved uniform/mixed recipe")


def smoke_user_ids(eligible: list[int], config: dict) -> list[int]:
    required = config["smoke_include_user_ids"]
    if len(set(required)) != len(required) or not set(required) <= set(eligible):
        raise ValueError("Smoke coverage users are not unique approved policy users")
    selected = set(required)
    # Explicit loop: selection must not depend on generator/set evaluation order.
    for uid in sorted(eligible):
        if len(selected) == config["smoke_users"]:
            break
        selected.add(uid)
    if len(selected) != config["smoke_users"]:
        raise ValueError("Not enough policy users for the smoke")
    return sorted(selected)


def validate_prefix_contract(config: dict, *, predecessor_candidate_sha: str,
                             predecessor_episode_sha: str, index_sha: str) -> None:
    fixed = {"schema_version": 2, "approval_status": "researcher_approved_2026-10-05",
             "predecessor_candidate_contract_sha256": predecessor_candidate_sha,
             "predecessor_episode_contract_sha256": predecessor_episode_sha,
             "index_manifest_sha256": index_sha, "change_only": "pseudo_semantic_anchor",
             "semantic_anchor": {
                 "source": "allowed_graph_prefix_plus_pre_target_warmup",
                 "pooling": "interaction_weighted_mean_of_available_prefix_vectors_float64",
                 "ordering": "ascending_item_identity_preserving_interaction_multiplicity",
                 "normalization": "l2_then_cast_float32", "minimum_norm": 1e-12,
                 "missing_vectors": "count_without_imputation_fail_if_none",
                 "degenerate_anchor": "fail_without_fallback_or_user_removal",
                 "pseudo_target_used": False},
             "preserve": ["uniform_and_popularity_selection", "3_plus_3_plus_3", "seed",
                          "episode_ids", "hash_shuffle", "encoder_index_text_contract",
                          "uniform_warmup", "cohort_snapshot", "original_evaluation_candidates", "audit_criterion"],
             "cohort_changes_allowed": False, "training_ready": False}
    if set(config) != set(fixed) | {"notes"} or any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("Not the approved prefix-only semantic repair")


def load_candidate_samplers(root: Path, index_dir: Path, snapshot, *, version: int = 1) -> dict:
    if version not in (1, 2):
        raise ValueError("Unsupported recipe version")
    pseudo_path = root / "configs/cmirank/candidate_sampler_v1.json"
    episode_path = root / "configs/cmirank/episode_candidates_v1.json"
    config = json.loads(pseudo_path.read_text())
    episode = json.loads(episode_path.read_text())
    pseudo_hash = file_sha256(pseudo_path)
    validate_candidate_contract(config)
    validate_episode_contract(episode, pseudo_hash)
    index_manifest_path = index_dir / "manifest.json"
    if file_sha256(index_manifest_path) != episode["index_manifest_sha256"]:
        raise ValueError("Index manifest differs from the frozen successful index")
    manifest = json.loads(index_manifest_path.read_text())
    if (manifest["status"] != "COMPLETE_METADATA_INDEX_NOT_MEMORY_OR_PPO_PROMOTION"
            or manifest["candidate_contract_sha256"] != pseudo_hash
            or manifest["metadata_sha256"] != config["metadata_sha256"]
            or manifest["model_id"] != config["encoder"]["model_id"]
            or manifest["model_revision"] != config["encoder"]["revision"]):
        raise ValueError("Full metadata index has a different contract")
    for name, key in (("item_ids.npy", "item_ids_sha256"), ("vectors.npy", "vectors_sha256"),
                      ("smoke.json", "smoke_sha256")):
        if file_sha256(index_dir / name) != manifest[key]:
            raise ValueError(f"Index artifact hash differs: {name}")
    smoke = json.loads((index_dir / "smoke.json").read_text())
    if (smoke["status"] != "ENCODER_TECHNICAL_SMOKE_PASS_NOT_RANKING_RESULT"
            or smoke["items"] != 20 or smoke["candidate_contract_sha256"] != pseudo_hash):
        raise ValueError("Index has no matching encoder smoke gate")
    ids = np.load(index_dir / "item_ids.npy", allow_pickle=False)
    vectors = np.load(index_dir / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if (list(vectors.shape) != manifest["index_shape"]
            or vectors.shape[1] != config["encoder"]["dimension"]):
        raise ValueError("Index shape differs")
    metadata = root / "data/processed/instructrec-books/instructrec-books.meta"
    if file_sha256(metadata) != config["metadata_sha256"]:
        raise ValueError("Static metadata differs from the successful index")
    csv.field_size_limit(16 * 1024 * 1024)
    with metadata.open(encoding="utf-8", newline="") as handle:
        catalog_ids = {int(row["item_id"]) for row in csv.DictReader(handle, delimiter="\t")}
    result = {
        "pseudo": MixedCandidateSampler(ids, vectors, popularity_from_snapshot(snapshot.train_data),
                                         seed=config["seed"]),
        "warmup": UniformWarmupSampler(ids, catalog_ids, seed=episode["warmup"]["seed"]),
        "catalog_ids": catalog_ids, "config": config, "episode_config": episode,
        "episode_contract_sha256": file_sha256(episode_path),
        "candidate_contract_sha256": pseudo_hash,
        "index_manifest_sha256": file_sha256(index_manifest_path),
    }
    if version == 2:
        delta_path = root / "configs/cmirank/candidate_sampler_v2.json"
        delta = json.loads(delta_path.read_text())
        validate_prefix_contract(delta, predecessor_candidate_sha=pseudo_hash,
                                 predecessor_episode_sha=result["episode_contract_sha256"],
                                 index_sha=result["index_manifest_sha256"])
        # V1 validates the unchanged encoder/text/catalog and index provenance.
        # The delta owns the v2 episode/sampler hash; never rewrite the index.
        result["predecessor_recipes"] = dict(result)
        result["pseudo"] = PrefixAnchoredCandidateSampler(
            ids, vectors, result["pseudo"].popularity, seed=config["seed"])
        result["candidate_contract_sha256"] = file_sha256(delta_path)
        result["episode_contract_sha256"] = file_sha256(delta_path)
        result["v2_contract"] = delta
    result["recipe_version"] = version
    return result
