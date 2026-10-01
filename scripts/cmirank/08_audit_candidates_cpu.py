#!/usr/bin/env python3
"""20-user real-vector candidate integrity audit; no LLM or ranking outcome.

This uses the full common graph snapshot even in smoke. Sampling metadata and
training labels live in reward_audit, separate from the policy_input allowlist.
Statistical shortcut robustness requires a later full-cohort audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cmirank.candidates import (
    MixedCandidateSampler, popularity_from_snapshot, validate_candidate_contract,
)
from src.cmirank.policy_inputs import load_locked_policy_inputs
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", type=int, default=20)
    args = parser.parse_args()
    if not 20 <= args.users <= 30:
        parser.error("Candidate smoke is limited to 20–30 users")
    started = time.monotonic()
    contract_path = ROOT / "configs/cmirank/candidate_sampler_v1.json"
    config = json.loads(contract_path.read_text())
    validate_candidate_contract(config)
    index_manifest = json.loads((args.index_dir / "manifest.json").read_text())
    if (index_manifest["status"] != "COMPLETE_METADATA_INDEX_NOT_MEMORY_OR_PPO_PROMOTION"
            or index_manifest["candidate_contract_sha256"] != file_sha256(contract_path)
            or index_manifest["metadata_sha256"] != config["metadata_sha256"]
            or index_manifest["model_id"] != config["encoder"]["model_id"]
            or index_manifest["model_revision"] != config["encoder"]["revision"]):
        raise ValueError("Full metadata index is missing or has a different contract")
    for name, key in (("item_ids.npy", "item_ids_sha256"), ("vectors.npy", "vectors_sha256"),
                      ("smoke.json", "smoke_sha256")):
        if file_sha256(args.index_dir / name) != index_manifest[key]:
            raise ValueError(f"Index artifact hash differs: {name}")
    smoke = json.loads((args.index_dir / "smoke.json").read_text())
    if (smoke["status"] != "ENCODER_TECHNICAL_SMOKE_PASS_NOT_RANKING_RESULT"
            or smoke["items"] != 20 or smoke["candidate_contract_sha256"] != file_sha256(contract_path)):
        raise ValueError("Index has no matching encoder smoke gate")
    inputs = load_locked_policy_inputs(ROOT)
    ids = np.load(args.index_dir / "item_ids.npy", allow_pickle=False)
    vectors = np.load(args.index_dir / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if (list(vectors.shape) != index_manifest["index_shape"]
            or vectors.shape[1] != config["encoder"]["dimension"]):
        raise ValueError("Index shape differs")
    sampler = MixedCandidateSampler(ids, vectors,
                                    popularity_from_snapshot(inputs["snapshot"].train_data),
                                    seed=config["seed"])
    query_ids = sorted(inputs["groups"]["policy_train"] + inputs["groups"]["policy_val"])[:args.users]
    rows = []
    position_counts: dict[str, Counter] = {"warmup": Counter(), "pseudo": Counter()}
    for uid in query_ids:
        graph_prefix = inputs["snapshot"].train_data[uid]
        warmup, target = inputs["warmups"][uid], inputs["targets"][uid]
        pseudo_prefix = (*graph_prefix, warmup)
        for kind, positive, prefix, extra in (
                ("warmup", warmup, graph_prefix, (target,)),
                ("pseudo", target, pseudo_prefix, ())):
            episode_id = f"books-policy-v1-user-{uid}-{kind}"
            candidates = sampler.sample(episode_id, positive, prefix,
                                        additionally_forbidden=extra)
            if (len(candidates.candidate_ids) != 10 or len(set(candidates.candidate_ids)) != 10
                    or candidates.candidate_ids.count(positive) != 1
                    or {key: len(value) for key, value in candidates.negative_components.items()}
                    != config["composition"]
                    or (set(candidates.candidate_ids) - {positive}) & (set(prefix) | set(extra))):
                raise AssertionError("Candidate integrity or prefix exclusion failed")
            repeated = sampler.sample(episode_id, positive, prefix,
                                      additionally_forbidden=extra)
            if repeated != candidates:
                raise AssertionError("Nonreproducible candidate construction")
            position_counts[kind][candidates.candidate_ids.index(positive)] += 1
            rows.append({"policy_input": candidates.policy_input(),
                         "reward_audit": candidates.reward_audit(),
                         "provenance": {"user_id": uid, "kind": kind,
                                        "snapshot_sha256": inputs["snapshot_sha256"],
                                        "candidate_contract_sha256": file_sha256(contract_path)}})
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite candidate audit output")
    args.output_dir.mkdir(parents=True)
    artifact = args.output_dir / "candidate-smoke.jsonl"
    with artifact.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "status": "CANDIDATE_INTEGRITY_SMOKE_PASS_NOT_G0_OR_RANKING_RESULT",
        "users": len(query_ids), "candidate_sets": len(rows),
        "composition_per_set": config["composition"], "catalog_items": len(ids),
        "common_snapshot_query_users": sum(map(len, inputs["groups"].values())),
        "graph_snapshot_sha256": inputs["snapshot_sha256"],
        "policy_split_manifest_sha256": inputs["manifest_sha256"],
        "candidate_contract_sha256": file_sha256(contract_path),
        "index_manifest_sha256": file_sha256(args.index_dir / "manifest.json"),
        "candidate_smoke_sha256": file_sha256(artifact),
        "positive_position_counts": {kind: dict(sorted(counts.items()))
                                     for kind, counts in position_counts.items()},
        "original_test_candidates_accessed": False,
        "original_instruction_accessed": False,
        "original_suffix_item_ids_parsed_or_used": False,
        "negative_history_collisions": 0,
        "warmup_own_pseudo_target_negatives": 0,
        "policy_fields": ["episode_id", "candidate_ids"],
        "python_version": sys.version, "numpy_version": np.__version__,
        "full_statistical_shortcut_audit_passed": False,
        "stage_w_memory_proven_safe": False, "training_ready": False,
        "real_llm_requests": 0, "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "report.json").write_text(artifact_json_dumps(report))
    print(artifact_json_dumps(report), end="", flush=True)


if __name__ == "__main__":
    main()
