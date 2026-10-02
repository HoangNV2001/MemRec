#!/usr/bin/env python3
"""Approved uniform warm-up / mixed PPO candidates: smoke first, then full.

No LLM calls, original evaluation candidates or original outcome identities.
The complete common snapshot is used even in the 20-user smoke. Model inputs
and reward/audit labels are persisted separately; no ranking-quality claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cmirank.candidate_artifacts import load_candidate_samplers, smoke_user_ids
from src.cmirank.policy_inputs import load_locked_policy_inputs
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def verify_full_gate(smoke_dir: Path, *, source_commit: str, recipes: dict, inputs: dict) -> dict:
    report = json.loads((smoke_dir / "report.json").read_text())
    expected = {
        "status": "CANDIDATE_INTEGRITY_SMOKE_PASS_NOT_G0_OR_RANKING_RESULT",
        "source_commit": source_commit, "users": 20, "candidate_sets": 40,
        "graph_snapshot_sha256": inputs["snapshot_sha256"],
        "policy_split_manifest_sha256": inputs["manifest_sha256"],
        "episode_contract_sha256": recipes["episode_contract_sha256"],
        "candidate_contract_sha256": recipes["candidate_contract_sha256"],
        "index_manifest_sha256": recipes["index_manifest_sha256"],
        "query_user_ids": smoke_user_ids(sorted(inputs["warmups"]), recipes["episode_config"]),
    }
    if (any(report.get(key) != value for key, value in expected.items())
            or file_sha256(smoke_dir / "candidate-rows.jsonl") != report["candidate_manifest_sha256"]):
        raise ValueError("Full candidate generation has no matching successful smoke")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", choices=("20", "all"), default="20")
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite candidate artifacts")
    if args.users == "all" and not args.smoke_dir:
        parser.error("Full candidate generation requires --smoke-dir")
    started = time.monotonic()
    inputs = load_locked_policy_inputs(ROOT)
    recipes = load_candidate_samplers(ROOT, args.index_dir, inputs["snapshot"])
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    all_ids = sorted(inputs["warmups"])
    query_ids = smoke_user_ids(all_ids, recipes["episode_config"]) if args.users == "20" else all_ids
    cached = {}
    if args.users == "all":
        verify_full_gate(args.smoke_dir, source_commit=source_commit, recipes=recipes, inputs=inputs)
        for line in (args.smoke_dir / "candidate-rows.jsonl").read_text().splitlines():
            row = json.loads(line)
            cached[row["policy_input"]["episode_id"]] = row
    args.output_dir.mkdir(parents=True)
    artifact = args.output_dir / "candidate-rows.jsonl"
    position_counts = {"warmup": Counter(), "pseudo": Counter()}
    stage_counts = Counter()
    blank_warmups = []
    reused = 0
    with artifact.open("w", encoding="utf-8") as handle:
        for user_index, uid in enumerate(query_ids):
            graph_prefix = inputs["snapshot"].train_data[uid]
            warmup, target = inputs["warmups"][uid], inputs["targets"][uid]
            if warmup not in recipes["pseudo"].positions:
                blank_warmups.append(uid)
            for kind, positive, prefix, extra in (
                    ("warmup", warmup, graph_prefix, (target,)),
                    ("pseudo", target, (*graph_prefix, warmup), ())):
                episode_id = f"books-policy-v1-user-{uid}-{kind}"
                if episode_id in cached:
                    row = cached[episode_id]
                    reused += 1
                else:
                    sampler = recipes[kind]
                    candidates = sampler.sample(episode_id, positive, prefix,
                                                additionally_forbidden=extra)
                    if args.users == "20" and candidates != sampler.sample(
                            episode_id, positive, prefix, additionally_forbidden=extra):
                        raise AssertionError("Nonreproducible candidate construction")
                    row = {"policy_input": candidates.policy_input(),
                           "reward_audit": candidates.reward_audit(),
                           "provenance": {"user_id": uid, "kind": kind,
                                          "snapshot_sha256": inputs["snapshot_sha256"],
                                          "episode_contract_sha256": recipes["episode_contract_sha256"]}}
                ids = row["policy_input"]["candidate_ids"]
                components = row["reward_audit"]["negative_components"]
                expected_composition = recipes["episode_config"][kind]["composition"]
                negatives = [item for group in components.values() for item in group]
                if (len(ids) != 10 or len(set(ids)) != 10 or ids.count(positive) != 1
                        or {key: len(value) for key, value in components.items()} != expected_composition
                        or len(set(negatives)) != 9 or set(negatives) != set(ids) - {positive}
                        or set(negatives) & (set(prefix) | set(extra))
                        or row["reward_audit"]["positive_item_id"] != positive
                        or row["reward_audit"]["positive_position"] != ids.index(positive)
                        or set(row["policy_input"]) != {"episode_id", "candidate_ids"}):
                    raise AssertionError("Candidate integrity, phase recipe or prefix exclusion failed")
                position_counts[kind][ids.index(positive)] += 1
                stage_counts[kind] += 1
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            if user_index % 100 == 0:
                handle.flush()
                print(json.dumps({"phase": "candidates", "users_done": user_index + 1,
                                  "total": len(query_ids), "seconds": time.monotonic() - started}), flush=True)
    report = {
        "status": ("CANDIDATE_INTEGRITY_SMOKE_PASS_NOT_G0_OR_RANKING_RESULT" if args.users == "20"
                   else "COMPLETE_APPROVED_POLICY_CANDIDATES_NOT_G0_OR_PPO_PROMOTION"),
        "source_commit": source_commit, "users": len(query_ids), "query_user_ids": query_ids,
        "candidate_sets": sum(stage_counts.values()), "stage_counts": dict(stage_counts),
        "composition_per_phase": {kind: recipes["episode_config"][kind]["composition"]
                                  for kind in ("warmup", "pseudo")},
        "catalog_items": len(recipes["pseudo"].item_ids),
        "common_snapshot_query_users": len(all_ids),
        "warmup_users_outside_nonempty_title_pool": blank_warmups,
        "smoke_candidate_sets_reused": reused,
        "graph_snapshot_sha256": inputs["snapshot_sha256"],
        "policy_split_manifest_sha256": inputs["manifest_sha256"],
        "episode_contract_sha256": recipes["episode_contract_sha256"],
        "candidate_contract_sha256": recipes["candidate_contract_sha256"],
        "index_manifest_sha256": recipes["index_manifest_sha256"],
        "candidate_manifest_sha256": file_sha256(artifact),
        "positive_position_counts": {kind: dict(sorted(counts.items()))
                                     for kind, counts in position_counts.items()},
        "original_test_candidates_accessed": False, "original_instruction_accessed": False,
        "original_suffix_item_ids_parsed_or_used": False,
        "negative_history_collisions": 0, "warmup_own_pseudo_target_negatives": 0,
        "policy_fields": ["episode_id", "candidate_ids"],
        "python_version": sys.version, "numpy_version": np.__version__,
        "full_statistical_shortcut_audit_passed": False, "stage_w_memory_proven_safe": False,
        "training_ready": False, "real_llm_requests": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({key: value for key, value in report.items() if key != "query_user_ids"},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
