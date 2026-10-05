#!/usr/bin/env python3
"""CPU-only, smoke-first shortcut diagnostics on frozen pseudo-candidates.

No sampler changes, model loading, original outcomes or training promotion.
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
from src.cmirank.shortcut_audit import (
    METRICS, PROBES, bind_completed_prefix_run, feature_rows, fit_probe_scores, metadata_features, null_pvalue,
    rank_credits, validate_audit_contract, validate_feature_row, validate_v2_audit_delta, verify_candidate_run,
)


def verify_smoke(smoke_dir: Path, provenance: dict, expected_ids: list[int]) -> dict:
    report = json.loads((smoke_dir / "report.json").read_text())
    if (report.get("status") != "CPU_SHORTCUT_FEATURE_SMOKE_PASS_NOT_STATISTICAL_PROMOTION"
            or report.get("query_user_ids") != expected_ids or report.get("users") != 20
            or report.get("provenance") != provenance
            or file_sha256(smoke_dir / "features.jsonl") != report.get("features_sha256")):
        raise ValueError("Shortcut full audit has no matching successful feature smoke")
    cached = {}
    for line in (smoke_dir / "features.jsonl").read_text().splitlines():
        feature = json.loads(line)
        if feature["episode_id"] in cached:
            raise ValueError("Duplicate smoke feature episode")
        cached[feature["episode_id"]] = feature
    if set(cached) != {f"books-policy-v1-user-{uid}-pseudo" for uid in expected_ids}:
        raise ValueError("Wrong smoke feature scope")
    return cached


def evaluate(features: list[dict], rows: list[dict], inputs: dict, config: dict) -> dict:
    values = {name: np.asarray([feature["features"][name] for feature in features], dtype=np.float64)
              for name in PROBES}
    target_positions = np.asarray([row["reward_audit"]["positive_position"] for row in rows])
    uids = [row["provenance"]["user_id"] for row in rows]
    member_sets = {name: set(members) for name, members in inputs["groups"].items()}
    masks = {name: np.asarray([uid in members for uid in uids], dtype=bool)
             for name, members in member_sets.items()}
    if (masks["policy_train"].sum() != 1497 or masks["policy_val"].sum() != 300
            or np.any(masks["policy_train"] == masks["policy_val"])):
        raise ValueError("Diagnostic fit/evaluation groups differ from locked split")
    scores, fitted = fit_probe_scores(values, target_positions, masks["policy_train"])
    chance_values = rank_credits(np.zeros((1, 10)))[0, 0]
    result = {"chance": dict(zip(METRICS, chance_values.tolist())), "probes": {}, "risk_flags": []}
    for index, name in enumerate(PROBES):
        credits = rank_credits(scores[name])
        metrics = {group: dict(zip(METRICS, credits[mask, target_positions[mask]].mean(axis=0).tolist()))
                   for group, mask in masks.items()}
        val = masks["policy_val"]
        observed = metrics["policy_val"]["hit_at_1"]
        p = null_pvalue(credits[val, :, 0], observed, seed=config["seed"] + index,
                        permutations=config["permutations"])
        adjusted = min(1.0, p * len(PROBES))
        delta = observed - result["chance"]["hit_at_1"]
        flag = delta >= config["material_hit1_delta"] - 1e-12 and adjusted <= config["familywise_alpha"]
        result["probes"][name] = {"fit": fitted[name], "metrics": metrics,
                                  "validation_hit1_delta": delta,
                                  "validation_null_p": p, "bonferroni_p": adjusted,
                                  "risk_flag": bool(flag)}
        if flag:
            result["risk_flags"].append(name)
    result["fit_users"] = int(masks["policy_train"].sum())
    result["validation_users"] = int(masks["policy_val"].sum())
    result["sampling_component_field_exposed_to_policy"] = False
    result["positive_position_counts"] = dict(Counter(str(pos) for pos in target_positions))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", choices=("20", "all"), default="20")
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--recipe-version", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite shortcut artifacts")
    if args.users == "all" and not args.smoke_dir:
        parser.error("Full audit requires --smoke-dir")
    started = time.monotonic()
    contract_path = ROOT / "configs/cmirank/shortcut_audit_v1.json"
    config = json.loads(contract_path.read_text())
    validate_audit_contract(config)
    inputs = load_locked_policy_inputs(ROOT)
    recipes = load_candidate_samplers(ROOT, args.index_dir, inputs["snapshot"], version=args.recipe_version)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    base_audit_sha = file_sha256(contract_path)
    if args.recipe_version == 2:
        contract_path = ROOT / "configs/cmirank/shortcut_audit_v2.json"
        delta = json.loads(contract_path.read_text())
        validate_v2_audit_delta(delta, base_audit_sha)
        config = {**config, "candidate_run_id": delta["candidate_run_id"]}
        if args.candidate_run_dir.name != config["candidate_run_id"]:
            raise ValueError("Wrong prefix candidate run identity")
        config.update(bind_completed_prefix_run(args.candidate_run_dir, source_commit=source_commit,
                                                inputs=inputs, recipes=recipes))
    elif args.candidate_run_dir.name != config["candidate_run_id"]:
        raise ValueError("Wrong candidate run identity")
    all_rows = verify_candidate_run(args.candidate_run_dir, config, inputs, recipes)
    smoke_ids = smoke_user_ids(sorted(inputs["warmups"]), recipes["episode_config"])
    ids = smoke_ids if args.users == "20" else sorted(inputs["warmups"])
    selected = set(ids)
    rows = [row for row in all_rows if row["provenance"]["kind"] == "pseudo"
            and row["provenance"]["user_id"] in selected]
    provenance = {"source_commit": source_commit,
                  "audit_contract_sha256": file_sha256(contract_path),
                  "candidate_source_commit": config["candidate_source_commit"],
                  "full_report_sha256": config["full_report_sha256"],
                  "full_candidates_sha256": config["full_candidates_sha256"],
                  "snapshot_sha256": inputs["snapshot_sha256"],
                  "split_sha256": inputs["manifest_sha256"],
                  "episode_contract_sha256": recipes["episode_contract_sha256"],
                  "index_manifest_sha256": recipes["index_manifest_sha256"]}
    if args.recipe_version == 2:
        provenance.update({"recipe_version": 2, "base_audit_contract_sha256": base_audit_sha})
    cached = verify_smoke(args.smoke_dir, provenance, smoke_ids) if args.users == "all" else {}
    metadata_path = ROOT / "data/processed/instructrec-books/instructrec-books.meta"
    texts, lengths = metadata_features(metadata_path, max_characters=recipes["config"]["encoder"]["max_text_characters"])
    features = []
    for row in rows:
        episode = row["policy_input"]["episode_id"]
        if episode in cached:
            feature = cached[episode]
            validate_feature_row(feature, row["policy_input"])
        else:
            feature = feature_rows([row], recipes, texts, lengths)[0]
        if args.users == "20" and feature != feature_rows([row], recipes, texts, lengths)[0]:
            raise ValueError("Feature extraction is not reproducible")
        features.append(feature)
    diagnostics = evaluate(features, rows, inputs, config) if args.users == "all" else None
    args.output_dir.mkdir(parents=True)
    feature_path = args.output_dir / "features.jsonl"
    feature_path.write_text("".join(json.dumps(feature, sort_keys=True, separators=(",", ":"), allow_nan=False)
                                    + "\n" for feature in features))
    report = {"status": ("CPU_SHORTCUT_FEATURE_SMOKE_PASS_NOT_STATISTICAL_PROMOTION" if args.users == "20"
                         else "CPU_SHORTCUT_AUDIT_COMPLETE_REVIEW_REQUIRED"),
              "users": len(ids), "query_user_ids": ids, "provenance": provenance,
              "verified_candidate_sets": len(all_rows), "candidate_smoke_sets_reused": 40,
              "features_sha256": file_sha256(feature_path), "smoke_feature_rows_reused": len(cached),
              "warmup_missing_title_users": [row["provenance"]["user_id"] for row in all_rows
                  if row["provenance"]["kind"] == "warmup"
                  and row["reward_audit"]["positive_item_id"] not in recipes["pseudo"].positions],
              "diagnostics": diagnostics, "training_ready": False,
              "full_statistical_shortcut_audit_passed": False,
              "stage_w_memory_proven_safe": False, "real_llm_requests": 0, "gpu_requested": False,
              "original_test_candidates_accessed": False, "original_instruction_accessed": False,
              "original_suffix_item_ids_parsed_or_used": False,
              "gate_decision": ("FEATURE_SMOKE_ONLY" if diagnostics is None else
                                "STOP_BEFORE_PPO_SHORTCUT_REVIEW" if diagnostics["risk_flags"] else
                                "NO_FLAG_IN_FIXED_PROBES_REAL_MEMORY_AND_PPO_GATES_REMAIN"),
              "elapsed_seconds": time.monotonic() - started,
              "numpy_version": np.__version__, "python_version": sys.version}
    (args.output_dir / "report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({key: value for key, value in report.items() if key not in ("query_user_ids", "diagnostics")}), flush=True)
    if diagnostics:
        print(json.dumps(diagnostics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
