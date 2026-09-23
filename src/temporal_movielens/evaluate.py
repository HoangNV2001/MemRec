"""One-time outcome evaluation for the frozen MovieLens transfer study."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file
from src.temporal_books.current_support import event_key, read_jsonl, successful_calls
from src.temporal_common.metrics import (
    event_hit_at_5,
    event_ndcg_at_5,
    paired_bootstrap_ci,
    rank_by_scores,
    residual_ranking,
)
from src.temporal_movielens.graph import VIEWS, load_locked_inputs


SCHEMA_VERSION = 1
RANKING_ARMS = (
    "local",
    "exact_one_step_graph_only",
    "exact_ppr_graph_only",
    "exact_one_step_residual",
    "exact_ppr_residual",
    "session_300s_one_step_graph_only",
    "session_300s_ppr_graph_only",
    "session_300s_one_step_residual",
    "session_300s_ppr_residual",
)


def burst_bucket(value: float) -> str:
    if value < 0.80:
        return "lt_0.80"
    if value <= 0.95:
        return "0.80_to_0.95"
    return "gt_0.95"


def preceding_gap_bucket(seconds: int) -> str:
    if seconds <= 300:
        raise ValueError("singleton-session target must have a preceding gap above 300 seconds")
    if seconds <= 3600:
        return "5m_to_1h"
    if seconds <= 86400:
        return "1h_to_1d"
    return "gt_1d"


def event_rankings(
    row: Mapping[str, Any], local: Sequence[str], *, alpha: float
) -> Dict[str, list[str]]:
    candidates = [str(value) for value in row["candidate_item_ids"]]
    local_values = [str(value) for value in local]
    if len(local_values) != len(candidates) or set(local_values) != set(candidates):
        raise RuntimeError(f"invalid local permutation for {row['event_key']}")
    rankings: Dict[str, list[str]] = {"local": local_values}
    for view in VIEWS:
        scores = row["views"][view]
        prefix = view
        for signal in ("one_step", "ppr"):
            values = scores[f"{signal}_scores"]
            rankings[f"{prefix}_{signal}_graph_only"] = rank_by_scores(candidates, values)
            rankings[f"{prefix}_{signal}_residual"] = residual_ranking(
                candidates, local_values, values, alpha=alpha
            )
    if set(rankings) != set(RANKING_ARMS):
        raise RuntimeError("MovieLens ranking arm contract mismatch")
    return rankings


def comparison(
    candidate_values: Sequence[float],
    local_values: Sequence[float],
    *,
    changed_events: int,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    delta = [value - base for value, base in zip(candidate_values, local_values)]
    lower, upper = paired_bootstrap_ci(delta, resamples=resamples, seed=seed)
    return {
        "delta_ndcg_at_5": sum(delta) / len(delta),
        "paired_bootstrap_95_ci": [lower, upper],
        "improved_events": sum(value > 0 for value in delta),
        "worsened_events": sum(value < 0 for value in delta),
        "unchanged_events": sum(value == 0 for value in delta),
        "changed_rankings": changed_events,
    }


def descriptive_bucket(
    indices: Sequence[int], per_event: Mapping[str, Sequence[float]]
) -> Dict[str, Any]:
    names = ("local", "exact_ppr_residual", "session_300s_ppr_residual")
    means = {
        name: sum(per_event[name][index] for index in indices) / len(indices)
        for name in names
    }
    return {
        "n_events": len(indices),
        "ndcg_at_5": means,
        "delta_vs_local": {
            "exact_ppr_residual": means["exact_ppr_residual"] - means["local"],
            "session_300s_ppr_residual": means["session_300s_ppr_residual"] - means["local"],
        },
    }


def analyze(
    events: Sequence[Mapping[str, Any]],
    graph_rows: Sequence[Mapping[str, Any]],
    calls: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    graph = config["graph"]
    by_key = {event_key(event): event for event in events}
    rows_by_key = {str(row["event_key"]): row for row in graph_rows}
    if len(by_key) != len(events) or set(rows_by_key) != set(by_key):
        raise RuntimeError("graph/event cohort mismatch")
    alpha = float(graph["fusion_alpha"])
    per_event: Dict[str, list[float]] = {name: [] for name in RANKING_ARMS}
    per_hit: Dict[str, list[float]] = {name: [] for name in RANKING_ARMS}
    changed = {name: 0 for name in RANKING_ARMS if name != "local"}
    coverage = {
        view: {
            "gold_one_step": 0,
            "gold_ppr": 0,
            "negative_one_step": 0,
            "negative_ppr": 0,
            "negative_slots": 0,
        }
        for view in VIEWS
    }
    burst_indices: Dict[str, list[int]] = {
        "lt_0.80": [],
        "0.80_to_0.95": [],
        "gt_0.95": [],
    }
    gap_indices: Dict[str, list[int]] = {
        "5m_to_1h": [],
        "1h_to_1d": [],
        "gt_1d": [],
    }
    ordered_keys = [event_key(event) for event in events]
    for index, key in enumerate(ordered_keys):
        event = by_key[key]
        row = rows_by_key[key]
        gold = str(event["gold_item_id"])
        candidates = [str(value) for value in event["candidate_item_ids"]]
        if row.get("gold_label_used_for_scoring") is not False:
            raise RuntimeError(f"graph score used a gold label: {key}")
        if [str(value) for value in row["candidate_item_ids"]] != candidates:
            raise RuntimeError(f"candidate order mismatch: {key}")
        if set(row.get("views", {})) != set(VIEWS):
            raise RuntimeError(f"graph view mismatch: {key}")
        call_key = f"rerank:local:{key}"
        if call_key not in calls:
            raise RuntimeError(f"missing local ranking: {call_key}")
        rankings = event_rankings(row, calls[call_key]["value"], alpha=alpha)
        for name, ranking in rankings.items():
            per_event[name].append(event_ndcg_at_5(ranking, gold))
            per_hit[name].append(event_hit_at_5(ranking, gold))
            if name != "local":
                changed[name] += int(ranking != rankings["local"])
        for view in VIEWS:
            view_scores = row["views"][view]
            coverage[view]["gold_one_step"] += int(float(view_scores["one_step_scores"][gold]) > 0)
            coverage[view]["gold_ppr"] += int(float(view_scores["ppr_scores"][gold]) > 0)
            for item_id in candidates:
                if item_id == gold:
                    continue
                coverage[view]["negative_slots"] += 1
                coverage[view]["negative_one_step"] += int(
                    float(view_scores["one_step_scores"][item_id]) > 0
                )
                coverage[view]["negative_ppr"] += int(float(view_scores["ppr_scores"][item_id]) > 0)
        burst_indices[burst_bucket(float(event["historical_burstiness_le_60s"]))].append(index)
        gap_indices[preceding_gap_bucket(int(event["previous_gap_seconds"]))].append(index)

    arms = {
        name: {
            "n_events": len(events),
            "ndcg_at_5": sum(per_event[name]) / len(events),
            "hit_at_5": sum(per_hit[name]) / len(events),
            "per_event_ndcg_at_5": per_event[name],
        }
        for name in RANKING_ARMS
    }
    comparisons = {
        name: comparison(
            per_event[name],
            per_event["local"],
            changed_events=changed[name],
            resamples=int(graph["bootstrap_resamples"]),
            seed=int(graph["bootstrap_seed"]),
        )
        for name in RANKING_ARMS
        if name != "local"
    }
    minimum = float(graph["admission_min_delta"])
    exact = comparisons["exact_ppr_residual"]
    session = comparisons["session_300s_ppr_residual"]
    exact_pass = exact["delta_ndcg_at_5"] >= minimum and exact["paired_bootstrap_95_ci"][0] > 0
    session_pass = session["delta_ndcg_at_5"] >= minimum and session["paired_bootstrap_95_ci"][0] > 0
    conclusions = {
        (True, True): "robust_cross_domain_transfer",
        (True, False): "gain_likely_tied_to_rating_entry_sequences",
        (False, True): "session_adaptation_helps_frozen_transfer_failed",
        (False, False): "no_movielens_transfer_evidence",
    }
    for view, values in coverage.items():
        values["gold_one_step_rate"] = values["gold_one_step"] / len(events)
        values["gold_ppr_rate"] = values["gold_ppr"] / len(events)
        values["negative_one_step_rate"] = values["negative_one_step"] / values["negative_slots"]
        values["negative_ppr_rate"] = values["negative_ppr"] / values["negative_slots"]
    oracle_values = [
        max(per_event["local"][index], per_event["exact_ppr_residual"][index], per_event["session_300s_ppr_residual"][index])
        for index in range(len(events))
    ]
    local_mean = arms["local"]["ndcg_at_5"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": len(events),
        "fusion_alpha": alpha,
        "arms": arms,
        "comparisons_vs_local": comparisons,
        "evidence_coverage": coverage,
        "descriptive_buckets": {
            "historical_burstiness_le_60s": {
                name: descriptive_bucket(indices, per_event)
                for name, indices in burst_indices.items()
                if indices
            },
            "target_preceding_gap": {
                name: descriptive_bucket(indices, per_event)
                for name, indices in gap_indices.items()
                if indices
            },
        },
        "posthoc_oracle_best_of_local_exact_session_ppr": {
            "ndcg_at_5": sum(oracle_values) / len(oracle_values),
            "delta_vs_local": sum(oracle_values) / len(oracle_values) - local_mean,
            "deployable": False,
        },
        "gates": {
            "criteria": {"min_delta_ndcg_at_5": minimum, "ci_lower_gt_zero": True},
            "primary_exact_ppr": "pass" if exact_pass else "fail",
            "secondary_session_300s_ppr": "pass" if session_pass else "fail",
            "conclusion": conclusions[(exact_pass, session_pass)],
        },
    }


def artifact_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    study, self_host = config["study"], config["self_host"]
    return {
        "prepared": project_path(study["prepared"]),
        "method_lock": project_path(study["method_lock"]),
        "prepare_manifest": project_path(study["prepare_manifest"]),
        "graph_smoke": project_path(study["graph_smoke"]),
        "graph_scores": project_path(study["graph_scores"]),
        "local_attempts": project_path(self_host["attempts"]),
        "local_calls": project_path(self_host["calls"]),
        "local_smoke": project_path(self_host["smoke_manifest"]),
        "local_manifest": project_path(self_host["manifest"]),
    }


def score_lock_path(config: Mapping[str, Any]) -> Path:
    return project_path(config["study"]["evaluation_manifest"]).with_name("m4_score_lock-hnv.json")


def validate_score_inputs(
    config: Mapping[str, Any], config_path: Path
) -> tuple[Dict[str, str], int, int]:
    """Validate and hash score artifacts without parsing the prepared labels."""
    paths = artifact_paths(config)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    prepare_manifest = json.loads(paths["prepare_manifest"].read_text(encoding="utf-8"))
    if prepare_manifest.get("recommendation_outcomes_evaluated") is not False:
        raise RuntimeError("prepare manifest was not outcome blind")
    config_sha = sha256_file(config_path)
    if prepare_manifest.get("config", {}).get("sha256") != config_sha:
        raise RuntimeError("prepare/config hash mismatch")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if prepare_manifest.get("prepared", {}).get("sha256") != hashes["prepared"]:
        raise RuntimeError("prepared cohort hash mismatch")
    if prepare_manifest.get("method_lock", {}).get("sha256") != hashes["method_lock"]:
        raise RuntimeError("method lock hash mismatch")
    local_manifest = json.loads(paths["local_manifest"].read_text(encoding="utf-8"))
    expected_successes = int(config["study"]["primary_events"]) * 2
    expected_local = {
        "decision": "complete",
        "config_sha256": config_sha,
        "prepared_sha256": prepare_manifest["prepared"]["sha256"],
        "method_lock_sha256": prepare_manifest["method_lock"]["sha256"],
        "smoke_manifest_sha256": hashes["local_smoke"],
        "calls_sha256": hashes["local_calls"],
        "attempts_sha256": hashes["local_attempts"],
        "successful_keys": expected_successes,
    }
    for key, value in expected_local.items():
        if local_manifest.get(key) != value:
            raise RuntimeError(f"local artifact mismatch: {key}")
    graph_smoke = json.loads(paths["graph_smoke"].read_text(encoding="utf-8"))
    expected_graph_smoke = {
        "decision": "pass",
        "config_sha256": config_sha,
        "prepared_sha256": hashes["prepared"],
        "method_lock_sha256": hashes["method_lock"],
        "gold_labels_used_for_scoring": False,
    }
    for key, value in expected_graph_smoke.items():
        if graph_smoke.get(key) != value:
            raise RuntimeError(f"graph smoke mismatch: {key}")
    rows = read_jsonl(paths["graph_scores"])
    if len(rows) != int(config["study"]["primary_events"]):
        raise RuntimeError("graph score count mismatch")
    if len({str(row.get("event_key")) for row in rows}) != len(rows):
        raise RuntimeError("duplicate graph event key")
    for row in rows:
        if row.get("gold_label_used_for_scoring") is not False:
            raise RuntimeError("graph score used a gold label")
        if set(row.get("views", {})) != set(VIEWS):
            raise RuntimeError("graph score view mismatch")
        if len(row.get("candidate_item_ids", [])) != int(config["study"]["candidates_per_event"]):
            raise RuntimeError("graph candidate count mismatch")
    calls = successful_calls(paths["local_calls"])
    if len(calls) != expected_successes:
        raise RuntimeError("successful local call count mismatch")
    if sum(key.startswith("stage_r:") for key in calls) != int(config["study"]["primary_events"]):
        raise RuntimeError("Stage-R call count mismatch")
    if sum(key.startswith("rerank:local:") for key in calls) != int(config["study"]["primary_events"]):
        raise RuntimeError("rerank call count mismatch")
    return hashes, len(rows), len(calls)


def run_score_lock(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    path = score_lock_path(config)
    if path.exists():
        raise FileExistsError("MovieLens score lock already exists")
    hashes, graph_rows, successful = validate_score_inputs(config, config_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "locked",
        "config_sha256": sha256_file(config_path),
        "locked_input_sha256": hashes,
        "graph_rows": graph_rows,
        "successful_calls": successful,
        "recommendation_labels_accessed": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def validate_score_lock(
    config: Mapping[str, Any], config_path: Path
) -> tuple[Dict[str, str], int, int]:
    """Recheck the score lock without parsing the outcome-bearing cohort."""
    lock_path = score_lock_path(config)
    if not lock_path.exists():
        raise RuntimeError("MovieLens evaluation blocked: run --lock before opening outcomes")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    hashes, graph_rows, successful = validate_score_inputs(config, config_path)
    expected_lock = {
        "decision": "locked",
        "config_sha256": sha256_file(config_path),
        "locked_input_sha256": hashes,
        "graph_rows": graph_rows,
        "successful_calls": successful,
        "recommendation_labels_accessed": False,
    }
    for key, value in expected_lock.items():
        if lock.get(key) != value:
            raise RuntimeError(f"score lock mismatch: {key}")
    return hashes, graph_rows, successful


def validate_artifacts(
    config: Mapping[str, Any], config_path: Path
) -> tuple[Dict[str, Any], list[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]]:
    hashes, _, _ = validate_score_lock(config, config_path)
    prepared, _, _ = load_locked_inputs(config)
    paths = artifact_paths(config)
    rows = read_jsonl(paths["graph_scores"])
    calls = successful_calls(paths["local_calls"])
    return prepared, rows, calls, hashes


def run_evaluation(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    study = config["study"]
    metrics_path = project_path(study["metrics"])
    manifest_path = project_path(study["evaluation_manifest"])
    if metrics_path.exists() or manifest_path.exists():
        raise FileExistsError("MovieLens outcomes are sealed; refusing to overwrite one-time evaluation")
    prepared, rows, calls, hashes = validate_artifacts(config, config_path)
    metrics = analyze(prepared["primary_events"], rows, calls, config)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "sealed",
        "config": {
            "path": str(config_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(config_path),
        },
        "locked_input_sha256": hashes,
        "score_lock": {
            "path": str(score_lock_path(config).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(score_lock_path(config)),
        },
        "metrics": {"path": study["metrics"], "sha256": sha256_file(metrics_path)},
        "outcomes_opened_once": True,
        "test_tuning_after_labels": False,
        "llm_requests_during_evaluation": 0,
        "gpu_used_during_evaluation": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen MovieLens temporal graph arms once")
    parser.add_argument("--config", default="configs/temporal_movielens32m/m1_frozen_transfer.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--lock", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.lock:
        result = run_score_lock(config, config_path)
    elif args.validate_only:
        hashes, graph_rows, successful = validate_score_lock(config, config_path)
        result: Dict[str, Any] = {
            "decision": "ready",
            "graph_rows": graph_rows,
            "successful_calls": successful,
            "locked_input_sha256": hashes,
            "outcomes_evaluated": False,
        }
    else:
        result = run_evaluation(config, config_path)
    printable = {
        key: value
        for key, value in result.items()
        if key not in {"arms", "comparisons_vs_local", "descriptive_buckets"}
    }
    if "arms" in result:
        printable["arms"] = {
            name: {key: value for key, value in arm.items() if key != "per_event_ndcg_at_5"}
            for name, arm in result["arms"].items()
        }
        printable["comparisons_vs_local"] = result["comparisons_vs_local"]
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
