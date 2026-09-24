"""One-time sealed evaluation for the MovieLens M7 graph-hard study."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file
from src.temporal_books.current_support import event_key, read_jsonl, successful_calls
from src.temporal_common.metrics import paired_bootstrap_ci, rank_by_scores, residual_ranking
from src.temporal_movielens.graph import VIEWS, load_locked_inputs


SCHEMA_VERSION = 1
BASELINE_ARMS = ("most_popular", "bpr_mf", "sasrec")
GRAPH_ARMS = tuple(
    f"{view}_{signal}_{method}"
    for view in VIEWS
    for signal in ("one_step", "ppr")
    for method in ("graph_only", "residual")
)
RANKING_ARMS = ("local", *GRAPH_ARMS, *BASELINE_ARMS)
PRIMARY_ARM = "exact_one_step_residual"


def event_ndcg(ranking: Sequence[str], gold_item_id: str, k: int) -> float:
    try:
        rank = list(ranking).index(gold_item_id) + 1
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def event_hit(ranking: Sequence[str], gold_item_id: str, k: int) -> float:
    return float(gold_item_id in ranking[:k])


def event_rankings(
    graph_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    local: Sequence[str],
    *,
    alpha: float,
) -> Dict[str, list[str]]:
    candidates = [str(value) for value in graph_row["candidate_item_ids"]]
    if [str(value) for value in baseline_row["candidate_item_ids"]] != candidates:
        raise RuntimeError(f"baseline/graph candidate mismatch: {graph_row['event_key']}")
    local_values = [str(value) for value in local]
    if len(local_values) != len(candidates) or set(local_values) != set(candidates):
        raise RuntimeError(f"invalid local permutation: {graph_row['event_key']}")
    rankings: Dict[str, list[str]] = {"local": local_values}
    for view in VIEWS:
        values = graph_row["views"][view]
        for signal in ("one_step", "ppr"):
            scores = values[f"{signal}_scores"]
            rankings[f"{view}_{signal}_graph_only"] = rank_by_scores(candidates, scores)
            rankings[f"{view}_{signal}_residual"] = residual_ranking(
                candidates, local_values, scores, alpha=alpha
            )
    baseline_scores = baseline_row.get("scores", {})
    for name in BASELINE_ARMS:
        rankings[name] = rank_by_scores(candidates, baseline_scores[name])
    if set(rankings) != set(RANKING_ARMS):
        raise RuntimeError("M7 ranking arm contract mismatch")
    return rankings


def comparison(
    candidate_values: Sequence[float],
    local_values: Sequence[float],
    *,
    changed_rankings: int,
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
        "changed_rankings": changed_rankings,
    }


def analyze(
    events: Sequence[Mapping[str, Any]],
    graph_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    calls: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    graph_by_key = {str(row["event_key"]): row for row in graph_rows}
    baseline_by_key = {str(row["event_key"]): row for row in baseline_rows}
    event_keys = [event_key(event) for event in events]
    if len(set(event_keys)) != len(events):
        raise RuntimeError("duplicate M7 event key")
    if set(graph_by_key) != set(event_keys) or set(baseline_by_key) != set(event_keys):
        raise RuntimeError("M7 score/event cohort mismatch")

    per_ndcg_3: Dict[str, list[float]] = {name: [] for name in RANKING_ARMS}
    per_ndcg_5: Dict[str, list[float]] = {name: [] for name in RANKING_ARMS}
    per_hit = {
        k: {name: [] for name in RANKING_ARMS}
        for k in (1, 3, 5)
    }
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
    alpha = float(config["graph"]["fusion_alpha"])
    for event in events:
        key = event_key(event)
        graph_row = graph_by_key[key]
        baseline_row = baseline_by_key[key]
        candidates = [str(value) for value in event["candidate_item_ids"]]
        if [str(value) for value in graph_row["candidate_item_ids"]] != candidates:
            raise RuntimeError(f"prepared/score candidate mismatch: {key}")
        if graph_row.get("gold_label_used_for_scoring") is not False:
            raise RuntimeError(f"graph score used the M7 label: {key}")
        if baseline_row.get("test_labels_used_for_scoring") is not False:
            raise RuntimeError(f"baseline score used the M7 label: {key}")
        call_key = f"rerank:local:{key}"
        if call_key not in calls:
            raise RuntimeError(f"missing M7 local ranking: {call_key}")
        rankings = event_rankings(
            graph_row, baseline_row, calls[call_key]["value"], alpha=alpha
        )
        gold = str(event["gold_item_id"])
        for name, ranking in rankings.items():
            per_ndcg_3[name].append(event_ndcg(ranking, gold, 3))
            per_ndcg_5[name].append(event_ndcg(ranking, gold, 5))
            for k in (1, 3, 5):
                per_hit[k][name].append(event_hit(ranking, gold, k))
            if name != "local":
                changed[name] += int(ranking != rankings["local"])
        for view in VIEWS:
            scores = graph_row["views"][view]
            coverage[view]["gold_one_step"] += int(
                float(scores["one_step_scores"][gold]) > 0
            )
            coverage[view]["gold_ppr"] += int(float(scores["ppr_scores"][gold]) > 0)
            for item_id in candidates:
                if item_id == gold:
                    continue
                coverage[view]["negative_slots"] += 1
                coverage[view]["negative_one_step"] += int(
                    float(scores["one_step_scores"][item_id]) > 0
                )
                coverage[view]["negative_ppr"] += int(
                    float(scores["ppr_scores"][item_id]) > 0
                )

    count = len(events)
    arms = {
        name: {
            "n_events": count,
            "hit_at_1": sum(per_hit[1][name]) / count,
            "hit_at_3": sum(per_hit[3][name]) / count,
            "hit_at_5": sum(per_hit[5][name]) / count,
            "ndcg_at_3": sum(per_ndcg_3[name]) / count,
            "ndcg_at_5": sum(per_ndcg_5[name]) / count,
            "per_event_ndcg_at_5": per_ndcg_5[name],
        }
        for name in RANKING_ARMS
    }
    comparisons = {
        name: comparison(
            per_ndcg_5[name],
            per_ndcg_5["local"],
            changed_rankings=changed[name],
            resamples=int(config["graph"]["bootstrap_resamples"]),
            seed=int(config["graph"]["bootstrap_seed"]),
        )
        for name in RANKING_ARMS
        if name != "local"
    }
    primary = comparisons[PRIMARY_ARM]
    minimum = float(config["graph"]["admission_min_delta"])
    passed = (
        primary["delta_ndcg_at_5"] >= minimum
        and primary["paired_bootstrap_95_ci"][0] > 0
    )
    for values in coverage.values():
        values["gold_one_step_rate"] = values["gold_one_step"] / count
        values["gold_ppr_rate"] = values["gold_ppr"] / count
        values["negative_one_step_rate"] = (
            values["negative_one_step"] / values["negative_slots"]
        )
        values["negative_ppr_rate"] = values["negative_ppr"] / values["negative_slots"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": count,
        "primary_arm": PRIMARY_ARM,
        "fusion_alpha": alpha,
        "arms": arms,
        "comparisons_vs_local": comparisons,
        "evidence_coverage": coverage,
        "gate": {
            "criteria": {
                "minimum_delta_ndcg_at_5": minimum,
                "paired_bootstrap_95_ci_lower_gt_zero": True,
            },
            "decision": "pass" if passed else "fail",
        },
    }


def artifact_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    study, baselines, self_host = (
        config["study"],
        config["baselines"],
        config["self_host"],
    )
    baseline_manifest = project_path(baselines["manifest"])
    return {
        "prepared": project_path(study["prepared"]),
        "method_lock": project_path(study["method_lock"]),
        "prepare_manifest": project_path(study["prepare_manifest"]),
        "graph_smoke": project_path(study["graph_smoke"]),
        "graph_scores": project_path(study["graph_scores"]),
        "baseline_smoke": project_path(baselines["smoke_manifest"]),
        "baseline_scores": project_path(baselines["scores"]),
        "baseline_manifest": baseline_manifest,
        "bpr_checkpoint": baseline_manifest.with_name("m7_bpr_mf_checkpoint-hnv.pt"),
        "sasrec_checkpoint": baseline_manifest.with_name("m7_sasrec_checkpoint-hnv.pt"),
        "local_attempts": project_path(self_host["attempts"]),
        "local_calls": project_path(self_host["calls"]),
        "local_smoke": project_path(self_host["smoke_manifest"]),
        "local_manifest": project_path(self_host["manifest"]),
    }


def validate_blind_scores(
    graph_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    calls: Mapping[str, Mapping[str, Any]],
    *,
    expected_events: int,
    candidates_per_event: int,
) -> None:
    graph_by_key = {str(row.get("event_key")): row for row in graph_rows}
    baseline_by_key = {str(row.get("event_key")): row for row in baseline_rows}
    if len(graph_by_key) != expected_events or len(graph_rows) != expected_events:
        raise RuntimeError("M7 graph score count/key mismatch")
    if len(baseline_by_key) != expected_events or len(baseline_rows) != expected_events:
        raise RuntimeError("M7 baseline score count/key mismatch")
    if set(graph_by_key) != set(baseline_by_key):
        raise RuntimeError("M7 graph/baseline event mismatch")
    for key, graph_row in graph_by_key.items():
        baseline_row = baseline_by_key[key]
        candidates = [str(value) for value in graph_row.get("candidate_item_ids", [])]
        if len(candidates) != candidates_per_event or len(set(candidates)) != len(candidates):
            raise RuntimeError(f"M7 graph candidate contract mismatch: {key}")
        if graph_row.get("gold_label_used_for_scoring") is not False:
            raise RuntimeError(f"M7 graph scoring was not blind: {key}")
        if set(graph_row.get("views", {})) != set(VIEWS):
            raise RuntimeError(f"M7 graph view mismatch: {key}")
        for view in VIEWS:
            view_scores = graph_row["views"][view]
            for signal in ("one_step_scores", "ppr_scores"):
                scores = view_scores.get(signal, {})
                if set(scores) != set(candidates) or not all(
                    math.isfinite(float(value)) for value in scores.values()
                ):
                    raise RuntimeError(
                        f"M7 graph score contract mismatch: {key}/{view}/{signal}"
                    )
        if [str(value) for value in baseline_row.get("candidate_item_ids", [])] != candidates:
            raise RuntimeError(f"M7 baseline candidate mismatch: {key}")
        if baseline_row.get("test_labels_used_for_scoring") is not False:
            raise RuntimeError(f"M7 baseline scoring was not blind: {key}")
        score_sets = baseline_row.get("scores", {})
        if set(score_sets) != set(BASELINE_ARMS):
            raise RuntimeError(f"M7 baseline arm mismatch: {key}")
        for name in BASELINE_ARMS:
            if set(score_sets[name]) != set(candidates) or not all(
                math.isfinite(float(value)) for value in score_sets[name].values()
            ):
                raise RuntimeError(f"M7 baseline score keys mismatch for {name}: {key}")
        call_key = f"rerank:local:{key}"
        if call_key not in calls:
            raise RuntimeError(f"missing M7 local call: {call_key}")
        ranking = [str(value) for value in calls[call_key].get("value", [])]
        if len(ranking) != candidates_per_event or set(ranking) != set(candidates):
            raise RuntimeError(f"invalid M7 local permutation: {key}")


def validate_score_inputs(
    config: Mapping[str, Any], config_path: Path
) -> tuple[Dict[str, str], int, int]:
    """Validate/hash every blind score without parsing the outcome-bearing cohort."""
    paths = artifact_paths(config)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    config_sha = sha256_file(config_path)
    prepare = json.loads(paths["prepare_manifest"].read_text(encoding="utf-8"))
    if prepare.get("config", {}).get("sha256") != config_sha:
        raise RuntimeError("M7 prepare/config hash mismatch")
    if prepare.get("prepared", {}).get("sha256") != hashes["prepared"]:
        raise RuntimeError("M7 prepared hash mismatch")
    if prepare.get("method_lock", {}).get("sha256") != hashes["method_lock"]:
        raise RuntimeError("M7 method-lock hash mismatch")
    if prepare.get("recommendation_outcomes_evaluated") is not False:
        raise RuntimeError("M7 prepare was not outcome blind")
    method_lock = json.loads(paths["method_lock"].read_text(encoding="utf-8"))
    if method_lock.get("primary_labels_used_for_lock") is not False:
        raise RuntimeError("M7 method lock used primary labels")
    if method_lock.get("manual_tuning_performed") is not False:
        raise RuntimeError("M7 method lock permits manual tuning")
    if method_lock.get("graph") != config["graph"]:
        raise RuntimeError("M7 graph config differs from method lock")
    if method_lock.get("baselines") != config["baselines"]:
        raise RuntimeError("M7 baseline config differs from method lock")

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
            raise RuntimeError(f"M7 graph-smoke mismatch: {key}")

    baseline_smoke = json.loads(paths["baseline_smoke"].read_text(encoding="utf-8"))
    expected_baseline_smoke = {
        "decision": "pass",
        "config_sha256": config_sha,
        "prepared_sha256": hashes["prepared"],
        "method_lock_sha256": hashes["method_lock"],
        "events": int(config["baselines"]["smoke_events"]),
        "manual_tuning_performed": False,
        "recommendation_outcomes_evaluated": False,
        "test_labels_used_for_training_or_scoring": False,
    }
    for key, value in expected_baseline_smoke.items():
        if baseline_smoke.get(key) != value:
            raise RuntimeError(f"M7 baseline-smoke mismatch: {key}")
    baseline_manifest = json.loads(paths["baseline_manifest"].read_text(encoding="utf-8"))
    expected_events = int(config["study"]["primary_events"])
    expected_baseline = {
        "decision": "complete",
        "config_sha256": config_sha,
        "prepared_sha256": hashes["prepared"],
        "method_lock_sha256": hashes["method_lock"],
        "smoke_manifest_sha256": hashes["baseline_smoke"],
        "manual_tuning_performed": False,
        "recommendation_outcomes_evaluated": False,
        "test_labels_used_for_training_or_scoring": False,
    }
    for key, value in expected_baseline.items():
        if baseline_manifest.get(key) != value:
            raise RuntimeError(f"M7 baseline artifact mismatch: {key}")
    if baseline_manifest.get("scores") != {
        "path": config["baselines"]["scores"],
        "records": expected_events,
        "sha256": hashes["baseline_scores"],
    }:
        raise RuntimeError("M7 baseline score manifest mismatch")
    if baseline_manifest.get("checkpoints_sha256") != {
        "bpr_mf": hashes["bpr_checkpoint"],
        "sasrec": hashes["sasrec_checkpoint"],
    }:
        raise RuntimeError("M7 baseline checkpoint mismatch")

    local_manifest = json.loads(paths["local_manifest"].read_text(encoding="utf-8"))
    expected_successes = expected_events * 2
    expected_local = {
        "decision": "complete",
        "config_sha256": config_sha,
        "prepared_sha256": hashes["prepared"],
        "method_lock_sha256": hashes["method_lock"],
        "smoke_manifest_sha256": hashes["local_smoke"],
        "calls_sha256": hashes["local_calls"],
        "attempts_sha256": hashes["local_attempts"],
        "successful_keys": expected_successes,
    }
    for key, value in expected_local.items():
        if local_manifest.get(key) != value:
            raise RuntimeError(f"M7 local artifact mismatch: {key}")

    graph_rows = read_jsonl(paths["graph_scores"])
    baseline_rows = read_jsonl(paths["baseline_scores"])
    calls = successful_calls(paths["local_calls"])
    if len(calls) != expected_successes:
        raise RuntimeError("M7 successful local-call count mismatch")
    if sum(key.startswith("stage_r:") for key in calls) != expected_events:
        raise RuntimeError("M7 Stage-R call count mismatch")
    validate_blind_scores(
        graph_rows,
        baseline_rows,
        calls,
        expected_events=expected_events,
        candidates_per_event=int(config["study"]["candidates_per_event"]),
    )
    return hashes, len(graph_rows), len(calls)


def run_score_lock(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    path = project_path(config["study"]["score_lock"])
    if path.exists():
        raise FileExistsError("M7 score lock already exists")
    hashes, graph_rows, successful = validate_score_inputs(config, config_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "locked",
        "config_sha256": sha256_file(config_path),
        "locked_input_sha256": hashes,
        "graph_rows": graph_rows,
        "baseline_rows": graph_rows,
        "successful_calls": successful,
        "primary_arm": PRIMARY_ARM,
        "recommendation_labels_accessed": False,
        "manual_tuning_performed": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def validate_score_lock(
    config: Mapping[str, Any], config_path: Path
) -> tuple[Dict[str, str], int, int]:
    path = project_path(config["study"]["score_lock"])
    if not path.exists():
        raise RuntimeError("M7 evaluation blocked: create the score lock first")
    lock = json.loads(path.read_text(encoding="utf-8"))
    hashes, graph_rows, successful = validate_score_inputs(config, config_path)
    expected = {
        "decision": "locked",
        "config_sha256": sha256_file(config_path),
        "locked_input_sha256": hashes,
        "graph_rows": graph_rows,
        "baseline_rows": graph_rows,
        "successful_calls": successful,
        "primary_arm": PRIMARY_ARM,
        "recommendation_labels_accessed": False,
        "manual_tuning_performed": False,
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise RuntimeError(f"M7 score-lock mismatch: {key}")
    return hashes, graph_rows, successful


def run_evaluation(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    study = config["study"]
    metrics_path = project_path(study["metrics"])
    manifest_path = project_path(study["evaluation_manifest"])
    if metrics_path.exists() or manifest_path.exists():
        raise FileExistsError("M7 outcomes are sealed; refusing to overwrite evaluation")
    hashes, _, _ = validate_score_lock(config, config_path)
    prepared, _, _ = load_locked_inputs(config)
    paths = artifact_paths(config)
    metrics = analyze(
        prepared["primary_events"],
        read_jsonl(paths["graph_scores"]),
        read_jsonl(paths["baseline_scores"]),
        successful_calls(paths["local_calls"]),
        config,
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    score_lock = project_path(study["score_lock"])
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
            "path": study["score_lock"],
            "sha256": sha256_file(score_lock),
        },
        "metrics": {"path": study["metrics"], "sha256": sha256_file(metrics_path)},
        "outcomes_opened_once": True,
        "test_tuning_after_labels": False,
        "llm_requests_during_evaluation": 0,
        "gpu_used_during_evaluation": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock and evaluate MovieLens M7 once")
    parser.add_argument(
        "--config", default="configs/temporal_movielens32m/m7_graph_hard_end2end.yaml"
    )
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
            "baseline_rows": graph_rows,
            "successful_calls": successful,
            "locked_input_sha256": hashes,
            "outcomes_evaluated": False,
        }
    else:
        result = run_evaluation(config, config_path)
    printable = {key: value for key, value in result.items() if key != "arms"}
    if "arms" in result:
        printable["arms"] = {
            name: {key: value for key, value in arm.items() if key != "per_event_ndcg_at_5"}
            for name, arm in result["arms"].items()
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
