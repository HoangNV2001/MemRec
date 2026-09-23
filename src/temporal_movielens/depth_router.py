"""No-manual-tuning graph-depth router for sealed MovieLens M5 scores."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.temporal_books.common import PROJECT_ROOT, jsonl_write, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.current_support import event_key, read_jsonl
from src.temporal_common.metrics import event_hit_at_5, event_ndcg_at_5, paired_bootstrap_ci, rank_by_scores


SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "one_step_nonzero_fraction",
    "ppr_nonzero_fraction",
    "one_step_max_share",
    "ppr_max_share",
    "one_step_top_margin_share",
    "ppr_top_margin_share",
    "one_step_normalized_entropy",
    "ppr_normalized_entropy",
    "distribution_l1_distance",
    "normalized_rank_footrule",
    "top1_agreement",
    "seed_fraction",
)


def validate_contract(config: Mapping[str, Any]) -> None:
    if tuple(config["features"]["names"]) != FEATURE_NAMES:
        raise RuntimeError("depth-router feature contract mismatch")
    model = config["model"]
    expected = {
        "estimator": "standardized_ordinary_least_squares",
        "solver": "numpy_linalg_lstsq",
        "regularization": "none",
        "folds": 5,
        "decision_threshold": 0.0,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise RuntimeError(f"depth-router no-tuning contract mismatch: {key}")


def source_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    study = config["study"]
    return {
        "prepared": project_path(study["source_prepared"]),
        "scores": project_path(study["source_scores"]),
        "score_manifest": project_path(study["source_score_manifest"]),
        "evaluation_manifest": project_path(study["source_evaluation_manifest"]),
    }


def verify_sources(config: Mapping[str, Any], *, include_prepared: bool) -> Dict[str, str]:
    study = config["study"]
    paths = source_paths(config)
    expected = {
        "scores": str(study["source_scores_sha256"]),
        "score_manifest": str(study["source_score_manifest_sha256"]),
        "evaluation_manifest": str(study["source_evaluation_manifest_sha256"]),
    }
    if include_prepared:
        expected["prepared"] = str(study["source_prepared_sha256"])
    hashes: Dict[str, str] = {}
    for name, digest in expected.items():
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[name] = sha256_file(path)
        if hashes[name] != digest:
            raise RuntimeError(f"depth-router source hash mismatch: {name}")
    manifest = json.loads(paths["score_manifest"].read_text(encoding="utf-8"))
    if manifest.get("decision") != "locked":
        raise RuntimeError("M5 score manifest is not locked")
    if manifest.get("recommendation_labels_used_for_scoring") is not False:
        raise RuntimeError("M5 graph scoring was not label blind")
    if manifest["graph_scores"]["sha256"] != hashes["scores"]:
        raise RuntimeError("M5 score-manifest hash mismatch")
    return hashes


def normalized_distribution(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("router scores must be a finite nonnegative vector")
    total = float(values.sum())
    return values / total if total > 0 else np.zeros_like(values)


def normalized_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if len(positive) == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / math.log(len(probabilities)))


def ranking_indices(values: Sequence[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))


def router_features(row: Mapping[str, Any], view: str, *, seed_cap: int) -> Dict[str, float]:
    candidates = [str(value) for value in row["candidate_item_ids"]]
    scores = row["views"][view]
    one_raw = [float(scores["one_step_scores"][item_id]) for item_id in candidates]
    ppr_raw = [float(scores["ppr_scores"][item_id]) for item_id in candidates]
    one = normalized_distribution(one_raw)
    ppr = normalized_distribution(ppr_raw)
    one_sorted = sorted(one, reverse=True)
    ppr_sorted = sorted(ppr, reverse=True)
    one_order, ppr_order = ranking_indices(one_raw), ranking_indices(ppr_raw)
    one_rank = {index: rank for rank, index in enumerate(one_order)}
    ppr_rank = {index: rank for rank, index in enumerate(ppr_order)}
    maximum_footrule = (len(candidates) ** 2) / 2 if len(candidates) % 2 == 0 else (len(candidates) ** 2 - 1) / 2
    values = {
        "one_step_nonzero_fraction": sum(value > 0 for value in one_raw) / len(candidates),
        "ppr_nonzero_fraction": sum(value > 0 for value in ppr_raw) / len(candidates),
        "one_step_max_share": float(one_sorted[0]),
        "ppr_max_share": float(ppr_sorted[0]),
        "one_step_top_margin_share": float(one_sorted[0] - one_sorted[1]),
        "ppr_top_margin_share": float(ppr_sorted[0] - ppr_sorted[1]),
        "one_step_normalized_entropy": normalized_entropy(one),
        "ppr_normalized_entropy": normalized_entropy(ppr),
        "distribution_l1_distance": float(0.5 * np.abs(one - ppr).sum()),
        "normalized_rank_footrule": sum(abs(one_rank[index] - ppr_rank[index]) for index in range(len(candidates))) / maximum_footrule,
        "top1_agreement": float(one_order[0] == ppr_order[0]),
        "seed_fraction": min(len(row["seed_item_ids"]), seed_cap) / seed_cap,
    }
    if tuple(values) != FEATURE_NAMES or any(not math.isfinite(value) for value in values.values()):
        raise RuntimeError("invalid depth-router feature row")
    return values


def feature_rows(config: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    study = config["study"]
    seed_cap = int(config["features"]["seed_cap"])
    rows: list[Dict[str, Any]] = []
    for score in score_rows:
        if score.get("gold_label_used_for_scoring") is not False:
            raise RuntimeError("router source score used a gold label")
        for view in study["views"]:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_key": str(score["event_key"]),
                    "view": str(view),
                    "features": router_features(score, str(view), seed_cap=seed_cap),
                    "gold_label_used": False,
                }
            )
    return rows


def fit_standardized_ols(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) <= x.shape[1]:
        raise ValueError("insufficient OLS training matrix")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "design_rank": int(rank),
        "singular_values": singular_values.tolist(),
        "training_rows": len(x),
    }


def predict_standardized_ols(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return float(model["intercept"]) + ((x - mean) / scale) @ coefficients


def fold_for_key(key: str, config: Mapping[str, Any]) -> int:
    model = config["model"]
    return stable_order(f"{model['fold_salt']}\0{key}") % int(model["folds"])


def verify_smoke(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    path = project_path(config["study"]["feature_smoke"])
    if not path.exists():
        raise RuntimeError("depth-router phase blocked: run --feature-smoke first")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "feature_names": list(FEATURE_NAMES),
        "development_labels_accessed": False,
    }
    for key, value in expected.items():
        if smoke.get(key) != value:
            raise RuntimeError(f"depth-router smoke mismatch: {key}")
    return smoke


def run_feature_smoke(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    validate_contract(config)
    study = config["study"]
    output = project_path(study["feature_smoke"])
    if output.exists():
        raise FileExistsError(output)
    hashes = verify_sources(config, include_prepared=False)
    score_rows = read_jsonl(source_paths(config)["scores"])
    count = int(study["feature_smoke_events"])
    selected = score_rows[:count]
    first = feature_rows(config, selected)
    second = feature_rows(config, selected)
    if first != second:
        raise RuntimeError("depth-router features are not deterministic")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "source_sha256": hashes,
        "events": count,
        "feature_rows": len(first),
        "feature_names": list(FEATURE_NAMES),
        "deterministic_replay": True,
        "development_labels_accessed": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def run_lock_features(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    validate_contract(config)
    study = config["study"]
    smoke = verify_smoke(config, config_path)
    output = project_path(study["features"])
    manifest_path = project_path(study["feature_manifest"])
    if output.exists() or manifest_path.exists():
        raise FileExistsError("depth-router feature lock exists")
    hashes = verify_sources(config, include_prepared=False)
    score_rows = read_jsonl(source_paths(config)["scores"])
    if len(score_rows) != int(study["events"]):
        raise RuntimeError("depth-router source score count mismatch")
    rows = feature_rows(config, score_rows)
    records, digest = jsonl_write(output, rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "locked",
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "feature_smoke_sha256": sha256_file(project_path(study["feature_smoke"])),
        "source_sha256": hashes,
        "features": {"path": study["features"], "records": records, "sha256": digest},
        "feature_names": list(FEATURE_NAMES),
        "model_contract": dict(config["model"]),
        "development_labels_accessed": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_feature_lock(config: Mapping[str, Any], config_path: Path) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    study = config["study"]
    feature_path = project_path(study["features"])
    manifest_path = project_path(study["feature_manifest"])
    for path in (feature_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "decision": "locked",
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "feature_names": list(FEATURE_NAMES),
        "model_contract": dict(config["model"]),
        "development_labels_accessed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"depth-router feature lock mismatch: {key}")
    if manifest["features"]["sha256"] != sha256_file(feature_path):
        raise RuntimeError("depth-router feature hash mismatch")
    return read_jsonl(feature_path), manifest


def evaluate_view(
    view: str,
    events: Sequence[Mapping[str, Any]],
    scores_by_key: Mapping[str, Mapping[str, Any]],
    features_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    x_rows: list[list[float]] = []
    one_values: list[float] = []
    ppr_values: list[float] = []
    one_hits: list[float] = []
    ppr_hits: list[float] = []
    keys: list[str] = []
    for event in events:
        key = event_key(event)
        score = scores_by_key[key]
        candidates = [str(value) for value in score["candidate_item_ids"]]
        gold = str(event["gold_item_id"])
        view_scores = score["views"][view]
        one_ranking = rank_by_scores(candidates, view_scores["one_step_scores"])
        ppr_ranking = rank_by_scores(candidates, view_scores["ppr_scores"])
        one_values.append(event_ndcg_at_5(one_ranking, gold))
        ppr_values.append(event_ndcg_at_5(ppr_ranking, gold))
        one_hits.append(event_hit_at_5(one_ranking, gold))
        ppr_hits.append(event_hit_at_5(ppr_ranking, gold))
        feature = features_by_key[(key, view)]["features"]
        x_rows.append([float(feature[name]) for name in FEATURE_NAMES])
        keys.append(key)
    x = np.asarray(x_rows, dtype=float)
    one = np.asarray(one_values, dtype=float)
    ppr = np.asarray(ppr_values, dtype=float)
    y = ppr - one
    predictions = np.zeros(len(events), dtype=float)
    fold_counts: Dict[str, int] = {}
    fold_models: Dict[str, Any] = {}
    folds = int(config["model"]["folds"])
    assignment = np.asarray([fold_for_key(key, config) for key in keys], dtype=int)
    for fold in range(folds):
        train = assignment != fold
        test = assignment == fold
        if int(train.sum()) <= x.shape[1] or not test.any():
            raise RuntimeError(f"invalid deterministic router fold: {fold}")
        fitted = fit_standardized_ols(x[train], y[train])
        predictions[test] = predict_standardized_ols(fitted, x[test])
        fold_counts[str(fold)] = int(test.sum())
        fold_models[str(fold)] = {
            "training_rows": fitted["training_rows"],
            "design_rank": fitted["design_rank"],
        }
    threshold = float(config["model"]["decision_threshold"])
    choose_ppr = predictions > threshold
    router = np.where(choose_ppr, ppr, one)
    router_hits = np.where(choose_ppr, np.asarray(ppr_hits), np.asarray(one_hits))
    delta = (router - one).tolist()
    lower, upper = paired_bootstrap_ci(
        delta,
        resamples=int(config["model"]["bootstrap_resamples"]),
        seed=int(config["model"]["bootstrap_seed"]),
    )
    mean_delta = float(np.mean(router - one))
    passed = mean_delta >= float(config["model"]["promotion_min_delta"]) and lower > 0
    non_ties = y != 0
    sign_accuracy = float(np.mean((predictions[non_ties] > 0) == (y[non_ties] > 0))) if non_ties.any() else 0.0
    metrics = {
        "one_step_graph_only": {"ndcg_at_5": float(one.mean()), "hit_at_5": float(np.mean(one_hits))},
        "always_ppr_graph_only": {"ndcg_at_5": float(ppr.mean()), "hit_at_5": float(np.mean(ppr_hits))},
        "oof_depth_router": {
            "ndcg_at_5": float(router.mean()),
            "hit_at_5": float(router_hits.mean()),
            "ppr_selected_events": int(choose_ppr.sum()),
            "ppr_selection_rate": float(choose_ppr.mean()),
            "non_tie_sign_accuracy": sign_accuracy,
            "per_event_ndcg_at_5": router.tolist(),
            "per_event_predicted_delta": predictions.tolist(),
        },
        "comparison_vs_one_step": {
            "delta_ndcg_at_5": mean_delta,
            "paired_bootstrap_95_ci": [lower, upper],
            "improved_events": sum(value > 0 for value in delta),
            "worsened_events": sum(value < 0 for value in delta),
            "unchanged_events": sum(value == 0 for value in delta),
        },
        "posthoc_oracle": {
            "ndcg_at_5": float(np.maximum(one, ppr).mean()),
            "delta_vs_one_step": float(np.maximum(one, ppr).mean() - one.mean()),
            "deployable": False,
        },
        "folds": {"counts": fold_counts, "fit_audit": fold_models},
        "gate": "pass" if passed else "fail",
    }
    final_model = fit_standardized_ols(x, y) if passed else None
    return metrics, final_model


def run_evaluate_cv(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    validate_contract(config)
    study = config["study"]
    metrics_path = project_path(study["cv_metrics"])
    manifest_path = project_path(study["cv_manifest"])
    model_path = project_path(study["final_model"])
    if metrics_path.exists() or manifest_path.exists() or model_path.exists():
        raise FileExistsError("depth-router CV output exists")
    feature_rows_value, feature_manifest = load_feature_lock(config, config_path)
    source_hashes = verify_sources(config, include_prepared=True)
    prepared = json.loads(source_paths(config)["prepared"].read_text(encoding="utf-8"))
    score_rows = read_jsonl(source_paths(config)["scores"])
    events = prepared["development_events"]
    if len(events) != int(study["events"]) or len(score_rows) != len(events):
        raise RuntimeError("depth-router event count mismatch")
    scores_by_key = {str(row["event_key"]): row for row in score_rows}
    features_by_key = {(str(row["event_key"]), str(row["view"])): row for row in feature_rows_value}
    expected_feature_keys = {(event_key(event), str(view)) for event in events for view in study["views"]}
    if set(features_by_key) != expected_feature_keys:
        raise RuntimeError("depth-router feature/event mismatch")
    views: Dict[str, Any] = {}
    final_models: Dict[str, Any] = {}
    pass_values: Dict[str, bool] = {}
    for view in study["views"]:
        metrics, final_model = evaluate_view(
            str(view), events, scores_by_key, features_by_key, config
        )
        views[str(view)] = metrics
        pass_values[str(view)] = metrics["gate"] == "pass"
        if final_model is not None:
            final_models[str(view)] = final_model
    decision = {
        (True, True): "promote_both_views",
        (True, False): "promote_exact_only",
        (False, True): "promote_session_only",
        (False, False): "stop_depth_routing",
    }[(pass_values["exact"], pass_values["session_300s"])]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": len(events),
        "feature_names": list(FEATURE_NAMES),
        "model_contract": dict(config["model"]),
        "views": views,
        "gate": {
            "criteria": {
                "min_delta_vs_one_step": float(config["model"]["promotion_min_delta"]),
                "ci_lower_gt_zero": True,
            },
            "exact": "pass" if pass_values["exact"] else "fail",
            "session_300s": "pass" if pass_values["session_300s"] else "fail",
            "decision": decision,
        },
        "manual_tuning_performed": False,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if final_models:
        model_payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": study["run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "locked",
            "feature_names": list(FEATURE_NAMES),
            "decision_threshold": float(config["model"]["decision_threshold"]),
            "models": final_models,
            "development_only": True,
        }
        model_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "sealed",
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes,
        "feature_manifest_sha256": sha256_file(project_path(study["feature_manifest"])),
        "features_sha256": feature_manifest["features"]["sha256"],
        "metrics": {"path": study["cv_metrics"], "sha256": sha256_file(metrics_path)},
        "final_model": {
            "written": bool(final_models),
            "path": study["final_model"] if final_models else None,
            "sha256": sha256_file(model_path) if final_models else None,
            "views": sorted(final_models),
        },
        "development_labels_accessed_once": True,
        "fresh_primary_labels_accessed": False,
        "manual_tuning_performed": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-tuning MovieLens graph-depth router")
    parser.add_argument("--config", default="configs/temporal_movielens32m/m6_depth_router.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--feature-smoke", action="store_true")
    modes.add_argument("--lock-features", action="store_true")
    modes.add_argument("--evaluate-cv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.feature_smoke:
        result = run_feature_smoke(config, config_path)
    elif args.lock_features:
        result = run_lock_features(config, config_path)
    else:
        result = run_evaluate_cv(config, config_path)
    printable = dict(result)
    if "views" in printable:
        printable["views"] = {
            view: {
                key: ({subkey: subvalue for subkey, subvalue in value.items() if not subkey.startswith("per_event_")} if isinstance(value, dict) else value)
                for key, value in metrics.items()
            }
            for view, metrics in result["views"].items()
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
