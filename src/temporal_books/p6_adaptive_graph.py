"""P6: learned depth-adaptive graph scoring with a fresh temporal test."""
from __future__ import annotations

import argparse
import bisect
import csv
import itertools
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Mapping, Sequence

import numpy as np

from src.temporal_books.common import (
    PROJECT_ROOT,
    jsonl_write,
    load_yaml,
    project_path,
    set_csv_field_limit,
    sha256_file,
    stable_order,
)
from src.temporal_books.p2_analysis import paired_bootstrap_ci
from src.temporal_books.p2_oracle import (
    candidate_base_memory,
    event_hit_at_5,
    event_ndcg_at_5,
    load_metadata,
    stable_sample_candidates,
    text,
)
from src.temporal_books.p3_oracle import event_key
from src.temporal_books.p5_candidate_graph import (
    ScoredEvent,
    aggregate_paths,
    candidate_paths,
    rank_by_scores,
    successful_calls,
)


SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86400
FEATURE_NAMES = (
    "has_layer3",
    "log_score_layer3",
    "log_best_layer3",
    "paths_layer3",
    "anchors_layer3",
    "peers_layer3",
    "has_deep_increment",
    "log_score_deep_increment",
    "paths_deep_increment",
    "anchors_deep_increment",
    "peers_deep_increment",
    "inverse_min_depth",
)


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_positive_graph(config: Mapping[str, Any]) -> tuple[Dict[str, list[ScoredEvent]], Dict[str, list[tuple[int, str, float]]]]:
    """Load all positive events; every search still applies strict target-time filters."""
    minimum = float(config["p6"]["positive_rating_min"])
    histories: DefaultDict[str, list[ScoredEvent]] = defaultdict(list)
    item_events: DefaultDict[str, list[tuple[int, str, float]]] = defaultdict(list)
    set_csv_field_limit()
    with project_path(config["dataset"]["reviews_file"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = (row.get("User_id") or "").strip()
            item_id = (row.get("Id") or "").strip()
            if not user_id or not item_id:
                continue
            try:
                timestamp, rating = int(row["review/time"]), float(row["review/score"])
            except (KeyError, TypeError, ValueError):
                continue
            if rating >= minimum:
                histories[user_id].append((timestamp, item_id, rating))
                item_events[item_id].append((timestamp, user_id, rating))
    for values in histories.values():
        values.sort()
    for values in item_events.values():
        values.sort()
    return dict(histories), dict(item_events)


def first_novel_positive_targets(
    histories: Mapping[str, Sequence[ScoredEvent]],
    *,
    start: int,
    end: int | None,
    candidate_pool: set[str],
    minimum_history: int,
    excluded_users: set[str],
) -> list[Dict[str, Any]]:
    targets: list[Dict[str, Any]] = []
    for user_id, events in histories.items():
        if user_id in excluded_users:
            continue
        seen: set[str] = set()
        previous = 0
        for timestamp, batch_iter in itertools.groupby(events, key=lambda event: event[0]):
            batch = list(batch_iter)
            in_window = timestamp >= start and (end is None or timestamp < end)
            eligible = sorted(item_id for _, item_id, _ in batch if item_id not in seen and item_id in candidate_pool)
            if in_window and previous >= minimum_history and eligible:
                targets.append({"user_id": user_id, "timestamp": timestamp, "gold_item_id": eligible[0]})
                break
            if end is None or timestamp < end:
                seen.update(item_id for _, item_id, _ in batch)
                previous += len(batch)
    return targets


def attach_candidates(
    targets: Sequence[Mapping[str, Any]],
    histories: Mapping[str, Sequence[ScoredEvent]],
    candidate_pool: Sequence[str],
    *,
    n_candidates: int,
    prefix: str,
) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    for target in targets:
        user_id, timestamp, gold = str(target["user_id"]), int(target["timestamp"]), str(target["gold_item_id"])
        history = histories[user_id]
        end = bisect.bisect_left(history, (timestamp, "", float("-inf")))
        known = {item_id for _, item_id, _ in history[:end]}
        candidates = stable_sample_candidates(
            candidate_pool,
            known,
            gold,
            n_candidates=n_candidates,
            seed_key=f"{prefix}\0{user_id}\0{timestamp}\0{gold}",
        )
        events.append({**target, "candidate_item_ids": candidates})
    return events


def add_test_prompt_context(
    events: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    by_user = {str(event["user_id"]): int(event["timestamp"]) for event in events}
    needed_items = {str(item_id) for event in events for item_id in event["candidate_item_ids"]}
    titles: Dict[str, str] = {}
    histories: DefaultDict[str, list[tuple[int, str, str, str, str]]] = defaultdict(list)
    set_csv_field_limit()
    with project_path(config["dataset"]["reviews_file"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = (row.get("User_id") or "").strip()
            item_id = (row.get("Id") or "").strip()
            if not item_id:
                continue
            title = (row.get("Title") or "").strip()
            if item_id in needed_items and title and item_id not in titles:
                titles[item_id] = title
            target_time = by_user.get(user_id)
            if target_time is None:
                continue
            try:
                timestamp = int(row["review/time"])
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp < target_time:
                histories[user_id].append(
                    (timestamp, item_id, title, row.get("review/summary") or "", row.get("review/text") or "")
                )
    needed_titles = {titles.get(item_id, "") for item_id in needed_items} - {""}
    metadata = load_metadata(project_path(config["dataset"]["metadata_file"]), needed_titles)
    enriched: list[Dict[str, Any]] = []
    for event in events:
        user_id = str(event["user_id"])
        recent = sorted(histories[user_id])[-6:]
        enriched.append(
            {
                **event,
                "history": [
                    {
                        "timestamp": timestamp,
                        "item_id": item_id,
                        "title": title,
                        "review_summary": text(summary, 180),
                        "review_text": text(review, 360),
                    }
                    for timestamp, item_id, title, summary, review in recent
                ],
            }
        )
    item_info = {
        item_id: {
            "title": titles.get(item_id, f"Item {item_id}"),
            "base_memory": candidate_base_memory(
                titles.get(item_id, f"Item {item_id}"),
                metadata.get(titles.get(item_id, ""), {}),
                420,
            ),
        }
        for item_id in needed_items
    }
    return enriched, item_info


def prepare(config: Mapping[str, Any], config_path: Path, *, force: bool) -> Dict[str, Any]:
    p6 = config["p6"]
    output = project_path(p6["prepared"])
    if output.exists() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    p0 = json.loads(project_path(config["dataset"]["p0_audit"]).read_text(encoding="utf-8"))
    train_cutoff = int(p0["temporal_split"]["train_cutoff"])
    validation_cutoff = int(p0["temporal_split"]["validation_cutoff"])
    inner_start = train_cutoff - int(p6["inner_train_window_days"]) * SECONDS_PER_DAY
    old_validation = json.loads(project_path(p6["old_validation_prepared"]).read_text(encoding="utf-8"))
    old_users = {str(event["user_id"]) for event in old_validation["events"]}
    histories, item_events = load_positive_graph(config)
    candidate_pool = sorted(
        item_id for item_id, values in item_events.items() if values and int(values[0][0]) < train_cutoff
    )
    candidate_pool_set = set(candidate_pool)
    minimum = int(p6["min_positive_history"])
    train_targets = first_novel_positive_targets(
        histories,
        start=inner_start,
        end=train_cutoff,
        candidate_pool=candidate_pool_set,
        minimum_history=minimum,
        excluded_users=old_users,
    )
    train_targets = sorted(
        train_targets,
        key=lambda row: stable_order(f"p6-train\0{row['user_id']}\0{row['timestamp']}\0{row['gold_item_id']}"),
    )[: int(p6["training_events"])]
    train_users = {str(row["user_id"]) for row in train_targets}
    test_targets = first_novel_positive_targets(
        histories,
        start=validation_cutoff,
        end=None,
        candidate_pool=candidate_pool_set,
        minimum_history=minimum,
        excluded_users=old_users | train_users,
    )
    test_targets = sorted(
        test_targets,
        key=lambda row: stable_order(f"p6-test\0{row['user_id']}\0{row['timestamp']}\0{row['gold_item_id']}"),
    )[: int(p6["test_events"])]
    if len(train_targets) != int(p6["training_events"]) or len(test_targets) != int(p6["test_events"]):
        raise RuntimeError(f"insufficient P6 targets: train={len(train_targets)}, test={len(test_targets)}")
    train_events = attach_candidates(
        train_targets,
        histories,
        candidate_pool,
        n_candidates=int(p6["candidates_per_event"]),
        prefix="p6-train-candidates",
    )
    test_events = attach_candidates(
        test_targets,
        histories,
        candidate_pool,
        n_candidates=int(p6["candidates_per_event"]),
        prefix="p6-test-candidates",
    )
    test_events, item_info = add_test_prompt_context(test_events, config)
    prepared = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "reviews": {"path": config["dataset"]["reviews_file"], "sha256": sha256_file(project_path(config["dataset"]["reviews_file"]))},
            "metadata": {"path": config["dataset"]["metadata_file"], "sha256": sha256_file(project_path(config["dataset"]["metadata_file"]))},
            "old_validation_prepared": {"path": p6["old_validation_prepared"], "sha256": sha256_file(project_path(p6["old_validation_prepared"]))},
        },
        "protocol": {
            "train_window": [inner_start, train_cutoff],
            "test_start": validation_cutoff,
            "strict_past_graph": True,
            "positive_rating_min": float(p6["positive_rating_min"]),
            "candidate_pool": "positive items first observed before global train cutoff",
            "candidate_sampling": "gold plus nine deterministic uniform unseen negatives",
            "old_validation_role": "calibration only",
        },
        "candidate_pool_items": len(candidate_pool),
        "train_events": train_events,
        "test_events": test_events,
        "item_info": item_info,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return prepared


def graph_candidate_features(
    event: Mapping[str, Any],
    histories: Mapping[str, Sequence[ScoredEvent]],
    item_events: Mapping[str, Sequence[tuple[int, str, float]]],
    p6: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for candidate in event["candidate_item_ids"]:
        paths = candidate_paths(
            histories,
            item_events,
            source_user=str(event["user_id"]),
            target_time=int(event["timestamp"]),
            candidate_item=str(candidate),
            minimum_rating=float(p6["positive_rating_min"]),
            half_life_days=float(p6["recency_half_life_days"]),
            length_decay=float(p6["length_decay"]),
            hub_penalty_weight=float(p6["hub_penalty_weight"]),
            source_anchor_limit=int(p6["source_anchor_cap"]),
            peers_per_item=int(p6["peers_per_item"]),
            items_per_peer=int(p6["items_per_peer"]),
            beam_width=int(p6["beam_width"]),
            max_item_layers=int(p6["deep_item_layers"]),
        )
        rows[str(candidate)] = {
            "layer3": aggregate_paths(
                paths,
                max_item_layers=int(p6["shallow_item_layers"]),
                max_paths=int(p6["max_paths_per_candidate"]),
            ),
            "layer5": aggregate_paths(
                paths,
                max_item_layers=int(p6["deep_item_layers"]),
                max_paths=int(p6["max_paths_per_candidate"]),
            ),
        }
    return rows


def _positive_log(value: float) -> float:
    return math.log(value) if value > 0 else 0.0


def feature_vector(row: Mapping[str, Any]) -> list[float]:
    shallow, deep = row["layer3"], row["layer5"]
    shallow_score, deep_score = float(shallow["score"]), float(deep["score"])
    increment = max(0.0, deep_score - shallow_score)
    min_depth = deep.get("min_item_layers")
    return [
        float(shallow_score > 0),
        _positive_log(shallow_score),
        _positive_log(float(shallow["best_path_score"])),
        float(shallow["selected_paths"]) / 8.0,
        float(shallow["independent_anchors"]) / 8.0,
        float(shallow["candidate_peers"]) / 8.0,
        float(increment > 0),
        _positive_log(increment),
        max(0.0, float(deep["selected_paths"]) - float(shallow["selected_paths"])) / 8.0,
        max(0.0, float(deep["independent_anchors"]) - float(shallow["independent_anchors"])) / 8.0,
        max(0.0, float(deep["candidate_peers"]) - float(shallow["candidate_peers"])) / 8.0,
        1.0 / float(min_depth) if min_depth else 0.0,
    ]


def score_graph_events(
    events: Sequence[Mapping[str, Any]],
    histories: Mapping[str, Sequence[ScoredEvent]],
    item_events: Mapping[str, Sequence[tuple[int, str, float]]],
    p6: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "event_key": event_key(event),
            "user_id": str(event["user_id"]),
            "timestamp": int(event["timestamp"]),
            "candidate_item_ids": [str(value) for value in event["candidate_item_ids"]],
            "candidate_features": graph_candidate_features(event, histories, item_events, p6),
        }
        for event in events
    ]


def fit_pairwise_model(rows: Sequence[Mapping[str, Any]], events_by_key: Mapping[str, Mapping[str, Any]], p6: Mapping[str, Any]) -> Dict[str, Any]:
    matrix = np.asarray(
        [feature_vector(row["candidate_features"][item]) for row in rows for item in row["candidate_item_ids"]],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    differences: list[np.ndarray] = []
    for row in rows:
        event = events_by_key[str(row["event_key"])]
        gold = str(event["gold_item_id"])
        positive = (np.asarray(feature_vector(row["candidate_features"][gold])) - means) / scales
        for item in row["candidate_item_ids"]:
            if item != gold:
                negative = (np.asarray(feature_vector(row["candidate_features"][item])) - means) / scales
                differences.append(positive - negative)
    design = np.asarray(differences, dtype=np.float64)
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    ridge = float(p6["pairwise_ridge_l2"])
    identity = np.eye(len(FEATURE_NAMES), dtype=np.float64)
    iterations = 0
    for iterations in range(1, int(p6["pairwise_newton_steps"]) + 1):
        margins = np.clip(design @ weights, -40.0, 40.0)
        losses = 1.0 / (1.0 + np.exp(margins))
        gradient = -(design.T @ losses) / len(design) + ridge * weights
        curvature = losses * (1.0 - losses)
        hessian = (design.T @ (design * curvature[:, None])) / len(design) + ridge * identity
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {
        "feature_names": list(FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": weights.tolist(),
        "ridge_l2": ridge,
        "newton_iterations": iterations,
        "training_pairs": len(design),
    }


def learned_scores(candidate_features: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, float]:
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    return {
        item_id: float(((np.asarray(feature_vector(row)) - means) / scales) @ weights)
        for item_id, row in candidate_features.items()
    }


def adaptive_ranking(
    candidates: Sequence[str],
    local: Sequence[str],
    graph_scores: Mapping[str, float],
    *,
    alpha: float,
    margin_gate: float,
) -> tuple[list[str], float, bool]:
    ordered = sorted((float(graph_scores[item]), item) for item in candidates)
    margin = ordered[-1][0] - ordered[-2][0] if len(ordered) > 1 else 0.0
    active = alpha > 0 and margin >= margin_gate and ordered[-1][0] > ordered[0][0]
    if not active:
        return list(local), margin, False
    minimum, maximum = ordered[0][0], ordered[-1][0]
    local_scores = {item: 1.0 / math.log2(index + 2) for index, item in enumerate(local)}
    combined = {
        item: local_scores[item] + alpha * (float(graph_scores[item]) - minimum) / (maximum - minimum)
        for item in candidates
    }
    return rank_by_scores(candidates, combined), margin, True


def validation_inputs(config: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    p6 = config["p6"]
    for key, hash_key in (
        ("old_validation_prepared", "old_validation_prepared_sha256"),
        ("old_validation_calls", "old_validation_calls_sha256"),
        ("old_validation_graph_scores", "old_validation_graph_scores_sha256"),
    ):
        if sha256_file(project_path(p6[key])) != str(p6[hash_key]):
            raise RuntimeError(f"P6 frozen validation hash mismatch: {key}")
    prepared = json.loads(project_path(p6["old_validation_prepared"]).read_text(encoding="utf-8"))
    calls = successful_calls(project_path(p6["old_validation_calls"]))
    rows = {str(row["event_key"]): row for row in read_jsonl(project_path(p6["old_validation_graph_scores"]))}
    return list(prepared["events"]), calls, rows


def calibrate(model: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    events, calls, rows = validation_inputs(config)
    p6 = config["p6"]
    grid: list[Dict[str, Any]] = []
    for alpha in [float(value) for value in p6["fusion_alpha_grid"]]:
        for gate in [float(value) for value in p6["margin_gate_grid"]]:
            ndcgs, hits, active = [], [], 0
            for event in events:
                key = event_key(event)
                row = rows[key]
                local = calls[f"rerank:local:{key}"]["value"]
                scores = learned_scores(row["candidate_features"], model)
                ranking, _, used = adaptive_ranking(
                    row["candidate_item_ids"], local, scores, alpha=alpha, margin_gate=gate
                )
                gold = str(event["gold_item_id"])
                ndcgs.append(event_ndcg_at_5(ranking, gold))
                hits.append(event_hit_at_5(ranking, gold))
                active += int(used)
            grid.append(
                {
                    "alpha": alpha,
                    "margin_gate": gate,
                    "ndcg_at_5": sum(ndcgs) / len(ndcgs),
                    "hit_at_5": sum(hits) / len(hits),
                    "active_events": active,
                }
            )
    selected = sorted(grid, key=lambda row: (-row["ndcg_at_5"], -row["margin_gate"], row["alpha"]))[0]
    local_ndcgs = [
        event_ndcg_at_5(calls[f"rerank:local:{event_key(event)}"]["value"], str(event["gold_item_id"]))
        for event in events
    ]
    return {
        "role": "old repeatedly-observed cohort; calibration only",
        "events": len(events),
        "local_ndcg_at_5": sum(local_ndcgs) / len(local_ndcgs),
        "grid": grid,
        "selected": selected,
    }


def verify_graph_smoke(config: Mapping[str, Any], prepared_path: Path) -> Dict[str, Any]:
    p6 = config["p6"]
    path = project_path(p6["graph_smoke"])
    if not path.exists():
        raise RuntimeError("P6 full graph work blocked: run --graph-smoke first")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "decision": "pass",
        "prepared_sha256": sha256_file(prepared_path),
        "config_sha256": sha256_file(project_path("configs/temporal_amazon_books_2014/p6_adaptive_graph.yaml")),
    }
    for key, value in expected.items():
        if smoke.get(key) != value:
            raise RuntimeError(f"P6 graph smoke mismatch: {key}")
    return smoke


def run_graph_smoke(config: Mapping[str, Any], config_path: Path, prepared: Mapping[str, Any]) -> None:
    p6 = config["p6"]
    histories, item_events = load_positive_graph(config)
    train_events = sorted(prepared["train_events"], key=lambda event: stable_order(f"p6-smoke-train\0{event_key(event)}"))[
        : int(p6["graph_smoke_train_events"])
    ]
    test_events = sorted(prepared["test_events"], key=lambda event: stable_order(f"p6-smoke-test\0{event_key(event)}"))[
        : int(p6["graph_smoke_test_events"])
    ]
    started = time.monotonic()
    rows = score_graph_events([*train_events, *test_events], histories, item_events, p6)
    vectors = [feature_vector(row["candidate_features"][item]) for row in rows for item in row["candidate_item_ids"]]
    if not vectors or not np.isfinite(np.asarray(vectors)).all():
        raise RuntimeError("P6 graph smoke produced empty/non-finite features")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p6["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": sha256_file(project_path(p6["prepared"])),
        "train_event_keys": [event_key(event) for event in train_events],
        "test_event_keys": [event_key(event) for event in test_events],
        "candidate_slots": sum(len(row["candidate_item_ids"]) for row in rows),
        "slots_with_layer3": sum(row["candidate_features"][item]["layer3"]["score"] > 0 for row in rows for item in row["candidate_item_ids"]),
        "slots_with_layer5": sum(row["candidate_features"][item]["layer5"]["score"] > 0 for row in rows for item in row["candidate_item_ids"]),
        "gold_labels_used_for_graph_scoring": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    path = project_path(p6["graph_smoke"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_fit(config: Mapping[str, Any], config_path: Path, prepared: Mapping[str, Any], *, force: bool) -> None:
    p6 = config["p6"]
    prepared_path = project_path(p6["prepared"])
    smoke = verify_graph_smoke(config, prepared_path)
    train_scores_path = project_path(p6["train_scores"])
    model_path = project_path(p6["locked_model"])
    manifest_path = project_path(p6["fit_manifest"])
    if any(path.exists() for path in (train_scores_path, model_path, manifest_path)) and not force:
        raise FileExistsError("P6 fit output exists; use --force only for an intentional rerun")
    histories, item_events = load_positive_graph(config)
    started = time.monotonic()
    rows = score_graph_events(prepared["train_events"], histories, item_events, p6)
    train_scores_path.parent.mkdir(parents=True, exist_ok=True)
    records, scores_sha = jsonl_write(train_scores_path, rows)
    events_by_key = {event_key(event): event for event in prepared["train_events"]}
    model = fit_pairwise_model(rows, events_by_key, p6)
    model.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": p6["run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration": calibrate(model, config),
        }
    )
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p6["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "prepared": {"path": p6["prepared"], "sha256": sha256_file(prepared_path)},
        "graph_smoke": {"path": p6["graph_smoke"], "sha256": sha256_file(project_path(p6["graph_smoke"])), "decision": smoke["decision"]},
        "train_scores": {"path": p6["train_scores"], "records": records, "sha256": scores_sha},
        "locked_model": {"path": p6["locked_model"], "sha256": sha256_file(model_path)},
        "test_labels_accessed_by_fit": False,
        "llm_requests": 0,
        "gpu_used": False,
        "runtime_seconds": time.monotonic() - started,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"training_events": records, "selected": model["calibration"]["selected"], "weights": dict(zip(FEATURE_NAMES, model["weights"])), "runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False, indent=2))


def analyze_test(rows: Sequence[Mapping[str, Any]], events_by_key: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    local, adaptive, hits_local, hits_adaptive = [], [], [], []
    active = 0
    for row in rows:
        event = events_by_key[str(row["event_key"])]
        gold = str(event["gold_item_id"])
        local.append(event_ndcg_at_5(row["rankings"]["local"], gold))
        adaptive.append(event_ndcg_at_5(row["rankings"]["adaptive"], gold))
        hits_local.append(event_hit_at_5(row["rankings"]["local"], gold))
        hits_adaptive.append(event_hit_at_5(row["rankings"]["adaptive"], gold))
        active += int(row["graph_active"])
    delta = [value - base for value, base in zip(adaptive, local)]
    p6 = config["p6"]
    lower, upper = paired_bootstrap_ci(delta, resamples=int(p6["bootstrap_resamples"]), seed=int(p6["bootstrap_seed"]))
    mean_delta = sum(delta) / len(delta)
    passed = mean_delta >= float(p6["admission_min_delta"]) and lower > 0
    oracle = [max(a, b) for a, b in zip(local, adaptive)]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": p6["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": {
            "fresh_local": {"n_events": len(local), "ndcg_at_5": sum(local) / len(local), "hit_at_5": sum(hits_local) / len(hits_local), "per_event_ndcg_at_5": local},
            "adaptive_multihop": {"n_events": len(adaptive), "ndcg_at_5": sum(adaptive) / len(adaptive), "hit_at_5": sum(hits_adaptive) / len(hits_adaptive), "per_event_ndcg_at_5": adaptive, "active_events": active},
        },
        "comparison": {
            "delta_ndcg_at_5": mean_delta,
            "paired_bootstrap_95_ci": [lower, upper],
            "improved_events": sum(value > 0 for value in delta),
            "worsened_events": sum(value < 0 for value in delta),
            "unchanged_events": sum(value == 0 for value in delta),
        },
        "posthoc_oracle_best_of_two": {"ndcg_at_5": sum(oracle) / len(oracle), "delta_vs_local": sum(oracle) / len(oracle) - sum(local) / len(local), "deployable": False},
        "gate": {"criteria": {"min_delta": float(p6["admission_min_delta"]), "ci_lower_gt_zero": True}, "decision": "pass" if passed else "hard_stop"},
    }


def run_evaluate(config: Mapping[str, Any], config_path: Path, prepared: Mapping[str, Any], *, force: bool) -> None:
    p6, self_host = config["p6"], config["self_host"]
    verify_graph_smoke(config, project_path(p6["prepared"]))
    model_path = project_path(p6["locked_model"])
    fit_manifest_path = project_path(p6["fit_manifest"])
    local_manifest_path = project_path(self_host["manifest"])
    for path in (model_path, fit_manifest_path, local_manifest_path, project_path(self_host["calls"])):
        if not path.exists():
            raise FileNotFoundError(path)
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    if local_manifest.get("decision") != "complete" or local_manifest.get("prepared_sha256") != sha256_file(project_path(p6["prepared"])):
        raise RuntimeError("P6 local LLM manifest does not match prepared test")
    scores_path, metrics_path, manifest_path = (project_path(p6[key]) for key in ("test_graph_scores", "metrics", "manifest"))
    if any(path.exists() for path in (scores_path, metrics_path, manifest_path)) and not force:
        raise FileExistsError("P6 test output exists; use --force only for an intentional rerun")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    selected = model["calibration"]["selected"]
    calls = successful_calls(project_path(self_host["calls"]))
    histories, item_events = load_positive_graph(config)
    started = time.monotonic()
    feature_rows = score_graph_events(prepared["test_events"], histories, item_events, p6)
    rows: list[Dict[str, Any]] = []
    for row in feature_rows:
        key = str(row["event_key"])
        local = calls[f"rerank:local:{key}"]["value"]
        graph = learned_scores(row["candidate_features"], model)
        ranking, margin, active = adaptive_ranking(
            row["candidate_item_ids"],
            local,
            graph,
            alpha=float(selected["alpha"]),
            margin_gate=float(selected["margin_gate"]),
        )
        rows.append({**row, "model_scores": graph, "model_margin": margin, "graph_active": active, "rankings": {"local": local, "adaptive": ranking}})
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    records, scores_sha = jsonl_write(scores_path, rows)
    events_by_key = {event_key(event): event for event in prepared["test_events"]}
    metrics = analyze_test(rows, events_by_key, config)
    metrics["locked_selection"] = selected
    metrics["runtime_seconds"] = time.monotonic() - started
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p6["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "prepared": {"path": p6["prepared"], "sha256": sha256_file(project_path(p6["prepared"]))},
        "fit_manifest": {"path": p6["fit_manifest"], "sha256": sha256_file(fit_manifest_path)},
        "locked_model": {"path": p6["locked_model"], "sha256": sha256_file(model_path)},
        "local_manifest": {"path": self_host["manifest"], "sha256": sha256_file(local_manifest_path)},
        "test_scores": {"path": p6["test_graph_scores"], "records": records, "sha256": scores_sha},
        "metrics": {"path": p6["metrics"], "sha256": sha256_file(metrics_path)},
        "test_tuning_after_labels": False,
        "graph_llm_requests": 0,
        "graph_gpu_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    printable_arms = {name: {key: value for key, value in arm.items() if key != "per_event_ndcg_at_5"} for name, arm in metrics["arms"].items()}
    print(json.dumps({"arms": printable_arms, "comparison": metrics["comparison"], "oracle": metrics["posthoc_oracle_best_of_two"], "gate": metrics["gate"], "selection": selected}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P6 learned adaptive multi-hop scorer")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p6_adaptive_graph.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--graph-smoke", action="store_true")
    modes.add_argument("--fit", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    p6 = config["p6"]
    prepared = prepare(config, config_path, force=args.force if args.prepare else False)
    if args.prepare or not any((args.graph_smoke, args.fit, args.evaluate)):
        print(json.dumps({"prepared": p6["prepared"], "train_events": len(prepared["train_events"]), "test_events": len(prepared["test_events"]), "llm_requests": 0}, ensure_ascii=False, indent=2))
    elif args.graph_smoke:
        run_graph_smoke(config, config_path, prepared)
    elif args.fit:
        run_fit(config, config_path, prepared, force=args.force)
    elif args.evaluate:
        run_evaluate(config, config_path, prepared, force=args.force)


if __name__ == "__main__":
    main()
