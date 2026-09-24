"""Fresh user-disjoint graph-hard cohort for the final MovieLens ranking study."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.current_support import event_key
from src.temporal_movielens.data import (
    eligible_singleton_targets,
    historical_burstiness,
    iter_user_ratings,
    load_movies,
    movie_memory,
    positive_candidate_pool,
    select_one_target_per_user,
    strict_history,
)
from src.temporal_movielens.graph import build_view, event_seeds
from src.temporal_movielens.hard_headroom import (
    graph_hard_candidate_event,
    development_scan,
    one_step_all_scores,
)
from src.temporal_movielens.prepare import smoke_adapter, verify_audit


SCHEMA_VERSION = 1


def _verified_json(path_value: str, expected_sha256: str) -> Dict[str, Any]:
    path = project_path(path_value)
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"sealed artifact hash mismatch: {path_value}")
    return json.loads(path.read_text(encoding="utf-8"))


def prior_user_exclusions(config: Mapping[str, Any]) -> tuple[set[str], Dict[str, Any]]:
    """Recover every prior MovieLens cohort/scan user without reading metrics."""
    study = config["study"]
    m1 = _verified_json(
        str(study["sealed_m1_prepared"]), str(study["sealed_m1_prepared_sha256"])
    )
    m1_users = {
        str(event["user_id"])
        for event in [*m1["development_events"], *m1["primary_events"]]
    }
    if len(m1_users) != len(m1["development_events"]) + len(m1["primary_events"]):
        raise RuntimeError("sealed M1 cohort users are not disjoint")

    m5_config_path = project_path(study["sealed_m5_config"])
    if sha256_file(m5_config_path) != str(study["sealed_m5_config_sha256"]):
        raise RuntimeError("sealed M5 config hash mismatch")
    _verified_json(
        str(study["sealed_m5_prepared"]), str(study["sealed_m5_prepared_sha256"])
    )
    m5_config = load_yaml(m5_config_path)
    m5_scan, _, m5_scan_stats = development_scan(m5_config)
    m5_users = {str(event["user_id"]) for event in m5_scan}
    if len(m5_users) != int(m5_config["study"]["development_scan_users"]):
        raise RuntimeError("M5 fixed scan exclusion count mismatch")
    if m1_users & m5_users:
        raise RuntimeError("sealed M1 and M5 fixed-scan users unexpectedly overlap")
    users = m1_users | m5_users
    return users, {
        "m1_users": len(m1_users),
        "m5_fixed_scan_users": len(m5_users),
        "total_excluded_users": len(users),
        "m5_scan_stats": m5_scan_stats,
    }


def primary_scan(
    config: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], set[str], Dict[str, Any]]:
    dataset, study = config["dataset"], config["study"]
    audit = verify_audit(config)
    ratings_path = project_path(dataset["ratings_file"])
    excluded, exclusion_stats = prior_user_exclusions(config)
    pool = set(
        positive_candidate_pool(
            ratings_path,
            train_cutoff=int(audit["temporal_split"]["train_cutoff"]),
            positive_rating_min=float(study["positive_rating_min"]),
        )
    )
    targets: list[Dict[str, Any]] = []
    for user_id, events in iter_user_ratings(ratings_path):
        if user_id in excluded:
            continue
        eligible = eligible_singleton_targets(
            user_id,
            events,
            start=int(audit["temporal_split"]["validation_cutoff"]),
            end=None,
            candidate_pool=pool,
            session_gap_seconds=int(study["session_gap_seconds"]),
            positive_rating_min=float(study["positive_rating_min"]),
            minimum_positive_history=int(study["minimum_positive_history"]),
        )
        if eligible:
            targets.append(select_one_target_per_user(eligible, salt=str(study["target_salt"])))
    targets.sort(
        key=lambda row: stable_order(
            f"{study['scan_salt']}\0{row['user_id']}\0{row['timestamp']}\0{row['gold_item_id']}"
        )
    )
    scan_limit = int(study["primary_scan_users"])
    selected = targets[:scan_limit]
    if len(selected) != scan_limit:
        raise RuntimeError(f"insufficient fresh primary scan users: {len(selected)}")
    by_user = {str(row["user_id"]): row for row in selected}
    enriched: list[Dict[str, Any]] = []
    for user_id, events in iter_user_ratings(ratings_path):
        target = by_user.get(user_id)
        if target is None:
            continue
        history = strict_history(events, int(target["timestamp"]))
        recent = history[-int(study["history_events"]):]
        enriched.append(
            {
                **target,
                "history": [
                    {"timestamp": timestamp, "item_id": item_id, "rating": rating}
                    for timestamp, item_id, rating in recent
                ],
                "historical_burstiness_le_60s": historical_burstiness(
                    events, int(audit["temporal_split"]["validation_cutoff"])
                ),
                "_known_item_ids": sorted({item_id for _, item_id, _ in history}, key=int),
            }
        )
    by_key = {event_key(event): event for event in enriched}
    ordered = [by_key[event_key(target)] for target in selected]
    if len(ordered) != scan_limit:
        raise RuntimeError("failed to enrich fixed fresh primary scan")
    return ordered, pool, {
        **exclusion_stats,
        "eligible_primary_users_after_exclusion": len(targets),
        "fixed_scan_users": len(ordered),
        "candidate_pool_items": len(pool),
    }


def candidate_cohort(
    config: Mapping[str, Any], count: int
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, str]], Dict[str, Any]]:
    audit = verify_audit(config)
    dataset, study, graph = config["dataset"], config["study"], config["graph"]
    scan, pool, scan_stats = primary_scan(config)
    ratings_path = project_path(dataset["ratings_file"])
    cutoff = int(audit["temporal_split"]["validation_cutoff"])
    exact, exact_stats, exact_seconds = build_view(
        ratings_path, cutoff=cutoff, session_gap_seconds=int(graph["exact_session_gap_seconds"])
    )
    session, session_stats, session_seconds = build_view(
        ratings_path,
        cutoff=cutoff,
        session_gap_seconds=int(graph["robustness_session_gap_seconds"]),
    )
    feasible: list[Dict[str, Any]] = []
    for event in scan:
        seeds = event_seeds(event, int(graph["seed_items"]))
        candidate = graph_hard_candidate_event(
            event,
            one_step_all_scores(seeds, exact),
            one_step_all_scores(seeds, session),
            pool,
            study,
        )
        if candidate is not None:
            feasible.append(candidate)
    selected = feasible[:count]
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} fresh graph-hard events in fixed scan")

    movies = load_movies(project_path(dataset["movies_file"]))
    needed_items = {
        str(item_id)
        for event in selected
        for item_id in [
            *event["candidate_item_ids"],
            *(row["item_id"] for row in event["history"]),
        ]
    }
    missing = needed_items - set(movies)
    if missing:
        raise RuntimeError(f"MovieLens metadata missing for {len(missing)} M7 items")
    for event in selected:
        event["history"] = [
            {
                **row,
                "title": movies[str(row["item_id"])]["title"],
                "genres": movies[str(row["item_id"])]["genres"],
            }
            for row in event["history"]
        ]
    item_info = {
        movie_id: {
            "title": movies[movie_id]["title"],
            "genres": movies[movie_id]["genres"],
            "base_memory": movie_memory(movies[movie_id]),
        }
        for movie_id in sorted(
            {str(item_id) for event in selected for item_id in event["candidate_item_ids"]},
            key=int,
        )
    }
    validate_prepared_events(selected, config)
    stats = {
        **scan_stats,
        "graph_hard_feasible_users": len(feasible),
        "selected_events": len(selected),
        "rejected_for_fewer_than_nine_reachable_negatives": len(scan) - len(feasible),
        "views": {
            "exact": {
                "session_gap_seconds": int(graph["exact_session_gap_seconds"]),
                "graph_stats": exact_stats,
                "build_seconds": exact_seconds,
            },
            "session_300s": {
                "session_gap_seconds": int(graph["robustness_session_gap_seconds"]),
                "graph_stats": session_stats,
                "build_seconds": session_seconds,
            },
        },
    }
    del exact, session
    gc.collect()
    return selected, item_info, stats


def validate_prepared_events(events: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    study = config["study"]
    if len({str(event["user_id"]) for event in events}) != len(events):
        raise RuntimeError("M7 one-event-per-user contract violated")
    for event in events:
        candidates = [str(value) for value in event["candidate_item_ids"]]
        known = {str(row["item_id"]) for row in event["history"]}
        if len(candidates) != int(study["candidates_per_event"]) or len(set(candidates)) != len(candidates):
            raise RuntimeError("M7 candidate uniqueness contract violated")
        if str(event["gold_item_id"]) not in candidates:
            raise RuntimeError("M7 gold absent from candidates")
        if set(candidates) & known or event["candidate_history_overlap"]:
            raise RuntimeError("M7 candidate/history collision")
        if len(event["negative_item_ids"]) != int(study["candidates_per_event"]) - 1:
            raise RuntimeError("M7 negative count mismatch")
        if any(int(row["timestamp"]) >= int(event["timestamp"]) for row in event["history"]):
            raise RuntimeError("M7 strict-past history violation")


def verify_candidate_smoke(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    path = project_path(config["study"]["candidate_smoke"])
    if not path.exists():
        raise RuntimeError("M7 prepare blocked: run candidate smoke first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "recommendation_outcomes_evaluated": False,
        "gold_labels_used_for_graph_scoring": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"M7 candidate smoke mismatch: {key}")
    return payload


def run_candidate_smoke(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    output = project_path(config["study"]["candidate_smoke"])
    if output.exists():
        raise FileExistsError(output)
    count = int(config["graph"]["graph_smoke_events"])
    events, _, stats = candidate_cohort(config, count)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "events": len(events),
        "candidate_slots": sum(len(event["candidate_item_ids"]) for event in events),
        "negative_slots_with_one_step_evidence_in_both_views": len(events) * 9,
        "cohort_stats": stats,
        "recommendation_outcomes_evaluated": False,
        "gold_labels_used_for_graph_scoring": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def run_prepare(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    study = config["study"]
    verify_candidate_smoke(config, config_path)
    prepared_path = project_path(study["prepared"])
    lock_path = project_path(study["method_lock"])
    manifest_path = project_path(study["prepare_manifest"])
    if any(path.exists() for path in (prepared_path, lock_path, manifest_path)):
        raise FileExistsError("M7 prepared/method-lock artifact exists")
    events, item_info, stats = candidate_cohort(config, int(study["primary_events"]))
    audit = verify_audit(config)
    prepared = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "partition": "fresh_primary_test",
            "graph_cutoff": int(audit["temporal_split"]["validation_cutoff"]),
            "candidate_sampling": "one gold plus nine hash-uniform negatives from exact/session one-step reachable intersection",
            "gold_reachability_required": False,
            "recommendation_outcomes_used_for_ranking": False,
        },
        "cohort_stats": stats,
        "development_events": [],
        "primary_events": events,
        "item_info": item_info,
    }
    method_lock = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": {
            key: value
            for key, value in study.items()
            if key
            not in {
                "candidate_smoke",
                "prepared",
                "method_lock",
                "prepare_manifest",
                "graph_smoke",
                "graph_scores",
                "score_lock",
                "metrics",
                "evaluation_manifest",
            }
        },
        "graph": dict(config["graph"]),
        "baselines": dict(config["baselines"]),
        "self_host": {
            key: value
            for key, value in config["self_host"].items()
            if key not in {"attempts", "calls", "smoke_manifest", "manifest"}
        },
        "primary_labels_used_for_lock": False,
        "manual_tuning_performed": False,
    }
    prepared_path.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    lock_path.write_text(
        json.dumps(method_lock, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "complete",
        "config": {
            "path": str(config_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(config_path),
        },
        "candidate_smoke": {
            "path": study["candidate_smoke"],
            "sha256": sha256_file(project_path(study["candidate_smoke"])),
        },
        "prepared": {
            "path": study["prepared"],
            "sha256": sha256_file(prepared_path),
            "events": len(events),
        },
        "method_lock": {"path": study["method_lock"], "sha256": sha256_file(lock_path)},
        "recommendation_outcomes_evaluated": False,
        "manual_tuning_performed": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fresh MovieLens graph-hard M7 cohort")
    parser.add_argument(
        "--config", default="configs/temporal_movielens32m/m7_graph_hard_end2end.yaml"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--data-smoke", action="store_true")
    modes.add_argument("--candidate-smoke", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke-users", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.data_smoke:
        result = smoke_adapter(config, args.smoke_users)
    elif args.candidate_smoke:
        result = run_candidate_smoke(config, config_path)
    else:
        result = run_prepare(config, config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
