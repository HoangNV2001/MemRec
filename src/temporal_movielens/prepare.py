"""Lock MovieLens development/primary cohorts without graph or LLM scoring."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.current_support import stable_sample_candidates
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


SCHEMA_VERSION = 1


def verify_audit(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = config["dataset"]
    path = project_path(dataset["audit"])
    digest = sha256_file(path)
    if digest != str(dataset["audit_sha256"]):
        raise RuntimeError(f"MovieLens audit hash mismatch: {digest}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("decision") != "pass" or audit.get("recommendation_outcomes_read") is not False:
        raise RuntimeError("MovieLens audit did not pass the outcome-free gate")
    return audit


def choose_cohorts(
    ratings_path: Path,
    candidate_pool: set[str],
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], dict[str, int]]:
    study = config["study"]
    train_cutoff = int(audit["temporal_split"]["train_cutoff"])
    validation_cutoff = int(audit["temporal_split"]["validation_cutoff"])
    development: list[Dict[str, Any]] = []
    primary: list[Dict[str, Any]] = []
    for user_id, events in iter_user_ratings(ratings_path):
        development_targets = eligible_singleton_targets(
            user_id,
            events,
            start=train_cutoff,
            end=validation_cutoff,
            candidate_pool=candidate_pool,
            session_gap_seconds=int(study["session_gap_seconds"]),
            positive_rating_min=float(study["positive_rating_min"]),
            minimum_positive_history=int(study["minimum_positive_history"]),
        )
        if development_targets:
            development.append(
                select_one_target_per_user(development_targets, salt=str(study["target_salt"]) + "-development")
            )
        primary_targets = eligible_singleton_targets(
            user_id,
            events,
            start=validation_cutoff,
            end=None,
            candidate_pool=candidate_pool,
            session_gap_seconds=int(study["session_gap_seconds"]),
            positive_rating_min=float(study["positive_rating_min"]),
            minimum_positive_history=int(study["minimum_positive_history"]),
        )
        if primary_targets:
            primary.append(select_one_target_per_user(primary_targets, salt=str(study["target_salt"]) + "-primary"))

    eligible_development_users = len(development)
    primary.sort(
        key=lambda row: stable_order(
            f"{study['primary_salt']}\0{row['user_id']}\0{row['timestamp']}\0{row['gold_item_id']}"
        )
    )
    selected_primary = primary[: int(study["primary_events"])]
    primary_users = {str(row["user_id"]) for row in selected_primary}
    development = [row for row in development if str(row["user_id"]) not in primary_users]
    development.sort(
        key=lambda row: stable_order(
            f"{study['development_salt']}\0{row['user_id']}\0{row['timestamp']}\0{row['gold_item_id']}"
        )
    )
    selected_development = development[: int(study["development_events"])]
    if len(selected_primary) != int(study["primary_events"]):
        raise RuntimeError(f"insufficient primary targets: {len(selected_primary)}")
    if len(selected_development) != int(study["development_events"]):
        raise RuntimeError(f"insufficient development targets: {len(selected_development)}")
    return selected_development, selected_primary, {
        "eligible_development_users": eligible_development_users,
        "eligible_primary_users": len(primary),
    }


def enrich_events(
    ratings_path: Path,
    movies: Mapping[str, Mapping[str, str]],
    candidate_pool: Sequence[str],
    targets: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    validation_cutoff: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    study = config["study"]
    by_user = {str(row["user_id"]): dict(row) for row in targets}
    enriched: list[Dict[str, Any]] = []
    needed_items: set[str] = set()
    for user_id, events in iter_user_ratings(ratings_path):
        target = by_user.get(user_id)
        if target is None:
            continue
        history = strict_history(events, int(target["timestamp"]))
        known = {movie_id for _, movie_id, _ in history}
        gold = str(target["gold_item_id"])
        candidates = stable_sample_candidates(
            candidate_pool,
            known,
            gold,
            n_candidates=int(study["candidates_per_event"]),
            seed_key=f"{study['candidate_salt']}\0{user_id}\0{target['timestamp']}\0{gold}",
        )
        recent = history[-int(study["history_events"]):]
        history_rows = []
        for timestamp, movie_id, rating in recent:
            metadata = movies[movie_id]
            history_rows.append(
                {
                    "timestamp": timestamp,
                    "item_id": movie_id,
                    "title": metadata["title"],
                    "genres": metadata["genres"],
                    "rating": rating,
                }
            )
        needed_items.update(candidates)
        enriched.append(
            {
                **target,
                "candidate_item_ids": candidates,
                "history": history_rows,
                "historical_burstiness_le_60s": historical_burstiness(events, validation_cutoff),
                "candidate_history_overlap": sorted(set(candidates) & known, key=int),
            }
        )
    if len(enriched) != len(targets):
        raise RuntimeError(f"failed to enrich all targets: {len(enriched)} of {len(targets)}")
    enriched.sort(key=lambda row: stable_order(f"prepared\0{row['user_id']}\0{row['timestamp']}"))
    item_info = {
        movie_id: {
            "title": movies[movie_id]["title"],
            "genres": movies[movie_id]["genres"],
            "base_memory": movie_memory(movies[movie_id]),
        }
        for movie_id in sorted(needed_items, key=int)
    }
    return enriched, item_info


def validate_prepared(prepared: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    study = config["study"]
    development = prepared["development_events"]
    primary = prepared["primary_events"]
    if len(development) != int(study["development_events"]) or len(primary) != int(study["primary_events"]):
        raise RuntimeError("prepared cohort size mismatch")
    development_users = {str(row["user_id"]) for row in development}
    primary_users = {str(row["user_id"]) for row in primary}
    if len(development_users) != len(development) or len(primary_users) != len(primary):
        raise RuntimeError("one-event-per-user contract violated")
    if development_users & primary_users:
        raise RuntimeError("development and primary users overlap")
    for event in [*development, *primary]:
        candidates = event["candidate_item_ids"]
        if len(candidates) != 10 or len(set(candidates)) != 10:
            raise RuntimeError("candidate uniqueness contract violated")
        if str(event["gold_item_id"]) not in candidates:
            raise RuntimeError("gold is absent from candidates")
        if event["candidate_history_overlap"]:
            raise RuntimeError("candidate/history collision")
        if len(event["history"]) > int(study["history_events"]):
            raise RuntimeError("history cap violated")
        if any(int(row["timestamp"]) >= int(event["timestamp"]) for row in event["history"]):
            raise RuntimeError("strict-past history violated")
        if str(event["gold_item_id"]) in {str(row["item_id"]) for row in event["history"]}:
            raise RuntimeError("gold-in-history leakage")


def run_prepare(config: Mapping[str, Any], config_path: Path, *, force: bool) -> Dict[str, Any]:
    study, dataset = config["study"], config["dataset"]
    output = project_path(study["prepared"])
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; use --force only for an intentional rerun")
    audit = verify_audit(config)
    ratings_path = project_path(dataset["ratings_file"])
    movies_path = project_path(dataset["movies_file"])
    for name, path in (("ratings", ratings_path), ("movies", movies_path)):
        expected_sha = str(audit["raw_integrity"]["sha256"][name])
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"{name} changed after M0 audit")
    pool = positive_candidate_pool(
        ratings_path,
        train_cutoff=int(audit["temporal_split"]["train_cutoff"]),
        positive_rating_min=float(study["positive_rating_min"]),
    )
    development_targets, primary_targets, eligibility = choose_cohorts(
        ratings_path, set(pool), config, audit
    )
    movies = load_movies(movies_path)
    all_targets = [*development_targets, *primary_targets]
    all_events, item_info = enrich_events(
        ratings_path,
        movies,
        pool,
        all_targets,
        config,
        int(audit["temporal_split"]["validation_cutoff"]),
    )
    development_keys = {(str(row["user_id"]), int(row["timestamp"])) for row in development_targets}
    development = [row for row in all_events if (str(row["user_id"]), int(row["timestamp"])) in development_keys]
    primary = [row for row in all_events if (str(row["user_id"]), int(row["timestamp"])) not in development_keys]
    prepared = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "audit": {"path": dataset["audit"], "sha256": sha256_file(project_path(dataset["audit"]))},
            "ratings": {"path": dataset["ratings_file"], "sha256": audit["raw_integrity"]["sha256"]["ratings"]},
            "movies": {"path": dataset["movies_file"], "sha256": audit["raw_integrity"]["sha256"]["movies"]},
        },
        "protocol": {
            "train_cutoff": int(audit["temporal_split"]["train_cutoff"]),
            "graph_cutoff": int(audit["temporal_split"]["validation_cutoff"]),
            "singleton_session_gap_seconds": int(study["session_gap_seconds"]),
            "candidate_pool": "positive movies observed before train cutoff",
            "candidate_sampling": "one positive plus nine deterministic uniform strict-past-unseen negatives",
            "recommendation_outcomes_used_for_selection": False,
        },
        "candidate_pool_items": len(pool),
        "eligibility": eligibility,
        "development_events": development,
        "primary_events": primary,
        "item_info": item_info,
    }
    validate_prepared(prepared, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    method_lock = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph": dict(config["graph"]),
        "self_host": {key: value for key, value in config["self_host"].items() if key not in {"attempts", "calls", "smoke_manifest", "manifest"}},
        "primary_labels_used_for_lock": False,
    }
    lock_path = project_path(study["method_lock"])
    lock_path.write_text(json.dumps(method_lock, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "audit": {"path": dataset["audit"], "sha256": sha256_file(project_path(dataset["audit"]))},
        "prepared": {"path": study["prepared"], "sha256": sha256_file(output)},
        "method_lock": {"path": study["method_lock"], "sha256": sha256_file(lock_path)},
        "llm_requests": 0,
        "gpu_used": False,
        "recommendation_outcomes_evaluated": False,
    }
    manifest_path = project_path(study["prepare_manifest"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return prepared


def smoke_adapter(config: Mapping[str, Any], users: int) -> Dict[str, Any]:
    if not 20 <= users <= 100:
        raise ValueError("adapter smoke requires 20-100 users")
    audit = verify_audit(config)
    ratings_path = project_path(config["dataset"]["ratings_file"])
    parsed_users = parsed_events = 0
    movie_ids: set[str] = set()
    for _, events in iter_user_ratings(ratings_path, limit_users=users):
        parsed_users += 1
        parsed_events += len(events)
        movie_ids.update(movie_id for _, movie_id, _ in events)
    movies = load_movies(project_path(config["dataset"]["movies_file"]))
    if parsed_users != users or not movie_ids <= set(movies):
        raise RuntimeError("adapter smoke failed metadata coverage")
    return {
        "decision": "pass",
        "users": parsed_users,
        "ratings": parsed_events,
        "rated_movies": len(movie_ids),
        "train_cutoff": audit["temporal_split"]["train_cutoff"],
        "validation_cutoff": audit["temporal_split"]["validation_cutoff"],
        "artifact_written": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen MovieLens temporal cohorts")
    parser.add_argument("--config", default="configs/temporal_movielens32m/m1_frozen_transfer.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-users", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.smoke:
        print(json.dumps(smoke_adapter(config, args.smoke_users), indent=2, sort_keys=True))
        return
    prepared = run_prepare(config, config_path, force=args.force)
    print(json.dumps({
        "prepared": config["study"]["prepared"],
        "development_events": len(prepared["development_events"]),
        "primary_events": len(prepared["primary_events"]),
        "candidate_pool_items": prepared["candidate_pool_items"],
        "llm_requests": 0,
        "gpu_used": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
