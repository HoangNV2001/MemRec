"""Smoke-first exact/session graph scoring for the frozen MovieLens cohort."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from src.temporal_books.common import jsonl_write, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.current_support import event_key
from src.temporal_common.graph import (
    TransitionEvent,
    build_transition_graph_from_users,
    one_step_scores,
    ppr_monte_carlo_scores,
)
from src.temporal_movielens.data import iter_user_ratings
from src.temporal_movielens.prepare import verify_audit


SCHEMA_VERSION = 1
VIEWS = ("exact", "session_300s")


def load_locked_inputs(config: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    study = config["study"]
    manifest_path = project_path(study["prepare_manifest"])
    prepared_path = project_path(study["prepared"])
    lock_path = project_path(study["method_lock"])
    for path in (manifest_path, prepared_path, lock_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    method_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if manifest["prepared"]["sha256"] != sha256_file(prepared_path):
        raise RuntimeError("prepared cohort hash mismatch")
    if manifest["method_lock"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("method lock hash mismatch")
    if manifest["recommendation_outcomes_evaluated"] is not False:
        raise RuntimeError("graph scoring blocked after outcomes were evaluated")
    return prepared, method_lock, manifest


def transition_histories(path: Path, *, limit_users: int | None = None) -> Iterable[list[TransitionEvent]]:
    for _, events in iter_user_ratings(path, limit_users=limit_users):
        yield [(timestamp, movie_id) for timestamp, movie_id, _ in events]


def build_view(
    ratings_path: Path,
    *,
    cutoff: int,
    session_gap_seconds: int,
    limit_users: int | None = None,
) -> tuple[Dict[str, list[tuple[str, ...]]], Dict[str, int], float]:
    started = time.monotonic()
    adjacency, stats = build_transition_graph_from_users(
        transition_histories(ratings_path, limit_users=limit_users),
        cutoff=cutoff,
        session_gap_seconds=session_gap_seconds,
    )
    return adjacency, stats, time.monotonic() - started


def event_seeds(event: Mapping[str, Any], limit: int) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()
    for row in reversed(event["history"]):
        movie_id = str(row["item_id"])
        if movie_id not in seen:
            seen.add(movie_id)
            seeds.append(movie_id)
        if len(seeds) >= limit:
            break
    return seeds


def score_view(
    events: Sequence[Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[tuple[str, ...]]],
    graph: Mapping[str, Any],
    *,
    walks: int,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for event in events:
        key = event_key(event)
        candidates = [str(value) for value in event["candidate_item_ids"]]
        seeds = event_seeds(event, int(graph["seed_items"]))
        rows.append(
            {
                "event_key": key,
                "candidate_item_ids": candidates,
                "seed_item_ids": seeds,
                "one_step_scores": one_step_scores(seeds, candidates, adjacency),
                "ppr_scores": ppr_monte_carlo_scores(
                    seeds,
                    candidates,
                    adjacency,
                    walks=walks,
                    restart_probability=float(graph["restart_probability"]),
                    max_steps=int(graph["max_walk_steps"]),
                    seed=stable_order(f"ml32m-ppr\0{key}"),
                ),
                "gold_label_used_for_scoring": False,
            }
        )
    return rows


def selected_smoke_events(prepared: Mapping[str, Any], count: int) -> list[Dict[str, Any]]:
    events = sorted(
        prepared["primary_events"], key=lambda row: stable_order(f"ml32m-graph-smoke\0{event_key(row)}")
    )[:count]
    if len(events) != count:
        raise RuntimeError("insufficient graph smoke events")
    return events


def score_both_views(
    config: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    walks: int,
    limit_users: int | None = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    dataset, graph = config["dataset"], config["graph"]
    audit = verify_audit(config)
    ratings_path = project_path(dataset["ratings_file"])
    cutoff = int(audit["temporal_split"]["validation_cutoff"])
    by_key = {
        event_key(event): {
            "schema_version": SCHEMA_VERSION,
            "event_key": event_key(event),
            "candidate_item_ids": [str(value) for value in event["candidate_item_ids"]],
            "seed_item_ids": event_seeds(event, int(graph["seed_items"])),
            "views": {},
            "gold_label_used_for_scoring": False,
        }
        for event in events
    }
    view_stats: Dict[str, Any] = {}
    gaps = {
        "exact": int(graph["exact_session_gap_seconds"]),
        "session_300s": int(graph["robustness_session_gap_seconds"]),
    }
    for view in VIEWS:
        adjacency, stats, build_seconds = build_view(
            ratings_path,
            cutoff=cutoff,
            session_gap_seconds=gaps[view],
            limit_users=limit_users,
        )
        scored = score_view(events, adjacency, graph, walks=walks)
        repeated = score_view(events[:1], adjacency, graph, walks=walks)[0]
        if scored[0] != repeated:
            raise RuntimeError(f"{view} graph scoring is not deterministic")
        for row in scored:
            by_key[row["event_key"]]["views"][view] = {
                "one_step_scores": row["one_step_scores"],
                "ppr_scores": row["ppr_scores"],
            }
        view_stats[view] = {
            "session_gap_seconds": gaps[view],
            "graph_stats": stats,
            "build_seconds": build_seconds,
            "one_step_nonzero_slots": sum(
                score > 0 for row in scored for score in row["one_step_scores"].values()
            ),
            "ppr_nonzero_slots": sum(score > 0 for row in scored for score in row["ppr_scores"].values()),
        }
        del adjacency
        gc.collect()
    rows = [by_key[event_key(event)] for event in events]
    if any(set(row["views"]) != set(VIEWS) for row in rows):
        raise RuntimeError("cross-view event mismatch")
    return rows, view_stats


def run_small_smoke(config: Mapping[str, Any], users: int) -> Dict[str, Any]:
    if not 20 <= users <= 100:
        raise ValueError("small graph smoke requires 20-100 users")
    prepared, _, _ = load_locked_inputs(config)
    events = selected_smoke_events(prepared, 20)
    rows, stats = score_both_views(config, events, walks=100, limit_users=users)
    return {
        "decision": "pass",
        "users": users,
        "events": len(rows),
        "views": stats,
        "artifact_written": False,
    }


def run_graph_smoke(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    prepared, method_lock, manifest = load_locked_inputs(config)
    graph = config["graph"]
    events = selected_smoke_events(prepared, int(graph["graph_smoke_events"]))
    rows, stats = score_both_views(config, events, walks=int(graph["smoke_walks"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": manifest["prepared"]["sha256"],
        "method_lock_sha256": manifest["method_lock"]["sha256"],
        "graph_contract": method_lock["graph"],
        "event_keys": [row["event_key"] for row in rows],
        "candidate_slots": sum(len(row["candidate_item_ids"]) for row in rows),
        "views": stats,
        "deterministic_replay": True,
        "gold_labels_used_for_scoring": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    path = project_path(config["study"]["graph_smoke"])
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def run_full_scores(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    prepared, _, manifest = load_locked_inputs(config)
    smoke_path = project_path(config["study"]["graph_smoke"])
    scores_path = project_path(config["study"]["graph_scores"])
    if not smoke_path.exists():
        raise RuntimeError("full graph scoring blocked: smoke manifest missing")
    if scores_path.exists():
        raise FileExistsError(scores_path)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    expected = {
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": manifest["prepared"]["sha256"],
        "method_lock_sha256": manifest["method_lock"]["sha256"],
        "gold_labels_used_for_scoring": False,
    }
    for key, value in expected.items():
        if smoke.get(key) != value:
            raise RuntimeError(f"graph smoke mismatch: {key}")
    rows, stats = score_both_views(
        config,
        prepared["primary_events"],
        walks=int(config["graph"]["monte_carlo_walks"]),
    )
    records, digest = jsonl_write(scores_path, rows)
    return {
        "decision": "complete",
        "records": records,
        "sha256": digest,
        "views": stats,
        "gold_labels_used_for_scoring": False,
        "llm_requests": 0,
        "gpu_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score frozen MovieLens temporal graph views")
    parser.add_argument("--config", default="configs/temporal_movielens32m/m1_frozen_transfer.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--small-smoke", action="store_true")
    modes.add_argument("--graph-smoke", action="store_true")
    modes.add_argument("--score", action="store_true")
    parser.add_argument("--smoke-users", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.small_smoke:
        result = run_small_smoke(config, args.smoke_users)
    elif args.graph_smoke:
        result = run_graph_smoke(config, config_path)
    else:
        result = run_full_scores(config, config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
