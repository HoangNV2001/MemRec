"""Graph-hard development headroom gate for MovieLens multi-hop propagation."""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Mapping, Sequence

from src.temporal_books.common import PROJECT_ROOT, jsonl_write, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.current_support import event_key, read_jsonl
from src.temporal_common.metrics import (
    event_hit_at_5,
    event_ndcg_at_5,
    paired_bootstrap_ci,
    rank_by_scores,
)
from src.temporal_movielens.data import (
    eligible_singleton_targets,
    historical_burstiness,
    iter_user_ratings,
    positive_candidate_pool,
    select_one_target_per_user,
    strict_history,
)
from src.temporal_movielens.graph import build_view, event_seeds, score_view
from src.temporal_movielens.prepare import smoke_adapter, verify_audit


SCHEMA_VERSION = 1
VIEWS = ("exact", "session_300s")


def load_sealed_exclusions(config: Mapping[str, Any]) -> tuple[set[str], str]:
    study = config["study"]
    path = project_path(study["sealed_prepared"])
    digest = sha256_file(path)
    if digest != str(study["sealed_prepared_sha256"]):
        raise RuntimeError("sealed M1 cohort hash mismatch")
    prepared = json.loads(path.read_text(encoding="utf-8"))
    users = {
        str(event["user_id"])
        for event in [*prepared["development_events"], *prepared["primary_events"]]
    }
    if len(users) != len(prepared["development_events"]) + len(prepared["primary_events"]):
        raise RuntimeError("sealed M1 users are not disjoint")
    return users, digest


def one_step_all_scores(
    seeds: Sequence[str], adjacency: Mapping[str, Sequence[tuple[str, ...]]]
) -> Dict[str, float]:
    scores: DefaultDict[str, float] = defaultdict(float)
    if not seeds:
        return {}
    seed_weight = 1.0 / len(seeds)
    for source in seeds:
        groups = adjacency.get(source, ())
        if not groups:
            continue
        group_weight = seed_weight / len(groups)
        for destinations in groups:
            destination_weight = group_weight / len(destinations)
            for item_id in destinations:
                scores[item_id] += destination_weight
    return dict(scores)


def graph_hard_candidate_event(
    event: Mapping[str, Any],
    exact_scores: Mapping[str, float],
    session_scores: Mapping[str, float],
    candidate_pool: set[str],
    study: Mapping[str, Any],
) -> Dict[str, Any] | None:
    gold = str(event["gold_item_id"])
    known = {str(value) for value in event["_known_item_ids"]}
    reachable = (
        {item_id for item_id, score in exact_scores.items() if score > 0}
        & {item_id for item_id, score in session_scores.items() if score > 0}
        & candidate_pool
    )
    eligible = reachable - known - {gold}
    needed = int(study["candidates_per_event"]) - 1
    if len(eligible) < needed:
        return None
    key = event_key(event)
    negatives = sorted(
        eligible,
        key=lambda item_id: stable_order(f"{study['candidate_salt']}\0{key}\0{item_id}"),
    )[:needed]
    candidates = sorted(
        [gold, *negatives],
        key=lambda item_id: stable_order(f"{study['candidate_order_salt']}\0{key}\0{item_id}"),
    )
    result = {key: value for key, value in event.items() if not key.startswith("_")}
    result.update(
        {
            "candidate_item_ids": candidates,
            "negative_item_ids": negatives,
            "candidate_history_overlap": sorted(set(candidates) & known, key=int),
            "reachable_intersection_items": len(eligible),
        }
    )
    return result


def development_scan(
    config: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], set[str], Dict[str, int]]:
    dataset, study = config["dataset"], config["study"]
    audit = verify_audit(config)
    ratings_path = project_path(dataset["ratings_file"])
    excluded, _ = load_sealed_exclusions(config)
    candidate_pool_values = positive_candidate_pool(
        ratings_path,
        train_cutoff=int(audit["temporal_split"]["train_cutoff"]),
        positive_rating_min=float(study["positive_rating_min"]),
    )
    candidate_pool = set(candidate_pool_values)
    targets: list[Dict[str, Any]] = []
    for user_id, events in iter_user_ratings(ratings_path):
        if user_id in excluded:
            continue
        eligible = eligible_singleton_targets(
            user_id,
            events,
            start=int(audit["temporal_split"]["train_cutoff"]),
            end=int(audit["temporal_split"]["validation_cutoff"]),
            candidate_pool=candidate_pool,
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
    scan_limit = int(study["development_scan_users"])
    selected = targets[:scan_limit]
    if len(selected) != scan_limit:
        raise RuntimeError(f"insufficient development scan users: {len(selected)}")
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
                    events, int(audit["temporal_split"]["train_cutoff"])
                ),
                "_known_item_ids": sorted({item_id for _, item_id, _ in history}, key=int),
            }
        )
    by_key = {event_key(event): event for event in enriched}
    ordered = [by_key[event_key(target)] for target in selected]
    if len(ordered) != scan_limit:
        raise RuntimeError("failed to enrich fixed development scan")
    return ordered, candidate_pool, {
        "sealed_excluded_users": len(excluded),
        "eligible_development_users_after_exclusion": len(targets),
        "fixed_scan_users": len(ordered),
        "candidate_pool_items": len(candidate_pool),
    }


def candidate_cohort(
    config: Mapping[str, Any], count: int
) -> tuple[list[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    audit = verify_audit(config)
    dataset, study, graph = config["dataset"], config["study"], config["graph"]
    scan, pool, scan_stats = development_scan(config)
    cutoff = int(audit["temporal_split"]["train_cutoff"])
    ratings_path = project_path(dataset["ratings_file"])
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
        raise RuntimeError(f"only {len(selected)} graph-hard events available in fixed scan")
    for event in selected:
        if event["candidate_history_overlap"]:
            raise RuntimeError("graph-hard candidate/history collision")
        if len(event["candidate_item_ids"]) != int(study["candidates_per_event"]):
            raise RuntimeError("graph-hard candidate count mismatch")
        if len(set(event["candidate_item_ids"])) != len(event["candidate_item_ids"]):
            raise RuntimeError("graph-hard candidates are not unique")
    stats = {
        **scan_stats,
        "graph_hard_feasible_users": len(feasible),
        "selected_events": len(selected),
        "rejected_for_fewer_than_nine_reachable_negatives": len(scan) - len(feasible),
    }
    graph_stats = {
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
    }
    return selected, {"exact": exact, "session_300s": session}, {"scan": stats, "views": graph_stats}


def score_with_adjacencies(
    events: Sequence[Mapping[str, Any]],
    adjacencies: Mapping[str, Mapping[str, Sequence[tuple[str, ...]]]],
    graph: Mapping[str, Any],
    *,
    walks: int,
) -> list[Dict[str, Any]]:
    blind_events = [
        {
            "user_id": event["user_id"],
            "timestamp": event["timestamp"],
            "candidate_item_ids": event["candidate_item_ids"],
            "history": event["history"],
        }
        for event in events
    ]
    by_key = {
        event_key(event): {
            "schema_version": SCHEMA_VERSION,
            "event_key": event_key(event),
            "candidate_item_ids": list(event["candidate_item_ids"]),
            "seed_item_ids": event_seeds(event, int(graph["seed_items"])),
            "views": {},
            "gold_label_used_for_scoring": False,
        }
        for event in blind_events
    }
    negatives = {event_key(event): set(event["negative_item_ids"]) for event in events}
    for view in VIEWS:
        rows = score_view(blind_events, adjacencies[view], graph, walks=walks)
        repeated = score_view(blind_events[:1], adjacencies[view], graph, walks=walks)[0]
        if rows[0] != repeated:
            raise RuntimeError(f"{view} graph-hard scoring is not deterministic")
        for row in rows:
            key = row["event_key"]
            if any(float(row["one_step_scores"][item_id]) <= 0 for item_id in negatives[key]):
                raise RuntimeError(f"{view} graph-hard negative lost one-step evidence: {key}")
            by_key[key]["views"][view] = {
                "one_step_scores": row["one_step_scores"],
                "ppr_scores": row["ppr_scores"],
            }
    return [by_key[event_key(event)] for event in events]


def verify_smoke(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    study = config["study"]
    path = project_path(study["graph_smoke"])
    if not path.exists():
        raise RuntimeError("graph-hard phase blocked: run --graph-smoke first")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "sealed_prepared_sha256": str(study["sealed_prepared_sha256"]),
        "recommendation_outcomes_evaluated": False,
    }
    for key, value in expected.items():
        if smoke.get(key) != value:
            raise RuntimeError(f"graph-hard smoke mismatch: {key}")
    return smoke


def run_graph_smoke(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    study, graph = config["study"], config["graph"]
    output = project_path(study["graph_smoke"])
    if output.exists():
        raise FileExistsError(output)
    count = int(graph["graph_smoke_events"])
    events, adjacencies, stats = candidate_cohort(config, count)
    rows = score_with_adjacencies(events, adjacencies, graph, walks=int(graph["smoke_walks"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "sealed_prepared_sha256": str(study["sealed_prepared_sha256"]),
        "events": len(events),
        "candidate_slots": sum(len(row["candidate_item_ids"]) for row in rows),
        "negative_slots_with_one_step_evidence_in_both_views": len(events) * 9,
        "deterministic_replay": True,
        "recommendation_outcomes_evaluated": False,
        "gold_labels_used_for_scoring": False,
        "llm_requests": 0,
        "gpu_used": False,
        "cohort_stats": stats,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    del adjacencies
    gc.collect()
    return payload


def run_prepare(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    study = config["study"]
    verify_smoke(config, config_path)
    output = project_path(study["prepared"])
    manifest_path = project_path(study["prepare_manifest"])
    if output.exists() or manifest_path.exists():
        raise FileExistsError("graph-hard prepared output exists")
    events, adjacencies, stats = candidate_cohort(config, int(study["development_events"]))
    audit = verify_audit(config)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "sealed_prepared_sha256": str(study["sealed_prepared_sha256"]),
        "protocol": {
            "partition": "development",
            "graph_cutoff": int(audit["temporal_split"]["train_cutoff"]),
            "candidate_negative_source": "uniform deterministic hash from exact/session one-step reachable intersection",
            "gold_reachability_required": False,
            "recommendation_outcomes_used_for_graph_ranking": False,
        },
        "cohort_stats": stats,
        "development_events": events,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "complete",
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "graph_smoke": {"path": study["graph_smoke"], "sha256": sha256_file(project_path(study["graph_smoke"]))},
        "prepared": {"path": study["prepared"], "sha256": sha256_file(output), "events": len(events)},
        "recommendation_outcomes_evaluated": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    del adjacencies
    gc.collect()
    return manifest


def load_prepared(config: Mapping[str, Any], config_path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    study = config["study"]
    manifest_path = project_path(study["prepare_manifest"])
    prepared_path = project_path(study["prepared"])
    for path in (manifest_path, prepared_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    expected = {
        "decision": "complete",
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"graph-hard prepare manifest mismatch: {key}")
    if manifest["prepared"]["sha256"] != sha256_file(prepared_path):
        raise RuntimeError("graph-hard prepared hash mismatch")
    if len(prepared["development_events"]) != int(study["development_events"]):
        raise RuntimeError("graph-hard development count mismatch")
    return prepared, manifest


def run_score(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    study, dataset, graph = config["study"], config["dataset"], config["graph"]
    verify_smoke(config, config_path)
    prepared, prepare_manifest = load_prepared(config, config_path)
    scores_path = project_path(study["graph_scores"])
    manifest_path = project_path(study["score_manifest"])
    if scores_path.exists() or manifest_path.exists():
        raise FileExistsError("graph-hard score output exists")
    events = prepared["development_events"]
    cutoff = int(prepared["protocol"]["graph_cutoff"])
    ratings_path = project_path(dataset["ratings_file"])
    by_key = {
        event_key(event): {
            "schema_version": SCHEMA_VERSION,
            "event_key": event_key(event),
            "candidate_item_ids": list(event["candidate_item_ids"]),
            "seed_item_ids": event_seeds(event, int(graph["seed_items"])),
            "views": {},
            "gold_label_used_for_scoring": False,
        }
        for event in events
    }
    blind = [
        {
            "user_id": event["user_id"],
            "timestamp": event["timestamp"],
            "candidate_item_ids": event["candidate_item_ids"],
            "history": event["history"],
        }
        for event in events
    ]
    negatives = {event_key(event): set(event["negative_item_ids"]) for event in events}
    view_stats: Dict[str, Any] = {}
    gaps = {
        "exact": int(graph["exact_session_gap_seconds"]),
        "session_300s": int(graph["robustness_session_gap_seconds"]),
    }
    for view in VIEWS:
        adjacency, stats, seconds = build_view(
            ratings_path, cutoff=cutoff, session_gap_seconds=gaps[view]
        )
        rows = score_view(blind, adjacency, graph, walks=int(graph["monte_carlo_walks"]))
        repeated = score_view(blind[:1], adjacency, graph, walks=int(graph["monte_carlo_walks"]))[0]
        if rows[0] != repeated:
            raise RuntimeError(f"{view} full graph-hard scoring is not deterministic")
        for row in rows:
            key = row["event_key"]
            if any(float(row["one_step_scores"][item_id]) <= 0 for item_id in negatives[key]):
                raise RuntimeError(f"{view} graph-hard negative lost evidence: {key}")
            by_key[key]["views"][view] = {
                "one_step_scores": row["one_step_scores"],
                "ppr_scores": row["ppr_scores"],
            }
        view_stats[view] = {
            "session_gap_seconds": gaps[view],
            "graph_stats": stats,
            "build_seconds": seconds,
            "one_step_nonzero_slots": sum(
                value > 0 for row in rows for value in row["one_step_scores"].values()
            ),
            "ppr_nonzero_slots": sum(value > 0 for row in rows for value in row["ppr_scores"].values()),
        }
        del adjacency
        gc.collect()
    output_rows = [by_key[event_key(event)] for event in events]
    records, digest = jsonl_write(scores_path, output_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "locked",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": prepare_manifest["prepared"]["sha256"],
        "graph_scores": {"path": study["graph_scores"], "records": records, "sha256": digest},
        "views": view_stats,
        "recommendation_labels_used_for_scoring": False,
        "recommendation_outcomes_evaluated": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def headroom_metrics(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    by_key = {event_key(event): event for event in events}
    if len(by_key) != len(events) or {str(row["event_key"]) for row in rows} != set(by_key):
        raise RuntimeError("graph-hard outcome/score mismatch")
    graph = config["graph"]
    arms: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    coverage: Dict[str, Any] = {}
    pass_by_view: Dict[str, bool] = {}
    for view in VIEWS:
        one_values: list[float] = []
        ppr_values: list[float] = []
        one_hits: list[float] = []
        ppr_hits: list[float] = []
        gold_one = gold_ppr = negative_one = negative_ppr = negative_slots = 0
        for row in rows:
            event = by_key[str(row["event_key"])]
            gold = str(event["gold_item_id"])
            candidates = [str(value) for value in row["candidate_item_ids"]]
            scores = row["views"][view]
            one_ranking = rank_by_scores(candidates, scores["one_step_scores"])
            ppr_ranking = rank_by_scores(candidates, scores["ppr_scores"])
            one_values.append(event_ndcg_at_5(one_ranking, gold))
            ppr_values.append(event_ndcg_at_5(ppr_ranking, gold))
            one_hits.append(event_hit_at_5(one_ranking, gold))
            ppr_hits.append(event_hit_at_5(ppr_ranking, gold))
            gold_one += int(float(scores["one_step_scores"][gold]) > 0)
            gold_ppr += int(float(scores["ppr_scores"][gold]) > 0)
            for item_id in event["negative_item_ids"]:
                negative_slots += 1
                negative_one += int(float(scores["one_step_scores"][item_id]) > 0)
                negative_ppr += int(float(scores["ppr_scores"][item_id]) > 0)
        delta = [ppr - one for one, ppr in zip(one_values, ppr_values)]
        lower, upper = paired_bootstrap_ci(
            delta,
            resamples=int(graph["bootstrap_resamples"]),
            seed=int(graph["bootstrap_seed"]),
        )
        mean_delta = sum(delta) / len(delta)
        passed = mean_delta >= float(graph["headroom_min_delta"]) and lower > 0
        pass_by_view[view] = passed
        arms[view] = {
            "one_step_graph_only": {
                "n_events": len(events),
                "ndcg_at_5": sum(one_values) / len(one_values),
                "hit_at_5": sum(one_hits) / len(one_hits),
                "per_event_ndcg_at_5": one_values,
            },
            "ppr_graph_only": {
                "n_events": len(events),
                "ndcg_at_5": sum(ppr_values) / len(ppr_values),
                "hit_at_5": sum(ppr_hits) / len(ppr_hits),
                "per_event_ndcg_at_5": ppr_values,
            },
        }
        comparisons[view] = {
            "delta_ppr_minus_one_step_ndcg_at_5": mean_delta,
            "paired_bootstrap_95_ci": [lower, upper],
            "improved_events": sum(value > 0 for value in delta),
            "worsened_events": sum(value < 0 for value in delta),
            "unchanged_events": sum(value == 0 for value in delta),
        }
        coverage[view] = {
            "gold_one_step": gold_one,
            "gold_ppr": gold_ppr,
            "negative_one_step": negative_one,
            "negative_ppr": negative_ppr,
            "negative_slots": negative_slots,
            "gold_one_step_rate": gold_one / len(events),
            "gold_ppr_rate": gold_ppr / len(events),
            "negative_one_step_rate": negative_one / negative_slots,
            "negative_ppr_rate": negative_ppr / negative_slots,
        }
    conclusion = {
        (True, True): "promote_both_views",
        (True, False): "promote_exact_only",
        (False, True): "promote_session_adaptation_only",
        (False, False): "stop_before_llm",
    }[(pass_by_view["exact"], pass_by_view["session_300s"])]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": len(events),
        "arms": arms,
        "comparisons": comparisons,
        "coverage": coverage,
        "gate": {
            "criteria": {
                "min_delta_ppr_minus_one_step_ndcg_at_5": float(graph["headroom_min_delta"]),
                "ci_lower_gt_zero": True,
            },
            "exact": "pass" if pass_by_view["exact"] else "fail",
            "session_300s": "pass" if pass_by_view["session_300s"] else "fail",
            "decision": conclusion,
        },
    }


def run_evaluate(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    study = config["study"]
    metrics_path = project_path(study["metrics"])
    output_manifest = project_path(study["evaluation_manifest"])
    if metrics_path.exists() or output_manifest.exists():
        raise FileExistsError("graph-hard evaluation is already sealed")
    prepared, prepare_manifest = load_prepared(config, config_path)
    score_manifest_path = project_path(study["score_manifest"])
    scores_path = project_path(study["graph_scores"])
    for path in (score_manifest_path, scores_path):
        if not path.exists():
            raise FileNotFoundError(path)
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "decision": "locked",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": prepare_manifest["prepared"]["sha256"],
        "recommendation_labels_used_for_scoring": False,
        "recommendation_outcomes_evaluated": False,
    }
    for key, value in expected.items():
        if score_manifest.get(key) != value:
            raise RuntimeError(f"graph-hard score manifest mismatch: {key}")
    if score_manifest["graph_scores"]["sha256"] != sha256_file(scores_path):
        raise RuntimeError("graph-hard score hash mismatch")
    rows = read_jsonl(scores_path)
    metrics = headroom_metrics(prepared["development_events"], rows, config)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "sealed",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": prepare_manifest["prepared"]["sha256"],
        "score_manifest_sha256": sha256_file(score_manifest_path),
        "graph_scores_sha256": sha256_file(scores_path),
        "metrics": {"path": study["metrics"], "sha256": sha256_file(metrics_path)},
        "development_outcomes_opened_once": True,
        "fresh_primary_outcomes_accessed": False,
        "llm_requests": 0,
        "gpu_used": False,
    }
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens graph-hard multi-hop headroom")
    parser.add_argument("--config", default="configs/temporal_movielens32m/m5_graph_hard_headroom.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--data-smoke", action="store_true")
    modes.add_argument("--graph-smoke", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--score", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    parser.add_argument("--smoke-users", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.data_smoke:
        result = smoke_adapter(config, args.smoke_users)
    elif args.graph_smoke:
        result = run_graph_smoke(config, config_path)
    elif args.prepare:
        result = run_prepare(config, config_path)
    elif args.score:
        result = run_score(config, config_path)
    else:
        result = run_evaluate(config, config_path)
    printable = dict(result)
    if "arms" in printable:
        printable["arms"] = {
            view: {
                arm: {key: value for key, value in values.items() if key != "per_event_ndcg_at_5"}
                for arm, values in view_arms.items()
            }
            for view, view_arms in result["arms"].items()
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
