"""P7: directed temporal item-transition graph with multi-hop PPR."""
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
from src.temporal_books.p2_oracle import event_hit_at_5, event_ndcg_at_5
from src.temporal_books.p3_oracle import event_key
from src.temporal_books.p5_candidate_graph import rank_by_scores, successful_calls
from src.temporal_books.p6_adaptive_graph import (
    add_test_prompt_context,
    attach_candidates,
    first_novel_positive_targets,
    load_positive_graph,
    read_jsonl,
)


SCHEMA_VERSION = 1
TransitionEvent = tuple[int, str]
TransitionGraph = Dict[str, list[tuple[str, ...]]]


def load_transition_histories(config: Mapping[str, Any]) -> Dict[str, list[TransitionEvent]]:
    histories: DefaultDict[str, list[TransitionEvent]] = defaultdict(list)
    set_csv_field_limit()
    with project_path(config["dataset"]["reviews_file"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = (row.get("User_id") or "").strip()
            item_id = (row.get("Id") or "").strip()
            if not user_id or not item_id:
                continue
            try:
                timestamp = int(row["review/time"])
            except (KeyError, TypeError, ValueError):
                continue
            histories[user_id].append((timestamp, item_id))
    for values in histories.values():
        values.sort()
    return dict(histories)


def timestamp_batches(events: Sequence[TransitionEvent], *, cutoff: int | None = None) -> list[tuple[int, tuple[str, ...]]]:
    limit = len(events) if cutoff is None else bisect.bisect_left(events, (cutoff, ""))
    batches: list[tuple[int, tuple[str, ...]]] = []
    for timestamp, group in itertools.groupby(events[:limit], key=lambda event: event[0]):
        items = tuple(sorted({item_id for _, item_id in group}))
        if items:
            batches.append((timestamp, items))
    return batches


def build_transition_graph(histories: Mapping[str, Sequence[TransitionEvent]], *, cutoff: int) -> tuple[TransitionGraph, Dict[str, int]]:
    adjacency: DefaultDict[str, list[tuple[str, ...]]] = defaultdict(list)
    temporal_pairs = 0
    source_group_links = 0
    for events in histories.values():
        batches = timestamp_batches(events, cutoff=cutoff)
        for (_, sources), (_, destinations) in zip(batches, batches[1:]):
            temporal_pairs += 1
            for source in sources:
                adjacency[source].append(destinations)
                source_group_links += 1
    return dict(adjacency), {
        "users": len(histories),
        "source_items": len(adjacency),
        "temporal_batch_pairs": temporal_pairs,
        "source_to_group_links": source_group_links,
    }


def recent_seed_items(events: Sequence[TransitionEvent], target_time: int, *, limit: int) -> list[str]:
    end = bisect.bisect_left(events, (target_time, ""))
    seeds: list[str] = []
    seen: set[str] = set()
    for _, item_id in reversed(events[:end]):
        if item_id not in seen:
            seen.add(item_id)
            seeds.append(item_id)
        if len(seeds) >= limit:
            break
    return seeds


def one_step_scores(seeds: Sequence[str], candidates: Sequence[str], adjacency: Mapping[str, Sequence[tuple[str, ...]]]) -> Dict[str, float]:
    scores = {item_id: 0.0 for item_id in candidates}
    candidate_set = set(candidates)
    if not seeds:
        return scores
    seed_weight = 1.0 / len(seeds)
    for source in seeds:
        groups = adjacency.get(source, ())
        if not groups:
            continue
        group_weight = seed_weight / len(groups)
        for destinations in groups:
            destination_weight = group_weight / len(destinations)
            for item_id in destinations:
                if item_id in candidate_set:
                    scores[item_id] += destination_weight
    return scores


def ppr_monte_carlo_scores(
    seeds: Sequence[str],
    candidates: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, ...]]],
    *,
    walks: int,
    restart_probability: float,
    max_steps: int,
    seed: int,
) -> Dict[str, float]:
    if walks < 1 or max_steps < 1 or not 0 < restart_probability < 1:
        raise ValueError("invalid PPR Monte Carlo contract")
    counts = {item_id: 0 for item_id in candidates}
    candidate_set = set(candidates)
    if not seeds:
        return {item_id: 0.0 for item_id in candidates}
    rng = random.Random(seed)
    for _ in range(walks):
        current = seeds[rng.randrange(len(seeds))]
        for _ in range(max_steps):
            if rng.random() < restart_probability:
                break
            groups = adjacency.get(current, ())
            if not groups:
                break
            destinations = groups[rng.randrange(len(groups))]
            current = destinations[rng.randrange(len(destinations))]
        if current in candidate_set:
            counts[current] += 1
    return {item_id: count / walks for item_id, count in counts.items()}


def score_event(
    event: Mapping[str, Any],
    histories: Mapping[str, Sequence[TransitionEvent]],
    adjacency: Mapping[str, Sequence[tuple[str, ...]]],
    p7: Mapping[str, Any],
    *,
    walks: int,
) -> Dict[str, Any]:
    key = event_key(event)
    candidates = [str(value) for value in event["candidate_item_ids"]]
    seeds = recent_seed_items(histories.get(str(event["user_id"]), ()), int(event["timestamp"]), limit=int(p7["seed_items"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "event_key": key,
        "user_id": str(event["user_id"]),
        "timestamp": int(event["timestamp"]),
        "candidate_item_ids": candidates,
        "seed_item_ids": seeds,
        "one_step_scores": one_step_scores(seeds, candidates, adjacency),
        "ppr_scores": ppr_monte_carlo_scores(
            seeds,
            candidates,
            adjacency,
            walks=walks,
            restart_probability=float(p7["restart_probability"]),
            max_steps=int(p7["max_walk_steps"]),
            seed=stable_order(f"p7-ppr\0{key}"),
        ),
        "gold_label_used_for_scoring": False,
    }


def score_events(
    events: Sequence[Mapping[str, Any]],
    histories: Mapping[str, Sequence[TransitionEvent]],
    adjacency: Mapping[str, Sequence[tuple[str, ...]]],
    p7: Mapping[str, Any],
    *,
    walks: int,
) -> list[Dict[str, Any]]:
    return [score_event(event, histories, adjacency, p7, walks=walks) for event in events]


def residual_ranking(
    candidates: Sequence[str], local: Sequence[str], graph_scores: Mapping[str, float], *, alpha: float
) -> list[str]:
    values = [float(graph_scores[item_id]) for item_id in candidates]
    minimum, maximum = min(values, default=0.0), max(values, default=0.0)
    if alpha <= 0 or maximum <= minimum:
        return list(local)
    local_scores = {item_id: 1.0 / math.log2(index + 2) for index, item_id in enumerate(local)}
    combined = {
        item_id: local_scores[item_id] + alpha * (float(graph_scores[item_id]) - minimum) / (maximum - minimum)
        for item_id in candidates
    }
    return rank_by_scores(candidates, combined)


def prepare(config: Mapping[str, Any], config_path: Path, *, force: bool) -> Dict[str, Any]:
    p7 = config["p7"]
    output = project_path(p7["prepared"])
    if output.exists() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    for key, hash_key in (
        ("p6_prepared", "p6_prepared_sha256"),
        ("p6_calls", "p6_calls_sha256"),
        ("old_validation_prepared", "old_validation_prepared_sha256"),
    ):
        if sha256_file(project_path(p7[key])) != str(p7[hash_key]):
            raise RuntimeError(f"P7 frozen input mismatch: {key}")
    p6 = json.loads(project_path(p7["p6_prepared"]).read_text(encoding="utf-8"))
    old_validation = json.loads(project_path(p7["old_validation_prepared"]).read_text(encoding="utf-8"))
    excluded_users = {
        str(event["user_id"])
        for event in [*p6["train_events"], *p6["test_events"], *old_validation["events"]]
    }
    p0 = json.loads(project_path(config["dataset"]["p0_audit"]).read_text(encoding="utf-8"))
    train_cutoff = int(p0["temporal_split"]["train_cutoff"])
    validation_cutoff = int(p0["temporal_split"]["validation_cutoff"])
    # Reuse P6's audited positive-graph loader with the P7 protocol namespace.
    positive_histories, positive_item_events = load_positive_graph({**config, "p6": p7})
    candidate_pool = sorted(
        item_id for item_id, events in positive_item_events.items() if events and int(events[0][0]) < train_cutoff
    )
    targets = first_novel_positive_targets(
        positive_histories,
        start=validation_cutoff,
        end=None,
        candidate_pool=set(candidate_pool),
        minimum_history=int(p7["min_positive_history"]),
        excluded_users=excluded_users,
    )
    targets = sorted(
        targets,
        key=lambda row: stable_order(f"p7-test\0{row['user_id']}\0{row['timestamp']}\0{row['gold_item_id']}"),
    )[: int(p7["test_events"])]
    if len(targets) != int(p7["test_events"]):
        raise RuntimeError(f"insufficient disjoint P7 targets: {len(targets)}")
    events = attach_candidates(
        targets,
        positive_histories,
        candidate_pool,
        n_candidates=int(p7["candidates_per_event"]),
        prefix="p7-test-candidates",
    )
    events, item_info = add_test_prompt_context(events, config)
    prepared = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "reviews": {"path": config["dataset"]["reviews_file"], "sha256": sha256_file(project_path(config["dataset"]["reviews_file"]))},
            "metadata": {"path": config["dataset"]["metadata_file"], "sha256": sha256_file(project_path(config["dataset"]["metadata_file"]))},
            "p6_prepared": {"path": p7["p6_prepared"], "sha256": sha256_file(project_path(p7["p6_prepared"]))},
        },
        "protocol": {
            "graph_cutoff": validation_cutoff,
            "strict_timestamp_batches": True,
            "candidate_pool": "positive items first observed before global train cutoff",
            "calibration": "frozen P6 test cohort and local rankings",
            "fresh_test_disjoint_from_p6_and_old_validation": True,
        },
        "candidate_pool_items": len(candidate_pool),
        "test_events": events,
        "item_info": item_info,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return prepared


def graph_contract(p7: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: p7[key]
        for key in (
            "seed_items",
            "restart_probability",
            "monte_carlo_walks",
            "max_walk_steps",
            "fusion_alpha_grid",
        )
    }


def smoke_subsets(config: Mapping[str, Any], prepared: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    p7 = config["p7"]
    p6 = json.loads(project_path(p7["p6_prepared"]).read_text(encoding="utf-8"))
    calibration = sorted(
        p6["test_events"], key=lambda event: stable_order(f"p7-smoke-calibration\0{event_key(event)}")
    )[: int(p7["smoke_calibration_events"])]
    test = sorted(
        prepared["test_events"], key=lambda event: stable_order(f"p7-smoke-test\0{event_key(event)}")
    )[: int(p7["smoke_test_events"])]
    return calibration, test


def run_graph_smoke(config: Mapping[str, Any], config_path: Path, prepared: Mapping[str, Any]) -> None:
    p7 = config["p7"]
    histories = load_transition_histories(config)
    graph_cutoff = int(prepared["protocol"]["graph_cutoff"])
    adjacency, stats = build_transition_graph(histories, cutoff=graph_cutoff)
    calibration, test = smoke_subsets(config, prepared)
    started = time.monotonic()
    rows = score_events(
        [*calibration, *test], histories, adjacency, p7, walks=int(p7["smoke_walks"])
    )
    repeated = score_event(calibration[0], histories, adjacency, p7, walks=int(p7["smoke_walks"]))
    if rows[0] != repeated:
        raise RuntimeError("P7 Monte Carlo smoke is not deterministic")
    if any(len(row["seed_item_ids"]) > int(p7["seed_items"]) for row in rows):
        raise RuntimeError("P7 seed cap violated")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p7["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": sha256_file(project_path(p7["prepared"])),
        "graph_contract": graph_contract(p7),
        "graph_stats": stats,
        "calibration_event_keys": [event_key(event) for event in calibration],
        "test_event_keys": [event_key(event) for event in test],
        "candidate_slots": sum(len(row["candidate_item_ids"]) for row in rows),
        "one_step_nonzero_slots": sum(score > 0 for row in rows for score in row["one_step_scores"].values()),
        "ppr_nonzero_slots": sum(score > 0 for row in rows for score in row["ppr_scores"].values()),
        "deterministic_replay": True,
        "gold_labels_used_for_scoring": False,
        "elapsed_seconds_after_graph_build": time.monotonic() - started,
    }
    path = project_path(p7["graph_smoke"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def verify_smoke(config: Mapping[str, Any], config_path: Path, prepared: Mapping[str, Any]) -> Dict[str, Any]:
    p7 = config["p7"]
    path = project_path(p7["graph_smoke"])
    if not path.exists():
        raise RuntimeError("P7 full calibration blocked: run --graph-smoke first")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": sha256_file(project_path(p7["prepared"])),
        "graph_contract": graph_contract(p7),
    }
    for key, value in expected.items():
        if smoke.get(key) != value:
            raise RuntimeError(f"P7 graph smoke mismatch: {key}")
    return smoke


def calibration_metrics(
    rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]], calls: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    events_by_key = {event_key(event): event for event in events}
    p7 = config["p7"]
    local_ndcgs = []
    graph_ndcgs = []
    grid: list[Dict[str, Any]] = []
    for row in rows:
        event = events_by_key[str(row["event_key"])]
        gold = str(event["gold_item_id"])
        local = calls[f"rerank:local:{row['event_key']}"]["value"]
        local_ndcgs.append(event_ndcg_at_5(local, gold))
        graph_ndcgs.append(event_ndcg_at_5(rank_by_scores(row["candidate_item_ids"], row["ppr_scores"]), gold))
    for alpha in [float(value) for value in p7["fusion_alpha_grid"]]:
        ndcgs, hits, changed = [], [], 0
        for row in rows:
            event = events_by_key[str(row["event_key"])]
            gold = str(event["gold_item_id"])
            local = calls[f"rerank:local:{row['event_key']}"]["value"]
            ranking = residual_ranking(row["candidate_item_ids"], local, row["ppr_scores"], alpha=alpha)
            ndcgs.append(event_ndcg_at_5(ranking, gold))
            hits.append(event_hit_at_5(ranking, gold))
            changed += int(ranking != local)
        grid.append({"alpha": alpha, "ndcg_at_5": sum(ndcgs) / len(ndcgs), "hit_at_5": sum(hits) / len(hits), "changed_events": changed})
    selected = sorted(grid, key=lambda row: (-row["ndcg_at_5"], row["alpha"]))[0]
    local_mean = sum(local_ndcgs) / len(local_ndcgs)
    delta = float(selected["ndcg_at_5"]) - local_mean
    admitted = float(selected["alpha"]) > 0 and delta >= float(p7["calibration_min_delta_to_run_test"])
    return {
        "events": len(rows),
        "local_ndcg_at_5": local_mean,
        "ppr_graph_only_ndcg_at_5": sum(graph_ndcgs) / len(graph_ndcgs),
        "grid": grid,
        "selected": selected,
        "selected_delta_vs_local": delta,
        "test_admission": {
            "criteria": {"alpha_gt_zero": True, "min_delta": float(p7["calibration_min_delta_to_run_test"])},
            "decision": "admit" if admitted else "hard_stop",
        },
    }


def run_calibrate(config: Mapping[str, Any], config_path: Path, prepared: Mapping[str, Any], *, force: bool) -> None:
    p7 = config["p7"]
    smoke = verify_smoke(config, config_path, prepared)
    scores_path = project_path(p7["calibration_scores"])
    selection_path = project_path(p7["locked_selection"])
    manifest_path = project_path(p7["calibration_manifest"])
    if any(path.exists() for path in (scores_path, selection_path, manifest_path)) and not force:
        raise FileExistsError("P7 calibration output exists; use --force only for an intentional rerun")
    p6 = json.loads(project_path(p7["p6_prepared"]).read_text(encoding="utf-8"))
    calls = successful_calls(project_path(p7["p6_calls"]))
    histories = load_transition_histories(config)
    adjacency, stats = build_transition_graph(histories, cutoff=int(prepared["protocol"]["graph_cutoff"]))
    started = time.monotonic()
    rows = score_events(
        p6["test_events"], histories, adjacency, p7, walks=int(p7["monte_carlo_walks"])
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    records, scores_sha = jsonl_write(scores_path, rows)
    metrics = calibration_metrics(rows, p6["test_events"], calls, config)
    locked = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p7["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph_contract": graph_contract(p7),
        "graph_stats": stats,
        "calibration": metrics,
    }
    selection_path.write_text(json.dumps(locked, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p7["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "prepared": {"path": p7["prepared"], "sha256": sha256_file(project_path(p7["prepared"]))},
        "graph_smoke": {"path": p7["graph_smoke"], "sha256": sha256_file(project_path(p7["graph_smoke"])), "decision": smoke["decision"]},
        "calibration_scores": {"path": p7["calibration_scores"], "records": records, "sha256": scores_sha},
        "locked_selection": {"path": p7["locked_selection"], "sha256": sha256_file(selection_path)},
        "fresh_test_labels_accessed": False,
        "llm_requests": 0,
        "gpu_used": False,
        "runtime_seconds_after_graph_build": time.monotonic() - started,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def analyze_test(rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    events_by_key = {event_key(event): event for event in events}
    local_values, ppr_values, local_hits, ppr_hits = [], [], [], []
    coverage = {"gold_one_step": 0, "gold_ppr": 0, "negative_one_step": 0, "negative_ppr": 0, "negative_slots": 0}
    changed = 0
    for row in rows:
        event = events_by_key[str(row["event_key"])]
        gold = str(event["gold_item_id"])
        local, ppr = row["rankings"]["local"], row["rankings"]["ppr_residual"]
        local_values.append(event_ndcg_at_5(local, gold))
        ppr_values.append(event_ndcg_at_5(ppr, gold))
        local_hits.append(event_hit_at_5(local, gold))
        ppr_hits.append(event_hit_at_5(ppr, gold))
        changed += int(local != ppr)
        coverage["gold_one_step"] += int(float(row["one_step_scores"][gold]) > 0)
        coverage["gold_ppr"] += int(float(row["ppr_scores"][gold]) > 0)
        for item_id in row["candidate_item_ids"]:
            if item_id == gold:
                continue
            coverage["negative_slots"] += 1
            coverage["negative_one_step"] += int(float(row["one_step_scores"][item_id]) > 0)
            coverage["negative_ppr"] += int(float(row["ppr_scores"][item_id]) > 0)
    delta = [value - base for value, base in zip(ppr_values, local_values)]
    p7 = config["p7"]
    lower, upper = paired_bootstrap_ci(delta, resamples=int(p7["bootstrap_resamples"]), seed=int(p7["bootstrap_seed"]))
    mean_delta = sum(delta) / len(delta)
    passed = mean_delta >= float(p7["admission_min_delta"]) and lower > 0
    oracle = [max(base, value) for base, value in zip(local_values, ppr_values)]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": p7["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": {
            "fresh_local": {"n_events": len(rows), "ndcg_at_5": sum(local_values) / len(local_values), "hit_at_5": sum(local_hits) / len(local_hits), "per_event_ndcg_at_5": local_values},
            "transition_ppr_residual": {"n_events": len(rows), "ndcg_at_5": sum(ppr_values) / len(ppr_values), "hit_at_5": sum(ppr_hits) / len(ppr_hits), "changed_events": changed, "per_event_ndcg_at_5": ppr_values},
        },
        "comparison": {"delta_ndcg_at_5": mean_delta, "paired_bootstrap_95_ci": [lower, upper], "improved_events": sum(value > 0 for value in delta), "worsened_events": sum(value < 0 for value in delta), "unchanged_events": sum(value == 0 for value in delta)},
        "evidence_coverage": {**coverage, "negative_one_step_rate": coverage["negative_one_step"] / coverage["negative_slots"], "negative_ppr_rate": coverage["negative_ppr"] / coverage["negative_slots"]},
        "posthoc_oracle_best_of_two": {"ndcg_at_5": sum(oracle) / len(oracle), "delta_vs_local": sum(oracle) / len(oracle) - sum(local_values) / len(local_values), "deployable": False},
        "gate": {"criteria": {"min_delta": float(p7["admission_min_delta"]), "ci_lower_gt_zero": True}, "decision": "pass" if passed else "hard_stop"},
    }


def run_evaluate(config: Mapping[str, Any], config_path: Path, prepared: Mapping[str, Any], *, force: bool) -> None:
    p7, self_host = config["p7"], config["self_host"]
    verify_smoke(config, config_path, prepared)
    selection_path = project_path(p7["locked_selection"])
    local_manifest_path = project_path(self_host["manifest"])
    for path in (selection_path, project_path(p7["calibration_manifest"]), local_manifest_path, project_path(self_host["calls"])):
        if not path.exists():
            raise FileNotFoundError(path)
    locked = json.loads(selection_path.read_text(encoding="utf-8"))
    if locked["calibration"]["test_admission"]["decision"] != "admit":
        raise RuntimeError("P7 fresh evaluation forbidden: calibration hard stop")
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    if local_manifest.get("decision") != "complete" or local_manifest.get("prepared_sha256") != sha256_file(project_path(p7["prepared"])):
        raise RuntimeError("P7 local manifest mismatch")
    scores_path, metrics_path, manifest_path = (project_path(p7[key]) for key in ("test_scores", "metrics", "manifest"))
    if any(path.exists() for path in (scores_path, metrics_path, manifest_path)) and not force:
        raise FileExistsError("P7 test output exists; use --force only for an intentional rerun")
    histories = load_transition_histories(config)
    adjacency, _ = build_transition_graph(histories, cutoff=int(prepared["protocol"]["graph_cutoff"]))
    started = time.monotonic()
    rows = score_events(prepared["test_events"], histories, adjacency, p7, walks=int(p7["monte_carlo_walks"]))
    calls = successful_calls(project_path(self_host["calls"]))
    alpha = float(locked["calibration"]["selected"]["alpha"])
    for row in rows:
        local = calls[f"rerank:local:{row['event_key']}"]["value"]
        row["rankings"] = {
            "local": local,
            "ppr_graph_only": rank_by_scores(row["candidate_item_ids"], row["ppr_scores"]),
            "ppr_residual": residual_ranking(row["candidate_item_ids"], local, row["ppr_scores"], alpha=alpha),
        }
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    records, scores_sha = jsonl_write(scores_path, rows)
    metrics = analyze_test(rows, prepared["test_events"], config)
    metrics["locked_alpha"] = alpha
    metrics["runtime_seconds_after_graph_build"] = time.monotonic() - started
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p7["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "prepared": {"path": p7["prepared"], "sha256": sha256_file(project_path(p7["prepared"]))},
        "locked_selection": {"path": p7["locked_selection"], "sha256": sha256_file(selection_path)},
        "local_manifest": {"path": self_host["manifest"], "sha256": sha256_file(local_manifest_path)},
        "test_scores": {"path": p7["test_scores"], "records": records, "sha256": scores_sha},
        "metrics": {"path": p7["metrics"], "sha256": sha256_file(metrics_path)},
        "test_tuning_after_labels": False,
        "graph_llm_requests": 0,
        "graph_gpu_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    printable = {name: {key: value for key, value in arm.items() if key != "per_event_ndcg_at_5"} for name, arm in metrics["arms"].items()}
    print(json.dumps({"arms": printable, "comparison": metrics["comparison"], "coverage": metrics["evidence_coverage"], "oracle": metrics["posthoc_oracle_best_of_two"], "gate": metrics["gate"], "alpha": alpha}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7 temporal transition PPR")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p7_transition_ppr.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--graph-smoke", action="store_true")
    modes.add_argument("--calibrate", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    p7 = config["p7"]
    prepared = prepare(config, config_path, force=args.force if args.prepare else False)
    if args.prepare or not any((args.graph_smoke, args.calibrate, args.evaluate)):
        print(json.dumps({"prepared": p7["prepared"], "test_events": len(prepared["test_events"]), "llm_requests": 0}, ensure_ascii=False, indent=2))
    elif args.graph_smoke:
        run_graph_smoke(config, config_path, prepared)
    elif args.calibrate:
        run_calibrate(config, config_path, prepared, force=args.force)
    elif args.evaluate:
        run_evaluate(config, config_path, prepared, force=args.force)


if __name__ == "__main__":
    main()
