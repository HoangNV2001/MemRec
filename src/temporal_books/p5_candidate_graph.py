"""Offline candidate-directed graph scoring at three and five item layers."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Mapping, Sequence

from src.temporal_books.common import PROJECT_ROOT, jsonl_write, load_yaml, project_path, set_csv_field_limit, sha256_file, stable_bucket, stable_order
from src.temporal_books.p2_analysis import paired_bootstrap_ci
from src.temporal_books.p2_oracle import event_hit_at_5, event_ndcg_at_5
from src.temporal_books.p3_oracle import event_key


SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86400.0
ScoredEvent = tuple[int, str, float]


@dataclass(frozen=True)
class SearchState:
    current_item: str
    items_from_candidate: tuple[str, ...]
    users_from_candidate: tuple[str, ...]
    log_score: float


def history_before(events: Sequence[ScoredEvent], timestamp: int) -> Sequence[ScoredEvent]:
    return events[: bisect.bisect_left(events, (timestamp, "", float("-inf")))]


def recent_unique_items(events: Sequence[ScoredEvent], timestamp: int, *, excluded: set[str], limit: int) -> list[ScoredEvent]:
    values: list[ScoredEvent] = []
    seen = set(excluded)
    for event in reversed(history_before(events, timestamp)):
        if event[1] not in seen:
            seen.add(event[1])
            values.append(event)
        if len(values) >= limit:
            break
    return values


def recent_unique_peers(
    item_events: Mapping[str, Sequence[tuple[int, str, float]]],
    item_id: str,
    timestamp: int,
    *,
    excluded: set[str],
    limit: int,
) -> list[tuple[int, str, float]]:
    values: list[tuple[int, str, float]] = []
    seen = set(excluded)
    events = item_events.get(item_id, ())
    end = bisect.bisect_left(events, (timestamp, "", float("-inf")))
    for event in reversed(events[:end]):
        if event[1] not in seen:
            seen.add(event[1])
            values.append(event)
        if len(values) >= limit:
            break
    return values


def edge_log_score(timestamp: int, rating: float, target_time: int, *, minimum_rating: float, half_life_days: float) -> float:
    age_days = max(0.0, (target_time - timestamp) / SECONDS_PER_DAY)
    recency = 2.0 ** (-age_days / half_life_days)
    rating_strength = max(1e-6, min(1.0, (rating - minimum_rating + 1.0) / 2.0))
    return math.log(recency) + math.log(rating_strength)


def source_anchors(
    histories: Mapping[str, Sequence[ScoredEvent]],
    user_id: str,
    target_time: int,
    *,
    limit: int,
) -> Dict[str, tuple[int, float]]:
    if user_id not in histories:
        return {}
    values: Dict[str, tuple[int, float]] = {}
    for timestamp, item_id, rating in reversed(history_before(histories[user_id], target_time)):
        if item_id not in values:
            values[item_id] = (timestamp, rating)
        if len(values) >= limit:
            break
    return values


def candidate_paths(
    histories: Mapping[str, Sequence[ScoredEvent]],
    item_events: Mapping[str, Sequence[tuple[int, str, float]]],
    *,
    source_user: str,
    target_time: int,
    candidate_item: str,
    minimum_rating: float,
    half_life_days: float,
    length_decay: float,
    hub_penalty_weight: float,
    source_anchor_limit: int,
    peers_per_item: int,
    items_per_peer: int,
    beam_width: int,
    max_item_layers: int,
) -> list[Dict[str, Any]]:
    if not 0 < length_decay <= 1:
        raise ValueError("length_decay must be in (0, 1]")
    anchors = source_anchors(histories, source_user, target_time, limit=source_anchor_limit)
    if not anchors:
        return []
    states = [SearchState(candidate_item, (candidate_item,), (), 0.0)]
    paths: list[Dict[str, Any]] = []
    for item_layers in range(2, max_item_layers + 1):
        expanded: list[SearchState] = []
        for state in states:
            current_events = item_events.get(state.current_item, ())
            degree = bisect.bisect_left(current_events, (target_time, "", float("-inf")))
            hub_log = -hub_penalty_weight * math.log(max(1.0, math.log2(2 + degree)))
            for peer_time, peer_user, peer_rating in recent_unique_peers(
                item_events,
                state.current_item,
                target_time,
                excluded={source_user, *state.users_from_candidate},
                limit=peers_per_item,
            ):
                for item_time, next_item, item_rating in recent_unique_items(
                    histories.get(peer_user, ()),
                    target_time,
                    excluded=set(state.items_from_candidate),
                    limit=items_per_peer,
                ):
                    log_score = (
                        state.log_score
                        + edge_log_score(peer_time, peer_rating, target_time, minimum_rating=minimum_rating, half_life_days=half_life_days)
                        + edge_log_score(item_time, item_rating, target_time, minimum_rating=minimum_rating, half_life_days=half_life_days)
                        + hub_log
                        + math.log(length_decay)
                    )
                    new_state = SearchState(
                        next_item,
                        state.items_from_candidate + (next_item,),
                        state.users_from_candidate + (peer_user,),
                        log_score,
                    )
                    expanded.append(new_state)
                    if next_item in anchors:
                        anchor_time, anchor_rating = anchors[next_item]
                        final_log = log_score + edge_log_score(
                            anchor_time,
                            anchor_rating,
                            target_time,
                            minimum_rating=minimum_rating,
                            half_life_days=half_life_days,
                        )
                        paths.append(
                            {
                                "item_layers": item_layers,
                                "score": math.exp(final_log),
                                "anchor_item_id": next_item,
                                "candidate_peer_user": state.users_from_candidate[0] if state.users_from_candidate else peer_user,
                                "items_from_candidate": list(new_state.items_from_candidate),
                                "users_from_candidate": list(new_state.users_from_candidate),
                            }
                        )
        best: Dict[tuple[str, str, str], SearchState] = {}
        for state in expanded:
            key = (state.current_item, state.users_from_candidate[0], state.users_from_candidate[-1])
            if key not in best or state.log_score > best[key].log_score:
                best[key] = state
        states = sorted(best.values(), key=lambda value: (-value.log_score, value.current_item, value.users_from_candidate))[:beam_width]
        if not states:
            break
    return paths


def aggregate_paths(paths: Sequence[Mapping[str, Any]], *, max_item_layers: int, max_paths: int) -> Dict[str, Any]:
    selected: list[Mapping[str, Any]] = []
    signatures: set[tuple[str, str]] = set()
    eligible = [path for path in paths if int(path["item_layers"]) <= max_item_layers]
    for path in sorted(eligible, key=lambda row: (-float(row["score"]), int(row["item_layers"]), str(row["anchor_item_id"]))):
        signature = (str(path["anchor_item_id"]), str(path["candidate_peer_user"]))
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append(path)
        if len(selected) >= max_paths:
            break
    scores = [float(path["score"]) for path in selected]
    return {
        "score": sum(scores),
        "best_path_score": max(scores, default=0.0),
        "selected_paths": len(selected),
        "independent_anchors": len({str(path["anchor_item_id"]) for path in selected}),
        "candidate_peers": len({str(path["candidate_peer_user"]) for path in selected}),
        "min_item_layers": min((int(path["item_layers"]) for path in selected), default=None),
        "paths": selected,
    }


def rank_by_scores(candidates: Sequence[str], scores: Mapping[str, float]) -> list[str]:
    order = {item_id: index for index, item_id in enumerate(candidates)}
    return sorted(candidates, key=lambda item_id: (-float(scores.get(item_id, 0.0)), order[item_id]))


def residual_ranking(candidates: Sequence[str], local_ranking: Sequence[str], graph_scores: Mapping[str, float], *, alpha: float) -> list[str]:
    local = {item_id: 1.0 / math.log2(index + 2) for index, item_id in enumerate(local_ranking)}
    maximum = max((float(graph_scores.get(item_id, 0.0)) for item_id in candidates), default=0.0)
    combined = {
        item_id: local[item_id] + alpha * (float(graph_scores.get(item_id, 0.0)) / maximum if maximum > 0 else 0.0)
        for item_id in candidates
    }
    return rank_by_scores(candidates, combined)


def load_graph(config: Mapping[str, Any]) -> tuple[Dict[str, list[ScoredEvent]], Dict[str, list[tuple[int, str, float]]]]:
    dataset, p1, p5 = config["dataset"], config["p1"], config["p5"]
    p0 = json.loads(project_path(dataset["p0_audit"]).read_text(encoding="utf-8"))
    validation_cutoff = int(p0["temporal_split"]["validation_cutoff"])
    modulus, remainder = int(p1["pilot_user_hash_modulus"]), int(p1["pilot_user_hash_remainder"])
    minimum = float(p5["positive_rating_min"])
    histories: DefaultDict[str, list[ScoredEvent]] = defaultdict(list)
    item_events: DefaultDict[str, list[tuple[int, str, float]]] = defaultdict(list)
    set_csv_field_limit()
    with project_path(dataset["reviews_file"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id, item_id = (row.get("User_id") or "").strip(), (row.get("Id") or "").strip()
            if not user_id or not item_id or stable_bucket(user_id, modulus) != remainder:
                continue
            try:
                timestamp, rating = int(row["review/time"]), float(row["review/score"])
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp < validation_cutoff and rating >= minimum:
                histories[user_id].append((timestamp, item_id, rating))
                item_events[item_id].append((timestamp, user_id, rating))
    for events in histories.values():
        events.sort()
    for events in item_events.values():
        events.sort()
    return dict(histories), dict(item_events)


def successful_calls(path: Path) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "success":
                key = str(row["key"])
                if key in values:
                    raise ValueError(f"duplicate frozen call: {key}")
                values[key] = row
    return values


def protocol(config: Mapping[str, Any]) -> Dict[str, Any]:
    p5 = config["p5"]
    return {key: p5[key] for key in (
        "positive_rating_min", "recency_half_life_days", "length_decay", "hub_penalty_weight",
        "source_anchor_cap", "peers_per_item", "items_per_peer", "beam_width",
        "max_paths_per_candidate", "shallow_item_layers", "deep_item_layers", "residual_alpha",
    )}


def score_event(
    event: Mapping[str, Any],
    histories: Mapping[str, Sequence[ScoredEvent]],
    item_events: Mapping[str, Sequence[tuple[int, str, float]]],
    frozen_calls: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    p5 = config["p5"]
    candidates = [str(value) for value in event["candidate_item_ids"]]
    maximum = int(p5["deep_item_layers"])
    candidate_rows: Dict[str, Any] = {}
    for candidate in candidates:
        paths = candidate_paths(
            histories,
            item_events,
            source_user=str(event["user_id"]),
            target_time=int(event["timestamp"]),
            candidate_item=candidate,
            minimum_rating=float(p5["positive_rating_min"]),
            half_life_days=float(p5["recency_half_life_days"]),
            length_decay=float(p5["length_decay"]),
            hub_penalty_weight=float(p5["hub_penalty_weight"]),
            source_anchor_limit=int(p5["source_anchor_cap"]),
            peers_per_item=int(p5["peers_per_item"]),
            items_per_peer=int(p5["items_per_peer"]),
            beam_width=int(p5["beam_width"]),
            max_item_layers=maximum,
        )
        candidate_rows[candidate] = {
            "layer3": aggregate_paths(paths, max_item_layers=int(p5["shallow_item_layers"]), max_paths=int(p5["max_paths_per_candidate"])),
            "layer5": aggregate_paths(paths, max_item_layers=maximum, max_paths=int(p5["max_paths_per_candidate"])),
        }
    local = frozen_calls[f"rerank:local:{event_key(event)}"]["value"]
    graph3 = {item_id: float(row["layer3"]["score"]) for item_id, row in candidate_rows.items()}
    graph5 = {item_id: float(row["layer5"]["score"]) for item_id, row in candidate_rows.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "event_key": event_key(event),
        "user_id": str(event["user_id"]),
        "timestamp": int(event["timestamp"]),
        "candidate_item_ids": candidates,
        "candidate_features": candidate_rows,
        "rankings": {
            "local": local,
            "graph_layer3": rank_by_scores(candidates, graph3),
            "graph_layer5": rank_by_scores(candidates, graph5),
            "residual_layer3": residual_ranking(candidates, local, graph3, alpha=float(p5["residual_alpha"])),
            "residual_layer5": residual_ranking(candidates, local, graph5, alpha=float(p5["residual_alpha"])),
        },
    }


def compare(values: Sequence[float], *, config: Mapping[str, Any]) -> Dict[str, Any]:
    p5 = config["p5"]
    lower, upper = paired_bootstrap_ci(values, resamples=int(p5["bootstrap_resamples"]), seed=int(p5["bootstrap_seed"]))
    return {
        "delta_ndcg_at_5": sum(values) / len(values),
        "paired_bootstrap_95_ci": [lower, upper],
        "improved_events": sum(value > 0 for value in values),
        "worsened_events": sum(value < 0 for value in values),
        "unchanged_events": sum(value == 0 for value in values),
    }


def analyze(rows: Sequence[Mapping[str, Any]], events_by_key: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    arms = ("local", "graph_layer3", "graph_layer5", "residual_layer3", "residual_layer5")
    per_arm: Dict[str, list[float]] = {arm: [] for arm in arms}
    hits: Dict[str, list[float]] = {arm: [] for arm in arms}
    evidence = {"layer3_gold": 0, "layer5_gold": 0, "layer3_negative_slots": 0, "layer5_negative_slots": 0, "negative_slots": 0}
    for row in rows:
        event = events_by_key[str(row["event_key"])]
        gold = str(event["gold_item_id"])
        for arm in arms:
            ranking = row["rankings"][arm]
            per_arm[arm].append(event_ndcg_at_5(ranking, gold))
            hits[arm].append(event_hit_at_5(ranking, gold))
        features = row["candidate_features"]
        evidence["layer3_gold"] += int(float(features[gold]["layer3"]["score"]) > 0)
        evidence["layer5_gold"] += int(float(features[gold]["layer5"]["score"]) > 0)
        for item_id in row["candidate_item_ids"]:
            if item_id == gold:
                continue
            evidence["negative_slots"] += 1
            evidence["layer3_negative_slots"] += int(float(features[item_id]["layer3"]["score"]) > 0)
            evidence["layer5_negative_slots"] += int(float(features[item_id]["layer5"]["score"]) > 0)
    arm_metrics = {arm: {"n_events": len(rows), "ndcg_at_5": sum(per_arm[arm]) / len(rows), "hit_at_5": sum(hits[arm]) / len(rows), "per_event_ndcg_at_5": per_arm[arm]} for arm in arms}
    vs_local = [value - base for value, base in zip(per_arm["residual_layer5"], per_arm["local"])]
    vs_shallow = [value - base for value, base in zip(per_arm["residual_layer5"], per_arm["residual_layer3"])]
    comparisons = {"residual_layer5_vs_local": compare(vs_local, config=config), "residual_layer5_vs_layer3": compare(vs_shallow, config=config)}
    p5 = config["p5"]
    gate = (
        comparisons["residual_layer5_vs_local"]["delta_ndcg_at_5"] >= float(p5["deep_vs_local_min_delta"])
        and comparisons["residual_layer5_vs_local"]["paired_bootstrap_95_ci"][0] > 0
        and comparisons["residual_layer5_vs_layer3"]["delta_ndcg_at_5"] >= float(p5["deep_vs_shallow_min_delta"])
        and comparisons["residual_layer5_vs_layer3"]["paired_bootstrap_95_ci"][0] > 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": p5["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": arm_metrics,
        "evidence_coverage": {
            "events": len(rows),
            "layer3_gold_events": evidence["layer3_gold"],
            "layer5_gold_events": evidence["layer5_gold"],
            "layer3_negative_slot_rate": evidence["layer3_negative_slots"] / evidence["negative_slots"],
            "layer5_negative_slot_rate": evidence["layer5_negative_slots"] / evidence["negative_slots"],
        },
        "comparisons": comparisons,
        "gate": {
            "criteria": {"deep_vs_local_min_delta": float(p5["deep_vs_local_min_delta"]), "deep_vs_shallow_min_delta": float(p5["deep_vs_shallow_min_delta"]), "both_ci_lower_gt_zero": True},
            "decision": "pass" if gate else "hard_stop",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate-directed 3/5-layer graph scoring")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p5_candidate_graph.yaml")
    parser.add_argument("--smoke-events", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    dataset, p5 = config["dataset"], config["p5"]
    prepared_path, calls_path = project_path(p5["p3_prepared"]), project_path(p5["p3_calls"])
    if sha256_file(prepared_path) != str(p5["p3_prepared_sha256"]) or sha256_file(calls_path) != str(p5["p3_calls_sha256"]):
        raise RuntimeError("frozen P3 input hash mismatch")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    frozen = successful_calls(calls_path)
    events = list(prepared["events"])
    required_local = [f"rerank:local:{event_key(event)}" for event in events]
    if any(key not in frozen for key in required_local):
        raise RuntimeError("frozen local ranking cache incomplete")
    histories, item_events = load_graph(config)
    started = time.monotonic()
    smoke_count = int(p5["smoke_events"])
    selected_smoke = sorted(events, key=lambda event: stable_order(f"p5-smoke\0{event_key(event)}"))[:smoke_count]
    if args.smoke_events is not None:
        if args.smoke_events != smoke_count or not 20 <= smoke_count <= 30:
            raise ValueError("P5 smoke must use preregistered 20-30 events")
        rows = [score_event(event, histories, item_events, frozen, config) for event in selected_smoke]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "pass",
            "config_sha256": sha256_file(config_path),
            "prepared_sha256": sha256_file(prepared_path),
            "calls_sha256": sha256_file(calls_path),
            "protocol": protocol(config),
            "event_keys": [event_key(event) for event in selected_smoke],
            "candidate_slots": sum(len(row["candidate_item_ids"]) for row in rows),
            "layer3_slots_with_evidence": sum(row["candidate_features"][item]["layer3"]["score"] > 0 for row in rows for item in row["candidate_item_ids"]),
            "layer5_slots_with_evidence": sum(row["candidate_features"][item]["layer5"]["score"] > 0 for row in rows for item in row["candidate_item_ids"]),
            "gold_labels_used_for_scoring": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        output = project_path(p5["smoke_manifest"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    smoke_path = project_path(p5["smoke_manifest"])
    if not smoke_path.exists():
        raise RuntimeError("P5 full probe blocked: run smoke first")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    expected = {"decision": "pass", "config_sha256": sha256_file(config_path), "prepared_sha256": sha256_file(prepared_path), "calls_sha256": sha256_file(calls_path), "protocol": protocol(config), "event_keys": [event_key(event) for event in selected_smoke]}
    for key, value in expected.items():
        if smoke.get(key) != value:
            raise RuntimeError(f"P5 full probe blocked: smoke {key} mismatch")
    scores_path, metrics_path, manifest_path = project_path(p5["scores"]), project_path(p5["metrics"]), project_path(p5["manifest"])
    if any(path.exists() for path in (scores_path, metrics_path, manifest_path)) and not args.force:
        raise FileExistsError("P5 derived output exists; use --force to replace")
    rows = [score_event(event, histories, item_events, frozen, config) for event in events]
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    records, scores_sha = jsonl_write(scores_path, rows)
    events_by_key = {event_key(event): event for event in events}
    metrics = analyze(rows, events_by_key, config)
    metrics["runtime"] = {"elapsed_seconds": time.monotonic() - started, "sampled_users": len(histories), "positive_items": len(item_events)}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p5["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "reviews": {"path": dataset["reviews_file"], "sha256": sha256_file(project_path(dataset["reviews_file"]))},
            "prepared": {"path": str(prepared_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(prepared_path)},
            "p3_calls": {"path": str(calls_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(calls_path)},
            "smoke": {"path": str(smoke_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(smoke_path)},
        },
        "protocol": protocol(config),
        "scores": {"path": str(scores_path.relative_to(PROJECT_ROOT)), "records": records, "sha256": scores_sha},
        "metrics": {"path": str(metrics_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(metrics_path)},
        "llm_requests": 0,
        "gpu_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    printable = {arm: {key: value for key, value in row.items() if key != "per_event_ndcg_at_5"} for arm, row in metrics["arms"].items()}
    print(json.dumps({"arms": printable, "evidence_coverage": metrics["evidence_coverage"], "comparisons": metrics["comparisons"], "gate": metrics["gate"], "runtime": metrics["runtime"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
