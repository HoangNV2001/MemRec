"""Candidate-blind filtered 3-hop routes for the Amazon Books temporal pilot."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Mapping, Sequence

from src.temporal_books.common import PROJECT_ROOT, jsonl_write, load_yaml, project_path, set_csv_field_limit, sha256_file, stable_bucket
from src.temporal_books.p1_preflight import coverage


SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86400.0
ScoredEvent = tuple[int, str, float, str, str]


def history_before(events: Sequence[ScoredEvent], timestamp: int) -> Sequence[ScoredEvent]:
    return events[: bisect.bisect_left(events, (timestamp, "", float("-inf"), "", ""))]


def latest_anchor_event(events: Sequence[ScoredEvent], item_id: str, packet_time: int) -> ScoredEvent | None:
    for event in reversed(events):
        if event[0] <= packet_time and event[1] == item_id:
            return event
    return None


def recent_positive_peers(
    item_events: Mapping[str, Sequence[tuple[int, str, float]]],
    item_id: str,
    packet_time: int,
    *,
    excluded_users: set[str],
    limit: int,
) -> list[tuple[str, int, float]]:
    candidates = item_events.get(item_id, ())
    end = bisect.bisect_left(candidates, (packet_time, "", float("-inf")))
    values: list[tuple[str, int, float]] = []
    seen = set(excluded_users)
    for timestamp, user_id, rating in reversed(candidates[:end]):
        if user_id not in seen:
            seen.add(user_id)
            values.append((user_id, timestamp, rating))
        if len(values) >= limit:
            break
    return values


def recent_positive_items(
    events: Sequence[ScoredEvent],
    packet_time: int,
    *,
    excluded_items: set[str],
    minimum_rating: float,
    limit: int,
) -> list[tuple[str, int, float]]:
    values: list[tuple[str, int, float]] = []
    seen = set(excluded_items)
    for timestamp, item_id, rating, _, _ in reversed(history_before(events, packet_time)):
        if rating >= minimum_rating and item_id not in seen:
            seen.add(item_id)
            values.append((item_id, timestamp, rating))
        if len(values) >= limit:
            break
    return values


def path_quality(
    *,
    packet_time: int,
    timestamps: Sequence[int],
    ratings: Sequence[float],
    anchor_degree: int,
    bridge_degree: int,
    minimum_rating: float,
    half_life_days: float,
) -> Dict[str, float]:
    if half_life_days <= 0:
        raise ValueError("recency half-life must be positive")
    if not timestamps or len(timestamps) != len(ratings):
        raise ValueError("path timestamps and ratings must be nonempty and aligned")
    mean_age_days = sum(max(0.0, (packet_time - value) / SECONDS_PER_DAY) for value in timestamps) / len(timestamps)
    recency = 2.0 ** (-mean_age_days / half_life_days)
    rating_strength = sum(max(0.0, min(1.0, (value - minimum_rating + 1.0) / 2.0)) for value in ratings) / len(ratings)
    hub_penalty = 1.0 / math.sqrt(math.log2(2 + anchor_degree) * math.log2(2 + bridge_degree))
    return {
        "score": recency * rating_strength * hub_penalty,
        "mean_age_days": mean_age_days,
        "recency": recency,
        "rating_strength": rating_strength,
        "hub_penalty": hub_penalty,
    }


def build_filtered_three_hop_ledger(
    histories: Mapping[str, Sequence[ScoredEvent]],
    packets: Sequence[Mapping[str, Any]],
    *,
    minimum_rating: float,
    half_life_days: float,
    diverse_paths_per_origin: int,
    peers1_per_anchor: int,
    bridge_items_per_peer1: int,
    peers2_per_bridge: int,
    endpoint_items_per_peer2: int,
    max_witnesses: int,
) -> Dict[str, Dict[str, Any]]:
    caps = (diverse_paths_per_origin, peers1_per_anchor, bridge_items_per_peer1, peers2_per_bridge, endpoint_items_per_peer2, max_witnesses)
    if min(caps) < 1:
        raise ValueError("all filtered 3-hop route caps must be positive")
    item_events: DefaultDict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for user_id, events in histories.items():
        for timestamp, item_id, rating, _, _ in events:
            if rating >= minimum_rating:
                item_events[item_id].append((timestamp, user_id, rating))
    for values in item_events.values():
        values.sort()

    paths_by_endpoint: DefaultDict[str, DefaultDict[str, list[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for packet in packets:
        source_user = str(packet["source_user_id"])
        packet_time = int(packet["packet_timestamp"])
        if source_user not in histories:
            raise ValueError(f"source user missing from sampled histories: {source_user}")
        source_seen = {event[1] for event in history_before(histories[source_user], packet_time)}
        source_seen.update(str(value) for value in packet["anchor_item_ids"])
        for anchor_item in (str(value) for value in packet["anchor_item_ids"]):
            source_anchor = latest_anchor_event(histories[source_user], anchor_item, packet_time)
            if source_anchor is None or source_anchor[2] < minimum_rating:
                continue
            anchor_degree = bisect.bisect_left(item_events.get(anchor_item, ()), (packet_time, "", float("-inf")))
            for peer1, anchor_time, anchor_rating in recent_positive_peers(
                item_events, anchor_item, packet_time, excluded_users={source_user}, limit=peers1_per_anchor
            ):
                for bridge_item, bridge1_time, bridge1_rating in recent_positive_items(
                    histories[peer1], packet_time, excluded_items=source_seen | {anchor_item}, minimum_rating=minimum_rating, limit=bridge_items_per_peer1
                ):
                    bridge_degree = bisect.bisect_left(item_events.get(bridge_item, ()), (packet_time, "", float("-inf")))
                    for peer2, bridge2_time, bridge2_rating in recent_positive_peers(
                        item_events, bridge_item, packet_time, excluded_users={source_user, peer1}, limit=peers2_per_bridge
                    ):
                        for endpoint_item, endpoint_time, endpoint_rating in recent_positive_items(
                            histories[peer2], packet_time, excluded_items=source_seen | {anchor_item, bridge_item}, minimum_rating=minimum_rating, limit=endpoint_items_per_peer2
                        ):
                            quality = path_quality(
                                packet_time=packet_time,
                                timestamps=(source_anchor[0], anchor_time, bridge1_time, bridge2_time, endpoint_time),
                                ratings=(source_anchor[2], anchor_rating, bridge1_rating, bridge2_rating, endpoint_rating),
                                anchor_degree=anchor_degree,
                                bridge_degree=bridge_degree,
                                minimum_rating=minimum_rating,
                                half_life_days=half_life_days,
                            )
                            paths_by_endpoint[endpoint_item][source_user].append(
                                {
                                    "origin_user": source_user,
                                    "packet_timestamp": packet_time,
                                    "anchor_item_id": anchor_item,
                                    "peer1_user": peer1,
                                    "bridge_item_id": bridge_item,
                                    "peer2_user": peer2,
                                    "endpoint_item_id": endpoint_item,
                                    "anchor_degree": anchor_degree,
                                    "bridge_degree": bridge_degree,
                                    **quality,
                                }
                            )

    ledger: Dict[str, Dict[str, Any]] = {}
    for endpoint_item, origin_paths in paths_by_endpoint.items():
        origin_rows: Dict[str, Dict[str, Any]] = {}
        for origin_user, paths in origin_paths.items():
            selected: list[Dict[str, Any]] = []
            seen_bridges: set[str] = set()
            for path in sorted(paths, key=lambda row: (-float(row["score"]), str(row["bridge_item_id"]), str(row["peer2_user"]))):
                bridge = str(path["bridge_item_id"])
                if bridge in seen_bridges:
                    continue
                seen_bridges.add(bridge)
                selected.append(path)
                if len(selected) >= diverse_paths_per_origin:
                    break
            if selected:
                origin_rows[origin_user] = {
                    "quality": sum(float(path["score"]) for path in selected),
                    "positive_path_count": len(paths),
                    "diverse_path_count": len(selected),
                    "paths": selected,
                }
        if not origin_rows:
            continue
        ranked_origins = sorted(origin_rows, key=lambda user_id: (-float(origin_rows[user_id]["quality"]), user_id))
        witnesses: list[Dict[str, Any]] = []
        for origin in ranked_origins:
            witnesses.extend(origin_rows[origin]["paths"])
        ledger[endpoint_item] = {
            "endpoint_item_id": endpoint_item,
            "origin_users": set(ranked_origins),
            "origin_users_by_quality": ranked_origins,
            "origin_quality_scores": {origin: origin_rows[origin]["quality"] for origin in ranked_origins},
            "origin_positive_path_counts": {origin: origin_rows[origin]["positive_path_count"] for origin in ranked_origins},
            "origin_diverse_path_counts": {origin: origin_rows[origin]["diverse_path_count"] for origin in ranked_origins},
            "n_paths": sum(row["positive_path_count"] for row in origin_rows.values()),
            "witness_paths": sorted(witnesses, key=lambda row: (-float(row["score"]), str(row["origin_user"])))[:max_witnesses],
        }
    return ledger


def serializable_filtered_ledger(ledger: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item_id in sorted(ledger):
        entry = ledger[item_id]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "endpoint_item_id": item_id,
                "n_independent_origins": len(entry["origin_users_by_quality"]),
                "origin_users_by_quality": entry["origin_users_by_quality"],
                "origin_quality_scores": entry["origin_quality_scores"],
                "origin_positive_path_counts": entry["origin_positive_path_counts"],
                "origin_diverse_path_counts": entry["origin_diverse_path_counts"],
                "n_paths": entry["n_paths"],
                "witness_paths": entry["witness_paths"],
            }
        )
    return rows


def load_histories(config: Mapping[str, Any]) -> Dict[str, list[ScoredEvent]]:
    dataset, p1 = config["dataset"], config["p1"]
    p0 = json.loads(project_path(dataset["p0_audit"]).read_text(encoding="utf-8"))
    train_cutoff = int(p0["temporal_split"]["train_cutoff"])
    modulus, remainder = int(p1["pilot_user_hash_modulus"]), int(p1["pilot_user_hash_remainder"])
    histories: DefaultDict[str, list[ScoredEvent]] = defaultdict(list)
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
            if timestamp < train_cutoff:
                histories[user_id].append((timestamp, item_id, rating, row.get("review/summary") or "", row.get("review/text") or ""))
    for events in histories.values():
        events.sort()
    return dict(histories)


def protocol(config: Mapping[str, Any]) -> Dict[str, Any]:
    p4 = config["p4"]
    return {
        "minimum_rating": float(p4["positive_rating_min"]),
        "half_life_days": float(p4["recency_half_life_days"]),
        "diverse_paths_per_origin": int(p4["diverse_paths_per_origin"]),
        "peers1_per_anchor": int(p4["peers1_per_anchor"]),
        "bridge_items_per_peer1": int(p4["bridge_items_per_peer1"]),
        "peers2_per_bridge": int(p4["peers2_per_bridge"]),
        "endpoint_items_per_peer2": int(p4["endpoint_items_per_peer2"]),
        "max_witnesses": int(p4["max_witness_paths_per_item"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filtered 3-hop structural preflight")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p4_filtered_three_hop.yaml")
    parser.add_argument("--smoke-sources", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    dataset, p4 = config["dataset"], config["p4"]
    packets_path = project_path(dataset["p1_packet_sources"])
    packets = [json.loads(line) for line in packets_path.read_text(encoding="utf-8").splitlines() if line]
    histories, frozen = load_histories(config), protocol(config)
    started = time.monotonic()
    smoke_count = int(p4["smoke_sources"])
    if args.smoke_sources is not None:
        if args.smoke_sources != smoke_count or not 20 <= args.smoke_sources <= 30:
            raise ValueError("filtered structural smoke must use the preregistered 20-30 sources")
        ledger = build_filtered_three_hop_ledger(histories, packets[:smoke_count], **frozen)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(config_path),
            "packet_sources": [str(row["source_user_id"]) for row in packets[:smoke_count]],
            "protocol": frozen,
            "endpoint_items": len(ledger),
            "paths": sum(int(row["n_paths"]) for row in ledger.values()),
            "labels_opened": False,
            "elapsed_seconds": time.monotonic() - started,
            "decision": "pass",
        }
        output = project_path(p4["structural_smoke_manifest"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    smoke_path = project_path(p4["structural_smoke_manifest"])
    if not smoke_path.exists():
        raise RuntimeError("filtered full preflight blocked: run structural smoke first")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("config_sha256") != sha256_file(config_path) or smoke.get("protocol") != frozen:
        raise RuntimeError("filtered full preflight blocked: smoke contract mismatch")
    ledger_path, manifest_path = project_path(p4["ledger"]), project_path(p4["preflight_manifest"])
    if (ledger_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError("filtered P4 outputs exist; use --force only for derived artifacts")
    ledger = build_filtered_three_hop_ledger(histories, packets, **frozen)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    records, output_sha = jsonl_write(ledger_path, serializable_filtered_ledger(ledger))
    targets_path = project_path(dataset["p1_eval_events"])
    targets = [json.loads(line) for line in targets_path.read_text(encoding="utf-8").splitlines() if line]
    fixed_coverage = coverage(ledger, targets, [1, 2, 3])
    support2 = float(fixed_coverage["2"]["gold_coverage"])
    minimum = float(p4["minimum_support2_coverage_to_run"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "packets": {"path": str(packets_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(packets_path)},
            "targets": {"path": str(targets_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(targets_path)},
            "smoke": {"path": str(smoke_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(smoke_path)},
        },
        "protocol": {**frozen, "candidate_blind": True, "labels_opened_after_ledger_serialization": True},
        "output": {"path": str(ledger_path.relative_to(PROJECT_ROOT)), "records": records, "sha256": output_sha},
        "coverage": {"fixed_evaluation_targets": fixed_coverage},
        "admission": {"minimum_support2_coverage": minimum, "decision": "pass" if support2 >= minimum else "hard_stop"},
        "runtime": {"elapsed_seconds": time.monotonic() - started, "sampled_users": len(histories), "packet_sources": len(packets)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"endpoints": records, "coverage": fixed_coverage, "admission": manifest["admission"], "elapsed_seconds": manifest["runtime"]["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
