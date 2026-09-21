"""Offline, smoke-first structural preflight for bounded temporal 3-hop routes.

The route is:

    source user -> anchor item -> peer-1 user -> bridge item
                -> peer-2 user -> endpoint item

All graph reads are strictly before the source packet timestamp.  The ledger is
fully source-only and candidate-blind; evaluation labels are opened only after
the canonical ledger has been constructed.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
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
    stable_bucket,
)
from src.temporal_books.p1_preflight import Event, coverage, history_before


SCHEMA_VERSION = 1


def recent_peer_users(
    item_events: Mapping[str, Sequence[tuple[int, str]]],
    item_id: str,
    packet_time: int,
    *,
    excluded_users: set[str],
    limit: int,
) -> list[str]:
    candidates = item_events.get(item_id, ())
    end = bisect.bisect_left(candidates, (packet_time, ""))
    values: list[str] = []
    seen = set(excluded_users)
    for _, user_id in reversed(candidates[:end]):
        if user_id not in seen:
            seen.add(user_id)
            values.append(user_id)
        if len(values) >= limit:
            break
    return values


def recent_unique_items(
    events: Sequence[Event],
    packet_time: int,
    *,
    excluded_items: set[str],
    limit: int,
) -> list[str]:
    values: list[str] = []
    seen = set(excluded_items)
    for _, item_id, _, _ in reversed(history_before(events, packet_time)):
        if item_id not in seen:
            seen.add(item_id)
            values.append(item_id)
        if len(values) >= limit:
            break
    return values


def build_three_hop_ledger(
    histories: Mapping[str, Sequence[Event]],
    packets: Sequence[Mapping[str, Any]],
    *,
    peers1_per_anchor: int,
    bridge_items_per_peer1: int,
    peers2_per_bridge: int,
    endpoint_items_per_peer2: int,
    max_witnesses: int,
) -> Dict[str, Dict[str, Any]]:
    """Build a bounded 3-hop ledger without reading candidates or labels."""
    caps = (peers1_per_anchor, bridge_items_per_peer1, peers2_per_bridge, endpoint_items_per_peer2, max_witnesses)
    if min(caps) < 1:
        raise ValueError("all 3-hop route caps must be positive")
    item_events: DefaultDict[str, list[tuple[int, str]]] = defaultdict(list)
    for user_id, events in histories.items():
        for timestamp, item_id, _, _ in events:
            item_events[item_id].append((timestamp, user_id))
    for values in item_events.values():
        values.sort()

    ledger: Dict[str, Dict[str, Any]] = {}
    for packet in packets:
        source_user = str(packet["source_user_id"])
        packet_time = int(packet["packet_timestamp"])
        anchors = [str(item_id) for item_id in packet["anchor_item_ids"]]
        if source_user not in histories:
            raise ValueError(f"source user missing from sampled histories: {source_user}")
        source_seen = {event[1] for event in history_before(histories[source_user], packet_time)}
        source_seen.update(anchors)
        for anchor_item in anchors:
            peers1 = recent_peer_users(
                item_events,
                anchor_item,
                packet_time,
                excluded_users={source_user},
                limit=peers1_per_anchor,
            )
            for peer1 in peers1:
                bridges = recent_unique_items(
                    histories[peer1],
                    packet_time,
                    excluded_items=source_seen | {anchor_item},
                    limit=bridge_items_per_peer1,
                )
                for bridge_item in bridges:
                    peers2 = recent_peer_users(
                        item_events,
                        bridge_item,
                        packet_time,
                        excluded_users={source_user, peer1},
                        limit=peers2_per_bridge,
                    )
                    for peer2 in peers2:
                        endpoints = recent_unique_items(
                            histories[peer2],
                            packet_time,
                            excluded_items=source_seen | {anchor_item, bridge_item},
                            limit=endpoint_items_per_peer2,
                        )
                        for endpoint_item in endpoints:
                            entry = ledger.setdefault(
                                endpoint_item,
                                {
                                    "endpoint_item_id": endpoint_item,
                                    "origin_users": set(),
                                    "origin_path_counts": defaultdict(int),
                                    "n_paths": 0,
                                    "witness_paths": [],
                                },
                            )
                            entry["origin_users"].add(source_user)
                            entry["origin_path_counts"][source_user] += 1
                            entry["n_paths"] += 1
                            if len(entry["witness_paths"]) < max_witnesses:
                                entry["witness_paths"].append(
                                    {
                                        "origin_user": source_user,
                                        "packet_timestamp": packet_time,
                                        "anchor_item_id": anchor_item,
                                        "peer1_user": peer1,
                                        "bridge_item_id": bridge_item,
                                        "peer2_user": peer2,
                                        "endpoint_item_id": endpoint_item,
                                    }
                                )
    return ledger


def serializable_three_hop_ledger(ledger: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item_id in sorted(ledger):
        entry = ledger[item_id]
        counts = {str(key): int(value) for key, value in entry["origin_path_counts"].items()}
        ranked_origins = sorted(entry["origin_users"], key=lambda user_id: (-counts[user_id], user_id))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "endpoint_item_id": item_id,
                "n_independent_origins": len(ranked_origins),
                "origin_users_by_path_support": ranked_origins,
                "origin_path_counts": counts,
                "n_paths": int(entry["n_paths"]),
                "witness_paths": entry["witness_paths"],
            }
        )
    return rows


def load_sampled_histories(config: Mapping[str, Any]) -> Dict[str, list[Event]]:
    dataset = config["dataset"]
    p1 = config["p1"]
    p0 = json.loads(project_path(dataset["p0_audit"]).read_text(encoding="utf-8"))
    train_cutoff = int(p0["temporal_split"]["train_cutoff"])
    modulus = int(p1["pilot_user_hash_modulus"])
    remainder = int(p1["pilot_user_hash_remainder"])
    histories: DefaultDict[str, list[Event]] = defaultdict(list)
    set_csv_field_limit()
    with project_path(dataset["reviews_file"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = (row.get("User_id") or "").strip()
            item_id = (row.get("Id") or "").strip()
            if not user_id or not item_id or stable_bucket(user_id, modulus) != remainder:
                continue
            try:
                timestamp = int(row["review/time"])
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp < train_cutoff:
                histories[user_id].append((timestamp, item_id, row.get("review/summary") or "", row.get("review/text") or ""))
    for events in histories.values():
        events.sort()
    return dict(histories)


def route_caps(p3: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "peers1_per_anchor": int(p3["peers1_per_anchor"]),
        "bridge_items_per_peer1": int(p3["bridge_items_per_peer1"]),
        "peers2_per_bridge": int(p3["peers2_per_bridge"]),
        "endpoint_items_per_peer2": int(p3["endpoint_items_per_peer2"]),
        "max_witnesses": int(p3["max_witness_paths_per_item"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-first bounded 3-hop temporal route preflight")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p3_selfhost.yaml")
    parser.add_argument("--smoke-sources", type=int, help="build only a source-only smoke subset; never opens eval labels")
    parser.add_argument("--force", action="store_true", help="replace canonical derived P3 outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    dataset = config["dataset"]
    p3 = config["p3"]
    packets_path = project_path(dataset["p1_packet_sources"])
    packets = [json.loads(line) for line in packets_path.read_text(encoding="utf-8").splitlines() if line]
    histories = load_sampled_histories(config)
    caps = route_caps(p3)
    started = time.monotonic()

    if args.smoke_sources is not None:
        if not 20 <= args.smoke_sources <= 30:
            raise ValueError("structural smoke must use 20-30 packet sources")
        selected = packets[: args.smoke_sources]
        ledger = build_three_hop_ledger(histories, selected, **caps)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(config_path),
            "packet_sources": [str(packet["source_user_id"]) for packet in selected],
            "route_caps": caps,
            "endpoint_items": len(ledger),
            "paths": sum(int(entry["n_paths"]) for entry in ledger.values()),
            "elapsed_seconds": time.monotonic() - started,
            "labels_opened": False,
            "decision": "pass",
        }
        smoke_path = project_path(p3["structural_smoke_manifest"])
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    smoke_path = project_path(p3["structural_smoke_manifest"])
    if not smoke_path.exists():
        raise RuntimeError("canonical 3-hop preflight is blocked: run --smoke-sources 24 first")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("config_sha256") != sha256_file(config_path) or smoke.get("route_caps") != caps:
        raise RuntimeError("canonical 3-hop preflight is blocked: smoke config/caps do not match")
    ledger_path = project_path(p3["ledger"])
    manifest_path = project_path(p3["preflight_manifest"])
    if (ledger_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError("P3 output exists; use --force only to replace derived offline artifacts")
    ledger = build_three_hop_ledger(histories, packets, **caps)
    rows = serializable_three_hop_ledger(ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row_count, ledger_sha = jsonl_write(ledger_path, rows)

    # Evaluation labels are opened only after the candidate-blind ledger is
    # complete and serialized.
    targets_path = project_path(dataset["p1_eval_events"])
    targets = [json.loads(line) for line in targets_path.read_text(encoding="utf-8").splitlines() if line]
    fixed_coverage = coverage(ledger, targets, [1, 2, 3])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "p1_packets": {"path": str(packets_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(packets_path)},
            "p1_targets": {"path": str(targets_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(targets_path)},
            "structural_smoke": {"path": str(smoke_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(smoke_path)},
        },
        "protocol": {
            "route": "source -> anchor item -> peer1 -> bridge item -> peer2 -> endpoint item",
            "time_semantics": "all graph reads strictly before each source packet timestamp",
            "candidate_blind_ledger": True,
            "route_caps": caps,
            "origin_order": "descending source-to-endpoint path count, then source user ID",
            "llm_requests_made": 0,
        },
        "output": {"path": str(ledger_path.relative_to(PROJECT_ROOT)), "records": row_count, "sha256": ledger_sha},
        "coverage": {"fixed_evaluation_targets": fixed_coverage},
        "runtime": {"elapsed_seconds": time.monotonic() - started, "sampled_users": len(histories), "packet_sources": len(packets)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"endpoints": row_count, "coverage": fixed_coverage, "elapsed_seconds": manifest["runtime"]["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
