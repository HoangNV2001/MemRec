"""P1: candidate-blind structural preflight for a 1,000-call temporal pilot.

P1 creates no LLM request.  It samples users using only a stable hash of their
identifier, then uses only training events to select 560 potential semantic
packets and construct causal ``source -> anchor -> peer -> endpoint`` routes.
Validation labels are read only after that ledger is closed, for coverage
reporting and for a fixed 100-event future evaluation cohort.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Mapping, Sequence

from src.temporal_books.common import (
    PROJECT_ROOT,
    jsonl_write,
    load_yaml,
    project_path,
    set_csv_field_limit,
    sha256_file,
    stable_bucket,
    stable_order,
)


SCHEMA_VERSION = 1
Event = tuple[int, str, str, str]


def truncate(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: max(0, limit - 1)] + "…"


def history_before(events: Sequence[Event], timestamp: int) -> Sequence[Event]:
    """Return only events at timestamps strictly less than ``timestamp``."""
    return events[: bisect.bisect_left(events, (timestamp, "", "", ""))]


def select_sources(histories: Mapping[str, Sequence[Event]], *, minimum_events: int, count: int) -> list[str]:
    eligible = [user_id for user_id, events in histories.items() if len(events) >= minimum_events]
    return sorted(eligible, key=lambda user_id: (stable_order(f"source\0{user_id}"), user_id))[:count]


def build_endpoint_ledger(
    histories: Mapping[str, Sequence[Event]],
    source_users: Sequence[str],
    *,
    max_anchors: int,
    max_peers: int,
    max_remote_items: int,
    max_witnesses: int,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Build a strict-past, candidate-blind item endpoint ledger.

    A source user emits one packet at their most recent training event.  No
    route may read events at that timestamp or later, including other reviews
    posted on the same date.  Each origin is one future LLM request, so support
    counts independent semantic packets rather than duplicate graph paths.
    """
    if min(max_anchors, max_peers, max_remote_items, max_witnesses) < 1:
        raise ValueError("all route caps must be positive")
    item_events: DefaultDict[str, list[tuple[int, str]]] = defaultdict(list)
    for user_id, events in histories.items():
        for timestamp, item_id, _, _ in events:
            item_events[item_id].append((timestamp, user_id))
    for events in item_events.values():
        events.sort()

    ledger: Dict[str, Dict[str, Any]] = {}
    packet_meta: Dict[str, Dict[str, Any]] = {}
    for source_user in source_users:
        source_events = histories[source_user]
        packet_time, packet_item, _, _ = source_events[-1]
        past_source_events = history_before(source_events, packet_time)
        source_seen_items = {event[1] for event in past_source_events}
        source_seen_items.add(packet_item)

        anchors = [packet_item]
        for _, item_id, _, _ in reversed(past_source_events):
            if item_id not in anchors:
                anchors.append(item_id)
            if len(anchors) >= max_anchors:
                break
        packet_meta[source_user] = {
            "source_user_id": source_user,
            "packet_timestamp": packet_time,
            "packet_item_id": packet_item,
            "anchor_item_ids": anchors,
        }

        for anchor_item in anchors:
            candidates = item_events.get(anchor_item, [])
            peer_limit = bisect.bisect_left(candidates, (packet_time, ""))
            peers: list[str] = []
            seen_peers = {source_user}
            for _, peer_user in reversed(candidates[:peer_limit]):
                if peer_user not in seen_peers:
                    seen_peers.add(peer_user)
                    peers.append(peer_user)
                if len(peers) >= max_peers:
                    break
            for peer_user in peers:
                remote_items: list[str] = []
                seen_remote = set(source_seen_items)
                seen_remote.add(anchor_item)
                for _, endpoint_item, _, _ in reversed(history_before(histories[peer_user], packet_time)):
                    if endpoint_item not in seen_remote:
                        seen_remote.add(endpoint_item)
                        remote_items.append(endpoint_item)
                    if len(remote_items) >= max_remote_items:
                        break
                for endpoint_item in remote_items:
                    entry = ledger.setdefault(
                        endpoint_item,
                        {
                            "endpoint_item_id": endpoint_item,
                            "origin_users": set(),
                            "n_paths": 0,
                            "witness_paths": [],
                        },
                    )
                    entry["origin_users"].add(source_user)
                    entry["n_paths"] += 1
                    if len(entry["witness_paths"]) < max_witnesses:
                        entry["witness_paths"].append(
                            {
                                "origin_user": source_user,
                                "packet_timestamp": packet_time,
                                "anchor_item_id": anchor_item,
                                "peer_user": peer_user,
                                "endpoint_item_id": endpoint_item,
                            }
                        )
    return ledger, packet_meta


def serializable_ledger(ledger: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item_id in sorted(ledger):
        entry = ledger[item_id]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "endpoint_item_id": item_id,
                "n_independent_origins": len(entry["origin_users"]),
                "origin_users": sorted(entry["origin_users"]),
                "n_paths": entry["n_paths"],
                "witness_paths": entry["witness_paths"],
            }
        )
    return rows


def coverage(ledger: Mapping[str, Mapping[str, Any]], targets: Sequence[Mapping[str, Any]], thresholds: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for threshold in sorted({int(value) for value in thresholds}):
        if threshold < 1:
            raise ValueError("support thresholds must be positive")
        covered = sum(
            len(ledger.get(str(target["gold_item_id"]), {}).get("origin_users", ())) >= threshold for target in targets
        )
        values[str(threshold)] = {
            "targets": len(targets),
            "covered_gold_targets": covered,
            "gold_coverage": covered / len(targets) if targets else 0.0,
        }
    return values


def validate_llm_budget(budget: Mapping[str, Any]) -> Dict[str, int]:
    semantic = int(budget["semantic_packet_requests"])
    events = int(budget["evaluation_events"])
    stage_r = int(budget["shared_stage_r_requests_per_event"])
    rerank_arms = int(budget["rerank_arms"])
    retries = int(budget["retry_reserve_requests"])
    total = semantic + events * (stage_r + rerank_arms) + retries
    maximum = int(budget["max_total_requests"])
    if total > maximum:
        raise ValueError(f"LLM budget {total} exceeds configured maximum {maximum}")
    return {
        "semantic_packet_requests": semantic,
        "shared_stage_r_requests": events * stage_r,
        "rerank_requests": events * rerank_arms,
        "retry_reserve_requests": retries,
        "maximum_total_requests": maximum,
        "planned_total_requests": total,
    }


def packet_rows(
    histories: Mapping[str, Sequence[Event]],
    packet_meta: Mapping[str, Mapping[str, Any]],
    *,
    max_history_reviews: int,
    summary_limit: int,
    text_limit: int,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for source_user in sorted(packet_meta):
        packet = dict(packet_meta[source_user])
        packet_time = int(packet["packet_timestamp"])
        current = histories[source_user][-1]
        context = list(history_before(histories[source_user], packet_time))[-max_history_reviews + 1 :] + [current]
        packet["semantic_history"] = [
            {
                "timestamp": timestamp,
                "item_id": item_id,
                "review_summary": truncate(summary, summary_limit),
                "review_text": truncate(text, text_limit),
            }
            for timestamp, item_id, summary, text in context
        ]
        rows.append(packet)
    return rows


def first_novel_validation_targets(
    histories: Mapping[str, Sequence[Event]],
    validation_events: Mapping[str, Sequence[Event]],
    minimum_events: int,
    available_items: set[str],
) -> list[Dict[str, Any]]:
    targets: list[Dict[str, Any]] = []
    for user_id in sorted(histories):
        train_history = histories[user_id]
        if len(train_history) < minimum_events:
            continue
        seen_items = {item_id for _, item_id, _, _ in train_history}
        for timestamp, item_id, _, _ in sorted(validation_events.get(user_id, ())):
            if item_id not in seen_items and item_id in available_items:
                targets.append({"user_id": user_id, "timestamp": timestamp, "gold_item_id": item_id})
                break
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 structural preflight for temporal Amazon Books pilot")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/pilot.yaml")
    parser.add_argument("--force", action="store_true", help="replace derived P1 artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    dataset = config["dataset"]
    p1 = config["p1"]
    budget = validate_llm_budget(config["llm_budget"])
    if int(p1["semantic_packet_sources"]) != budget["semantic_packet_requests"]:
        raise ValueError("P1 source count must equal the reserved semantic-packet request budget")
    if int(p1["fixed_evaluation_events"]) != int(config["llm_budget"]["evaluation_events"]):
        raise ValueError("P1 fixed evaluation cohort must equal the LLM evaluation-event budget")
    reviews_path = project_path(dataset["reviews_file"])
    p0_path = project_path(dataset["p0_audit"])
    ledger_path = project_path(dataset["p1_ledger"])
    sources_path = project_path(dataset["p1_packet_sources"])
    eval_path = project_path(dataset["p1_eval_events"])
    manifest_path = project_path(dataset["p1_manifest"])
    for path in (config_path, reviews_path, p0_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if any(path.exists() for path in (ledger_path, sources_path, eval_path, manifest_path)) and not args.force:
        raise FileExistsError("P1 output exists; use --force only to replace derived offline artifacts")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    if not p0["decision"]["strict_past_timestamp_batch_replay_admissible"]:
        raise ValueError("P0 did not admit strict-past timestamp-batch replay")
    train_cutoff = int(p0["temporal_split"]["train_cutoff"])
    validation_cutoff = int(p0["temporal_split"]["validation_cutoff"])
    modulus = int(p1["pilot_user_hash_modulus"])
    remainder = int(p1["pilot_user_hash_remainder"])
    if not 0 <= remainder < modulus:
        raise ValueError("pilot_user_hash_remainder must be in [0, modulus)")

    histories: DefaultDict[str, list[Event]] = defaultdict(list)
    validation_events: DefaultDict[str, list[Event]] = defaultdict(list)
    globally_available_train_items: set[str] = set()
    sampled_users_seen: set[str] = set()
    set_csv_field_limit()
    with reviews_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_id = (row.get("User_id") or "").strip()
            item_id = (row.get("Id") or "").strip()
            if not user_id or not item_id:
                continue
            try:
                timestamp = int(row["review/time"])
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp < train_cutoff:
                globally_available_train_items.add(item_id)
            if stable_bucket(user_id, modulus) != remainder:
                continue
            sampled_users_seen.add(user_id)
            event = (timestamp, item_id, row.get("review/summary") or "", row.get("review/text") or "")
            if timestamp < train_cutoff:
                histories[user_id].append(event)
            elif timestamp < validation_cutoff:
                validation_events[user_id].append(event)

    for events in histories.values():
        events.sort()
    for events in validation_events.values():
        events.sort()
    minimum_events = int(p1["min_train_events_per_user"])
    selected_sources = select_sources(
        histories,
        minimum_events=minimum_events,
        count=int(p1["semantic_packet_sources"]),
    )
    ledger, packet_meta = build_endpoint_ledger(
        histories,
        selected_sources,
        max_anchors=int(p1["max_anchor_items_per_packet"]),
        max_peers=int(p1["max_peer_users_per_anchor"]),
        max_remote_items=int(p1["max_remote_items_per_peer"]),
        max_witnesses=int(p1["max_witness_paths_per_item"]),
    )
    all_targets = first_novel_validation_targets(
        histories,
        validation_events,
        minimum_events,
        globally_available_train_items,
    )
    fixed_count = int(p1["fixed_evaluation_events"])
    fixed_targets = sorted(
        all_targets,
        key=lambda row: (stable_order(f"eval\0{row['user_id']}\0{row['timestamp']}\0{row['gold_item_id']}"), row["user_id"]),
    )[:fixed_count]
    thresholds = [int(value) for value in p1["support_thresholds"]]
    all_coverage = coverage(ledger, all_targets, thresholds)
    fixed_coverage = coverage(ledger, fixed_targets, thresholds)
    ledger_rows = serializable_ledger(ledger)
    source_rows = packet_rows(
        histories,
        packet_meta,
        max_history_reviews=int(p1["packet_history_reviews"]),
        summary_limit=int(p1["packet_summary_char_cap"]),
        text_limit=int(p1["packet_text_char_cap"]),
    )
    output_dir = ledger_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_count, ledger_hash = jsonl_write(ledger_path, ledger_rows)
    source_count, source_hash = jsonl_write(sources_path, source_rows)
    eval_count, eval_hash = jsonl_write(eval_path, fixed_targets)
    admission = p1["admission"]
    support1 = fixed_coverage.get("1", {"gold_coverage": 0.0})["gold_coverage"]
    support2 = fixed_coverage.get("2", {"gold_coverage": 0.0})["gold_coverage"]
    p1_pass = (
        source_count >= int(admission["min_packet_sources"])
        and support1 >= float(admission["min_support1_gold_coverage"])
        and support2 >= float(admission["min_support2_gold_coverage"])
        and eval_count == fixed_count
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "reviews": p0["inputs"]["reviews"],
            "p0_audit": {"path": str(p0_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(p0_path)},
        },
        "protocol": {
            "time_semantics": "strict past: every packet at t uses only events with timestamp < t",
            "pilot_user_sampling": {"hash": "blake2b-64", "modulus": modulus, "remainder": remainder},
            "train_cutoff": train_cutoff,
            "validation_cutoff": validation_cutoff,
            "source_selection": "stable hash among users with sufficient training history; never reads validation/test labels",
            "route": "source user -> anchor item -> peer user -> endpoint item",
            "route_caps": {
                "anchors": int(p1["max_anchor_items_per_packet"]),
                "peers_per_anchor": int(p1["max_peer_users_per_anchor"]),
                "remote_items_per_peer": int(p1["max_remote_items_per_peer"]),
                "witness_paths_per_endpoint": int(p1["max_witness_paths_per_item"]),
            },
            "candidate_blind_ledger": True,
            "llm_requests_made": 0,
            "conditional_llm_budget": budget,
        },
        "cohort": {
            "hash_sampled_users_seen": len(sampled_users_seen),
            "users_with_train_history": len(histories),
            "users_with_minimum_train_events": sum(len(events) >= minimum_events for events in histories.values()),
            "selected_packet_sources": source_count,
            "all_novel_validation_targets": len(all_targets),
            "fixed_evaluation_targets": eval_count,
            "candidate_pool_items_available_before_train_cutoff": len(globally_available_train_items),
        },
        "outputs": {
            "ledger": {"path": str(ledger_path.relative_to(PROJECT_ROOT)), "records": ledger_count, "sha256": ledger_hash},
            "packet_sources": {"path": str(sources_path.relative_to(PROJECT_ROOT)), "records": source_count, "sha256": source_hash},
            "evaluation_events": {"path": str(eval_path.relative_to(PROJECT_ROOT)), "records": eval_count, "sha256": eval_hash},
        },
        "coverage": {"all_novel_validation_targets": all_coverage, "fixed_evaluation_targets": fixed_coverage},
        "admission": {
            "criteria": admission,
            "decision": "pass" if p1_pass else "hard_stop",
            "reason": "P2 needs enough source packets and source-only route coverage before any LLM request.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"P1 sampled users={len(sampled_users_seen):,}; packet sources={source_count:,}; endpoints={ledger_count:,}")
    for threshold in thresholds:
        audit = fixed_coverage[str(threshold)]
        print(f"  fixed eval support>={threshold}: {audit['covered_gold_targets']}/{audit['targets']} ({audit['gold_coverage']:.3%})")
    print(f"  decision={manifest['admission']['decision']}; LLM requests made=0")
    print(f"  wrote {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
