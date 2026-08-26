"""P1 — candidate-blind structural feasibility audit for item-side buffering.

This does not call an LLM and does not alter memories.  It constructs a full
item endpoint ledger from *source-split users only*.  Each source user's most
recent distinct pre-heldout items can be anchors ``x``.  A packet can reach
endpoint item ``y`` only through a source-only witness path
``u -> x -> v -> y``.  Candidate lists and gold labels are consulted only after
the ledger is final, to report coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def read_histories(path: Path) -> Dict[int, List[int]]:
    events: MutableMapping[int, List[tuple[float, int, int]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"user_id", "item_id", "timestamp"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"{path} must contain TSV columns {sorted(required)}")
        for order, row in enumerate(reader):
            events[int(row["user_id"])].append((float(row["timestamp"]), order, int(row["item_id"])))
    histories: Dict[int, List[int]] = {}
    for user_id, values in events.items():
        values.sort(key=lambda value: (value[0], value[1]))
        if len(values) >= 3:
            histories[user_id] = [item for _, _, item in values[:-2]]
    return histories


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_endpoint_ledger(
    source_histories: Mapping[int, Sequence[int]], *, max_anchors: int, max_peers: int, max_remote_items: int, max_witnesses: int
) -> Dict[int, Dict[str, Any]]:
    """Build candidate-blind ``u-x-v-y`` endpoint support from source users."""
    if max_anchors < 1 or max_peers < 1 or max_remote_items < 1 or max_witnesses < 1:
        raise ValueError("route caps must be positive")
    item_users: MutableMapping[int, List[int]] = defaultdict(list)
    for user_id, history in source_histories.items():
        for item_id in set(history):
            item_users[int(item_id)].append(int(user_id))
    for users in item_users.values():
        users.sort()
    ledger: Dict[int, Dict[str, Any]] = {}
    for origin_user in sorted(source_histories):
        history = list(source_histories[origin_user])
        if not history:
            continue
        anchors = []
        seen_anchors = set()
        for item_id in reversed(history):
            item_id = int(item_id)
            if item_id not in seen_anchors:
                seen_anchors.add(item_id)
                anchors.append(item_id)
            if len(anchors) >= max_anchors:
                break
        for anchor_item in anchors:
            peer_candidates = []
            anchor_degree = len(item_users[anchor_item])
            for peer_user in item_users[anchor_item]:
                if peer_user == origin_user:
                    continue
                peer_history = source_histories[peer_user]
                strength = 1.0 / math.sqrt(max(1, anchor_degree) * max(1, len(peer_history)))
                peer_candidates.append((strength, peer_user))
            peers = sorted(peer_candidates, key=lambda value: (-value[0], value[1]))[:max_peers]
            for strength, peer_user in peers:
                remote_items = []
                seen = {anchor_item}
                for item_id in reversed(source_histories[peer_user]):
                    item_id = int(item_id)
                    if item_id not in seen:
                        seen.add(item_id)
                        remote_items.append(item_id)
                    if len(remote_items) >= max_remote_items:
                        break
                for endpoint_item in remote_items:
                    entry = ledger.setdefault(
                        endpoint_item,
                        {
                            "endpoint_item_id": endpoint_item,
                            "origin_users": set(),
                            "n_paths": 0,
                            "max_path_strength": 0.0,
                            "witness_paths": [],
                        },
                    )
                    entry["origin_users"].add(origin_user)
                    entry["n_paths"] += 1
                    entry["max_path_strength"] = max(entry["max_path_strength"], strength)
                    if len(entry["witness_paths"]) < max_witnesses:
                        entry["witness_paths"].append(
                            {
                                "origin_user": origin_user,
                                "anchor_item": anchor_item,
                                "peer_user": peer_user,
                                "endpoint_item": endpoint_item,
                                "path_strength": strength,
                            }
                        )
    return ledger


def serializable_ledger(ledger: Mapping[int, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for endpoint_item in sorted(ledger):
        entry = ledger[endpoint_item]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "endpoint_item_id": int(endpoint_item),
                "n_independent_origins": len(entry["origin_users"]),
                "origin_users": sorted(entry["origin_users"]),
                "n_paths": int(entry["n_paths"]),
                "max_path_strength": float(entry["max_path_strength"]),
                "witness_paths": entry["witness_paths"],
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def coverage_audit(
    ledger: Mapping[int, Mapping[str, Any]], controls: Sequence[Mapping[str, Any]], *, support_thresholds: Sequence[int]
) -> Dict[str, Any]:
    """Post-hoc coverage only; it never affects the candidate-blind ledger."""
    thresholds = sorted({int(value) for value in support_thresholds})
    if not thresholds or thresholds[0] < 1:
        raise ValueError("support thresholds must be positive")
    values: Dict[str, Any] = {}
    for threshold in thresholds:
        n_candidates = 0
        n_covered_candidates = 0
        n_users_any = 0
        n_gold = 0
        n_covered_gold = 0
        for control in controls:
            ranking = control["ranking_context"]
            candidates = [int(item) for item in ranking["candidates"]]
            covered = [item for item in candidates if len(ledger.get(item, {}).get("origin_users", ())) >= threshold]
            n_candidates += len(candidates)
            n_covered_candidates += len(covered)
            n_users_any += int(bool(covered))
            n_gold += 1
            gold = int(ranking["gold_item_id"])
            n_covered_gold += int(gold in covered)
        values[str(threshold)] = {
            "candidate_slots": n_candidates,
            "covered_candidate_slots": n_covered_candidates,
            "candidate_slot_coverage": n_covered_candidates / n_candidates if n_candidates else 0.0,
            "users_with_any_covered_candidate": n_users_any,
            "user_coverage": n_users_any / len(controls) if controls else 0.0,
            "gold_items": n_gold,
            "covered_gold_items": n_covered_gold,
            "gold_coverage": n_covered_gold / n_gold if n_gold else 0.0,
        }
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-only item propagation route ledger")
    parser.add_argument("--config", default="configs/multihop/mh0_books.yaml")
    parser.add_argument("--force", action="store_true", help="replace derived P1 ledger/manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    section = config.get("propagation", {})
    p1 = section.get("p1", {})
    interaction_path = project_path(section["interaction_file"])
    splits_path = project_path(section["user_splits_file"])
    source_controls_path = project_path(section["mh0_control_train"])
    controls_path = project_path(section["mh0_control_val"])
    ledger_path = project_path(section["p1_ledger"])
    manifest_path = project_path(section["p1_manifest"])
    required = [config_path, interaction_path, splits_path, source_controls_path, controls_path]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("P1 required input(s) missing: " + ", ".join(missing))
    if (ledger_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError("P1 output exists; use --force only to replace derived offline artifacts")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]
    source_split = str(p1.get("source_split", "train"))
    split_source_users = {int(user_id) for user_id in splits[source_split]}
    source_users = {int(row["user_id"]) for row in read_jsonl(source_controls_path)}
    if not source_users <= split_source_users:
        raise ValueError("MH0 source controls include users outside the configured source split")
    histories = read_histories(interaction_path)
    source_histories = {user_id: histories[user_id] for user_id in source_users if user_id in histories}
    if source_users != set(source_histories):
        raise ValueError("some source users do not have a usable pre-heldout history")
    controls = list(read_jsonl(controls_path))
    eval_users = {int(row["user_id"]) for row in controls}
    if source_users & eval_users:
        raise ValueError("source and evaluation users must be disjoint")
    ledger = build_endpoint_ledger(
        source_histories,
        max_anchors=int(p1["max_anchor_items_per_source"]),
        max_peers=int(p1["max_peer_users_per_anchor"]),
        max_remote_items=int(p1["max_remote_items_per_peer"]),
        max_witnesses=int(p1["max_witness_paths_per_item"]),
    )
    rows = serializable_ledger(ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    count, ledger_hash = write_jsonl(ledger_path, rows)
    thresholds = [int(value) for value in p1["support_thresholds"]]
    coverage = coverage_audit(ledger, controls, support_thresholds=thresholds)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256(config_path)},
            "interaction_file": {"path": str(interaction_path.relative_to(PROJECT_ROOT)), "sha256": sha256(interaction_path)},
            "user_splits": {"path": str(splits_path.relative_to(PROJECT_ROOT)), "sha256": sha256(splits_path)},
            "mh0_control_train": {"path": str(source_controls_path.relative_to(PROJECT_ROOT)), "sha256": sha256(source_controls_path)},
            "mh0_control_val": {"path": str(controls_path.relative_to(PROJECT_ROOT)), "sha256": sha256(controls_path)},
            "source_split": source_split,
            "source_users": len(source_users),
            "evaluation_users": len(eval_users),
            "route": "origin_user -> anchor_item -> peer_user -> endpoint_item",
            "anchor": "up to latest distinct pre-heldout source-history items",
            "max_anchor_items_per_source": int(p1["max_anchor_items_per_source"]),
            "max_peer_users_per_anchor": int(p1["max_peer_users_per_anchor"]),
            "max_remote_items_per_peer": int(p1["max_remote_items_per_peer"]),
            "max_witness_paths_per_item": int(p1["max_witness_paths_per_item"]),
            "support_thresholds": thresholds,
            "candidate_blind_ledger": True,
        },
        "output": {"path": str(ledger_path.relative_to(PROJECT_ROOT)), "records": count, "sha256": ledger_hash},
        "posthoc_candidate_coverage": coverage,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"P1 source-only route ledger: {count} endpoint items from {len(source_users)} source users")
    for threshold in thresholds:
        audit = coverage[str(threshold)]
        print(
            f"  support>={threshold}: candidate coverage={audit['candidate_slot_coverage']:.3f}, "
            f"gold coverage={audit['gold_coverage']:.3f}, users={audit['user_coverage']:.3f}"
        )
    print(f"  ledger sha256: {ledger_hash}")


if __name__ == "__main__":
    main()
