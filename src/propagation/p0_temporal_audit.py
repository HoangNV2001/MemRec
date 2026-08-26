"""P0 — establish whether the interaction file supports dynamic propagation.

The audit distinguishes a per-user sequence from a global event clock.  Dynamic
cross-user propagation is admissible only when event timestamps establish a
strict, unambiguous order across users.  A file sorted by user with local
positions (1, 2, ...) is useful for leave-one-out evaluation but is not enough
to claim that one user's update preceded another user's target event.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

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


def read_events(path: Path) -> Iterable[tuple[int, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"user_id", "timestamp"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"{path} must contain TSV columns {sorted(required)}")
        for row in reader:
            yield int(row["user_id"]), float(row["timestamp"])


def audit_events(events: Iterable[tuple[int, float]]) -> Dict[str, Any]:
    timestamp_users: Dict[float, set[int]] = defaultdict(set)
    per_user: Dict[int, list[float]] = defaultdict(list)
    previous: float | None = None
    global_inversions = 0
    n_events = 0
    for user_id, timestamp in events:
        if previous is not None and timestamp < previous:
            global_inversions += 1
        previous = timestamp
        timestamp_users[timestamp].add(user_id)
        per_user[user_id].append(timestamp)
        n_events += 1
    if not n_events:
        raise ValueError("interaction file has no events")
    reused = {timestamp: users for timestamp, users in timestamp_users.items() if len(users) > 1}
    ambiguous_events = sum(len(users) for users in reused.values())
    per_user_monotonic = sum(
        all(left <= right for left, right in zip(timestamps, timestamps[1:]))
        for timestamps in per_user.values()
    )
    global_clock = global_inversions == 0 and not reused
    reason = (
        "strict global timestamp order is available"
        if global_clock
        else "timestamps are not a strict global cross-user event clock"
    )
    return {
        "n_events": n_events,
        "n_users": len(per_user),
        "timestamp_values": len(timestamp_users),
        "global_file_order_inversions": global_inversions,
        "per_user_monotonic_users": per_user_monotonic,
        "timestamp_values_reused_across_users": len(reused),
        "events_with_cross_user_ambiguous_timestamp": ambiguous_events,
        "max_users_at_one_timestamp": max(len(users) for users in timestamp_users.values()),
        "strict_global_clock": global_clock,
        "dynamic_cross_user_propagation_admissible": global_clock,
        "decision_reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether timestamps permit dynamic propagation")
    parser.add_argument("--config", default="configs/multihop/mh0_books.yaml")
    parser.add_argument("--force", action="store_true", help="replace the derived audit artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    section = config.get("propagation", {})
    interaction_path = project_path(section["interaction_file"])
    output_path = project_path(section["p0_temporal_audit"])
    if not interaction_path.exists():
        raise FileNotFoundError(interaction_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; use --force to replace this derived audit")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = audit_events(read_events(interaction_path))
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {"path": str(interaction_path.relative_to(PROJECT_ROOT)), "sha256": sha256(interaction_path)},
        }
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"P0 temporal audit: dynamic propagation admissible = {result['dynamic_cross_user_propagation_admissible']}")
    print(
        "  events={n_events}, global inversions={global_file_order_inversions}, "
        "cross-user ambiguous events={events_with_cross_user_ambiguous_timestamp}".format(**result)
    )
    print(f"  wrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
