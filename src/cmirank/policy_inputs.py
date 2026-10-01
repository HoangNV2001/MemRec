"""Train-only adapter for the hash-locked Books pseudo-episode protocol.

The physical .inter includes suffix rows. Discard the last two per user by
position before parsing item identities into this adapter; never expose the
original validation/test dictionaries, instructions, reviews or ranked lists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.data.books_protocol import books_cohorts, books_dev_cost_subset, cohort_digest
from .policy_data import (
    NEUTRAL_PSEUDO_INSTRUCTION, draft_policy_split, pseudo_target_from_train,
)
from .provenance import file_sha256
from .snapshot import make_prefix_snapshot, snapshot_sha256


COMMON_SNAPSHOT_SHA256 = "22c78a5b483e137f2b0cfff23c6027471c6c553f7313b4c1aa51ec67d9a37921"


def read_train_histories(source: Path) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[tuple[int, str]]] = {}
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"user_id", "item_id", "timestamp"}.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected interaction schema")
        for row in reader:
            uid = int(row["user_id"])
            grouped.setdefault(uid, []).append((int(row["timestamp"]), row["item_id"]))
    histories = {}
    for uid, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row[0])
        if len({timestamp for timestamp, _ in ordered}) != len(ordered):
            raise ValueError("Ambiguous tied within-user order")
        if len(ordered) >= 3:
            histories[uid] = tuple(int(item) for _, item in ordered[:-2])
    return histories


def load_locked_policy_inputs(root: Path) -> dict:
    manifest_path = root / "configs/cmirank/policy_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["approval_status"] != "researcher_approved_2026-09-30"
            or manifest["training_instruction"] != NEUTRAL_PSEUDO_INSTRUCTION):
        raise ValueError("Policy protocol is not approved")
    source = root / "data/processed/instructrec-books/instructrec-books.inter"
    if file_sha256(source) != manifest["source_inter_sha256"]:
        raise ValueError("Interaction source differs from locked protocol")
    histories = read_train_histories(source)
    cohorts, locked = books_cohorts(list(histories))
    _, exposed = books_dev_cost_subset(cohorts["all"], cohorts["dev"])
    if (locked["cohort_sha256"]["dev"] != manifest["locked_dev_sha256"]
            or cohort_digest(exposed) != manifest["exposed200_sha256"]):
        raise ValueError("Locked cohort differs")
    splits = draft_policy_split(cohorts["dev"], exposed, seed=manifest["seed"])
    groups = {}
    for name, members in splits.items():
        if cohort_digest(members) != manifest["split_sha256"][name]:
            raise ValueError(f"Locked split differs: {name}")
        if name == "integration_exposed":
            continue
        groups[name] = [uid for uid in members if pseudo_target_from_train(
            histories[uid], min_prefix_length=manifest["min_prefix_length"]) is not None]
        if cohort_digest(groups[name]) != manifest["eligible_sha256"][name]:
            raise ValueError(f"Locked eligibility differs: {name}")
    eligible = sorted(groups["policy_train"] + groups["policy_val"])
    snapshot, warmups, targets = make_prefix_snapshot(
        histories, eligible, min_prefix_length=manifest["min_prefix_length"])
    if snapshot_sha256(snapshot) != COMMON_SNAPSHOT_SHA256:
        raise ValueError("Common 1797-user snapshot differs")
    return {"manifest": manifest, "manifest_sha256": file_sha256(manifest_path),
            "groups": groups, "snapshot": snapshot,
            "warmups": warmups, "targets": targets,
            "snapshot_sha256": COMMON_SNAPSHOT_SHA256}
