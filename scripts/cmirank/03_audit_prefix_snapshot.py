#!/usr/bin/env python3
"""CPU-only graph leakage audit; smoke 20 users before --query-users all.

This does not create LLM memories or prove Stage-W cache safety. It reads no
original test candidate list, instruction, validation item, or test target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cmirank.policy_data import (
    NEUTRAL_PSEUDO_INSTRUCTION, draft_policy_split, pseudo_target_from_train,
)
from src.cmirank.snapshot import make_prefix_snapshot, snapshot_sha256
from src.cmirank.provenance import file_sha256
from src.data.books_protocol import books_cohorts, books_dev_cost_subset, cohort_digest
from src.data.dataset_base import RecDataset
from src.memory.graph import UserItemGraph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-users", choices=("20", "all"), default="20")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "configs/cmirank/policy_split_manifest.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("approval_status") != "researcher_approved_2026-09-30"
            or manifest.get("pseudo_target_rule") != "last_train_interaction_novel_to_prefix"
            or manifest.get("training_instruction") != NEUTRAL_PSEUDO_INSTRUCTION):
        raise ValueError("Policy split is not approved")
    source = ROOT / "data/processed/instructrec-books/instructrec-books.inter"
    observed_sha = file_sha256(source)
    if observed_sha != manifest["source_inter_sha256"]:
        raise ValueError("Interaction source differs from approved split")

    cohorts, locked = books_cohorts(range(7377))
    _, exposed200 = books_dev_cost_subset(cohorts["all"], cohorts["dev"])
    if (locked["cohort_sha256"]["dev"] != manifest["locked_dev_sha256"]
            or cohort_digest(exposed200) != manifest["exposed200_sha256"]):
        raise ValueError("Parent cohort differs from approved split")
    split = draft_policy_split(cohorts["dev"], exposed200, seed=manifest["seed"])
    for name, users in split.items():
        if cohort_digest(users) != manifest["split_sha256"][name]:
            raise ValueError(f"Approved split differs: {name}")

    dataset = RecDataset(str(source), seed=42, precompute_negatives=False)
    eligible: list[int] = []
    for name in ("policy_train", "policy_val"):
        group = [user_id for user_id in split[name]
                 if pseudo_target_from_train(
                     dataset.train_data[user_id],
                     min_prefix_length=manifest["min_prefix_length"],
                 ) is not None]
        if cohort_digest(group) != manifest["eligible_sha256"][name]:
            raise ValueError(f"Approved eligibility differs: {name}")
        eligible.extend(group)
    query_ids = sorted(eligible)[:20] if args.query_users == "20" else eligible
    snapshot, warmup_events, pseudo_targets = make_prefix_snapshot(
        dataset.train_data, query_ids,
        min_prefix_length=manifest["min_prefix_length"],
    )
    graph = UserItemGraph(snapshot)
    if any(tuple(graph.get_user_items(user_id)) !=
           tuple(dataset.train_data[user_id][:-2]) for user_id in query_ids):
        raise AssertionError("Query graph history is not exactly train[:-2]")
    if any(warmup_events[user_id] != dataset.train_data[user_id][-2]
           or pseudo_targets[user_id] != dataset.train_data[user_id][-1]
           for user_id in query_ids):
        raise AssertionError("Warm-up/target positions differ from approved protocol")
    if any(target in graph.get_user_items(user_id)
           for user_id, target in pseudo_targets.items()):
        raise AssertionError("Pseudo-target leaked into a query user's graph edges")
    if any(hasattr(snapshot, key) for key in
           ("test_data", "valid_data", "ranked_lists", "instructions", "reviews")):
        raise AssertionError("PrefixSnapshot exposes non-train fields")
    print(json.dumps({
        "status": "GRAPH_SNAPSHOT_AUDIT_PASS_NOT_STAGE_W_PROOF",
        "scope": args.query_users,
        "query_users": len(query_ids),
        "removed_query_edges": 2 * len(pseudo_targets),
        "stage_w_events_prepared_not_run": len(warmup_events),
        "source_train_edges": sum(len(items) for items in dataset.train_data.values()),
        "graph_users": graph.get_stats()["n_users"],
        "graph_edges": graph.get_stats()["n_edges"],
        "snapshot_sha256": snapshot_sha256(snapshot),
        "other_user_target_edges_allowed": sum(
            len(graph.get_item_users(target)) for target in pseudo_targets.values()),
        "original_test_candidates_accessed": False,
        "original_instruction_accessed": False,
        "test_or_valid_target_accessed": False,
        "stage_w_memory_proven_safe": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
