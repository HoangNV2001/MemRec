#!/usr/bin/env python3
"""CPU-only split audit; never writes a manifest or reads test labels."""

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
from src.cmirank.provenance import file_sha256
from src.data.books_protocol import books_cohorts, books_dev_cost_subset, cohort_digest
from src.data.dataset_base import RecDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True,
                        help="draft split seed; requires human signoff before training")
    parser.add_argument("--min-prefix-length", type=int, default=5,
                        help="minimum prefix length; verified against locked manifest")
    parser.add_argument("--verify-locked", type=Path,
                        help="read-only verification of a researcher-approved manifest")
    args = parser.parse_args()

    cohorts, locked = books_cohorts(range(7377))
    _, exposed200 = books_dev_cost_subset(cohorts["all"], cohorts["dev"])
    split = draft_policy_split(cohorts["dev"], exposed200, seed=args.seed)
    dataset_path = ROOT / "data/processed/instructrec-books/instructrec-books.inter"
    dataset = RecDataset(
        str(dataset_path),
        seed=42, precompute_negatives=False,
    )
    counts = {}
    eligible_sha256 = {}
    ineligible_user_ids = {}
    for name in ("policy_train", "policy_val"):
        eligible = [
            user_id for user_id in split[name]
            if pseudo_target_from_train(dataset.train_data[user_id],
                                        min_prefix_length=args.min_prefix_length) is not None
        ]
        ineligible_user_ids[name] = sorted(set(split[name]) - set(eligible))
        eligible_sha256[name] = cohort_digest(eligible)
        counts[name] = {"users": len(split[name]),
                        "eligible_pseudo_targets": len(eligible)}

    observed = {
        "seed": args.seed,
        "min_prefix_length": args.min_prefix_length,
        "source_inter_sha256": file_sha256(dataset_path),
        "locked_dev_sha256": locked["cohort_sha256"]["dev"],
        "exposed200_sha256": cohort_digest(exposed200),
        "split_sha256": {name: cohort_digest(ids) for name, ids in split.items()},
        "eligible_sha256": eligible_sha256,
        "ineligible_user_ids": ineligible_user_ids,
        "counts": counts,
    }
    status = "DRAFT_UNAPPROVED_NO_TRAINING"
    if args.verify_locked:
        manifest = json.loads(args.verify_locked.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported locked manifest schema")
        if manifest.get("approval_status") != "researcher_approved_2026-09-30":
            raise ValueError("Manifest lacks the recorded researcher approval")
        for key, value in observed.items():
            if manifest.get(key) != value:
                raise ValueError(f"Locked policy manifest mismatch: {key}")
        if manifest.get("pseudo_target_rule") != "last_train_interaction_novel_to_prefix":
            raise ValueError("Locked pseudo-target rule mismatch")
        if manifest.get("training_instruction") != NEUTRAL_PSEUDO_INSTRUCTION:
            raise ValueError("Locked target-blind training instruction mismatch")
        status = "LOCKED_MANIFEST_VERIFIED_NO_TRAINING"
    print(json.dumps({
        "status": status,
        **observed,
        "heldout_targets_accessed": False,
        "original_test_candidates_accessed": False,
        "original_instruction_reused_for_pseudo_target": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
