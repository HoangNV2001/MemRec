"""Audit whether Kaggle Amazon Books reviews permit causal temporal replay.

The source has a real Unix ``review/time`` field, unlike the existing
InstructRec conversion.  It still has many same-day ties, so the admissible
semantics are *strict-past batches*: every event with timestamp ``t`` reads
state built from timestamps strictly smaller than ``t``.  P0 never makes an LLM
request and streams the CSV instead of loading its multi-gigabyte review text.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, set_csv_field_limit, sha256_file


SCHEMA_VERSION = 1
REVIEW_COLUMNS = {"Id", "Title", "User_id", "review/score", "review/time", "review/text"}
METADATA_COLUMNS = {"Title", "description", "authors", "categories"}


def timestamp_at_quantile(counts: Counter[int], quantile: float) -> int:
    if not 0.0 < quantile < 1.0:
        raise ValueError("temporal quantiles must lie strictly between 0 and 1")
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot select a cutoff from zero valid timestamps")
    # The first timestamp whose cumulative event count reaches the requested
    # fraction.  Events equal to the cutoff go to the later split, preserving
    # strict-past batch semantics.
    target = quantile * total
    cumulative = 0
    for timestamp in sorted(counts):
        cumulative += counts[timestamp]
        if cumulative >= target:
            return timestamp
    raise AssertionError("unreachable quantile selection")


def utc_day(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def audit_reviews(
    path: Path, *, train_quantile: float, validation_quantile: float, metadata_titles: set[str]
) -> Dict[str, Any]:
    set_csv_field_limit()
    timestamp_counts: Counter[int] = Counter()
    invalid_timestamp_rows = 0
    missing_required_rows = 0
    missing_by_column: Counter[str] = Counter()
    n_rows = 0
    previous_timestamp: int | None = None
    file_order_inversions = 0
    usable_metadata_title_matches = 0
    usable_metadata_title_missing = 0
    usable_empty_titles = 0

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REVIEW_COLUMNS <= set(reader.fieldnames):
            raise ValueError(f"{path} missing required review column(s): {sorted(REVIEW_COLUMNS)}")
        for row in reader:
            n_rows += 1
            missing_columns = [column for column in ("Id", "User_id", "review/time") if not (row.get(column) or "").strip()]
            if missing_columns:
                missing_required_rows += 1
                missing_by_column.update(missing_columns)
                continue
            try:
                timestamp = int(row["review/time"])
            except (TypeError, ValueError):
                invalid_timestamp_rows += 1
                continue
            if timestamp <= 0:
                invalid_timestamp_rows += 1
                continue
            review_title = (row.get("Title") or "").strip()
            if not review_title:
                usable_empty_titles += 1
            elif review_title in metadata_titles:
                usable_metadata_title_matches += 1
            else:
                usable_metadata_title_missing += 1
            if previous_timestamp is not None and timestamp < previous_timestamp:
                file_order_inversions += 1
            previous_timestamp = timestamp
            timestamp_counts[timestamp] += 1

    valid_events = sum(timestamp_counts.values())
    if not valid_events:
        raise ValueError("no valid timestamped review events")
    train_cutoff = timestamp_at_quantile(timestamp_counts, train_quantile)
    validation_cutoff = timestamp_at_quantile(timestamp_counts, validation_quantile)
    if train_cutoff >= validation_cutoff:
        raise ValueError("temporal split cutoffs collapse; choose less extreme quantiles")
    tied_events = sum(count for count in timestamp_counts.values() if count > 1)
    split_counts = {
        "train_strictly_before": sum(count for timestamp, count in timestamp_counts.items() if timestamp < train_cutoff),
        "validation": sum(count for timestamp, count in timestamp_counts.items() if train_cutoff <= timestamp < validation_cutoff),
        "test_from": sum(count for timestamp, count in timestamp_counts.items() if timestamp >= validation_cutoff),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "review_schema": {"required_columns": sorted(REVIEW_COLUMNS)},
        "n_rows": n_rows,
        "valid_timestamped_events": valid_events,
        "missing_required_rows": missing_required_rows,
        "missing_required_rows_by_column": dict(sorted(missing_by_column.items())),
        "usable_event_fraction": valid_events / n_rows if n_rows else 0.0,
        "metadata_title_join": {
            "usable_events_with_exact_metadata_title": usable_metadata_title_matches,
            "usable_events_without_exact_metadata_title": usable_metadata_title_missing,
            "usable_events_without_title": usable_empty_titles,
            "exact_title_coverage": usable_metadata_title_matches / valid_events if valid_events else 0.0,
            "policy": "Use metadata only for exact title matches; unmatched items use review-derived text and never title normalization/fuzzy matching.",
        },
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "timestamp_min": min(timestamp_counts),
        "timestamp_max": max(timestamp_counts),
        "timestamp_min_utc": utc_day(min(timestamp_counts)),
        "timestamp_max_utc": utc_day(max(timestamp_counts)),
        "distinct_timestamps": len(timestamp_counts),
        "same_timestamp_events": tied_events,
        "max_events_at_one_timestamp": max(timestamp_counts.values()),
        "global_file_order_inversions": file_order_inversions,
        "temporal_split": {
            "train_quantile": train_quantile,
            "validation_quantile": validation_quantile,
            "train_cutoff": train_cutoff,
            "train_cutoff_utc": utc_day(train_cutoff),
            "validation_cutoff": validation_cutoff,
            "validation_cutoff_utc": utc_day(validation_cutoff),
            "event_counts": split_counts,
        },
        "decision": {
            "strict_event_total_order_admissible": False,
            "strict_past_timestamp_batch_replay_admissible": valid_events > 0,
            "required_row_filter": "Drop rows without Id/User_id/review-time or an integer timestamp before replay; do not impute anonymous users.",
            "same_timestamp_policy": "Events at t may read/write only after the complete t batch; routes read state at timestamps < t.",
            "reason": "Usable rows have Unix review dates and establish a global coarse timeline; ties prohibit an arbitrary within-timestamp order.",
        },
    }


def audit_metadata(path: Path) -> tuple[Dict[str, Any], set[str]]:
    set_csv_field_limit()
    n_rows = 0
    empty_titles = 0
    nonempty_description = 0
    title_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not METADATA_COLUMNS <= set(reader.fieldnames):
            raise ValueError(f"{path} missing required metadata column(s): {sorted(METADATA_COLUMNS)}")
        for row in reader:
            n_rows += 1
            title = (row.get("Title") or "").strip()
            if not title:
                empty_titles += 1
            else:
                title_counts[title] += 1
            nonempty_description += int(bool((row.get("description") or "").strip()))
    return {
        "metadata_schema": {"required_columns": sorted(METADATA_COLUMNS)},
        "n_rows": n_rows,
        "empty_title_rows": empty_titles,
        "distinct_nonempty_titles": len(title_counts),
        "duplicate_title_rows": sum(count for count in title_counts.values() if count > 1),
        "nonempty_description_rows": nonempty_description,
        "join_warning": "Metadata has no Id column. Only exact title matching is eligible and its review-row coverage is reported separately.",
    }, set(title_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Kaggle Amazon Books temporal replay")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/dataset_audit.yaml")
    parser.add_argument("--force", action="store_true", help="replace the derived P0 artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    dataset = config["dataset"]
    split = config["temporal_split"]
    reviews_path = project_path(dataset["reviews_file"])
    metadata_path = project_path(dataset["metadata_file"])
    output_path = project_path(dataset["p0_audit"])
    for path in (config_path, reviews_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; use --force to replace a derived artifact")

    metadata, metadata_titles = audit_metadata(metadata_path)
    result = audit_reviews(
        reviews_path,
        train_quantile=float(split["train_quantile"]),
        validation_quantile=float(split["validation_quantile"]),
        metadata_titles=metadata_titles,
    )
    result["metadata"] = metadata
    result["inputs"] = {
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "reviews": {"path": str(reviews_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(reviews_path)},
        "metadata": {"path": str(metadata_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(metadata_path)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    decision = result["decision"]["strict_past_timestamp_batch_replay_admissible"]
    temporal = result["temporal_split"]
    print(f"P0 strict-past batch replay admissible = {decision}")
    print(f"  events={result['valid_timestamped_events']:,}, timestamps={result['distinct_timestamps']:,}, tied events={result['same_timestamp_events']:,}")
    print(f"  train < {temporal['train_cutoff_utc']}; validation < {temporal['validation_cutoff_utc']}")
    print(f"  wrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
