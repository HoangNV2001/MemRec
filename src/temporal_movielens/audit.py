"""Reproducible raw-data and temporal audit for MovieLens 32M.

The audit is deliberately outcome-free: it verifies source integrity, measures
rating-entry burstiness, freezes event-quantile cutoffs, and counts whether the
pre-registered singleton-session cohort is feasible. It does not sample a
cohort, construct recommendation candidates, score a graph, or call an LLM.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file


SCHEMA_VERSION = 1
RATING_HEADER = ["userId", "movieId", "rating", "timestamp"]
MOVIE_HEADER = ["movieId", "title", "genres"]
TAG_HEADER = ["userId", "movieId", "tag", "timestamp"]
LINK_HEADER = ["movieId", "imdbId", "tmdbId"]
Event = tuple[int, int, float]


def utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_at_quantile(values: np.ndarray, quantile: float) -> int:
    """Return the first timestamp reaching an event quantile.

    Equal timestamps are assigned to the later partition by the caller, which
    preserves strict-past batch semantics.
    """
    if values.ndim != 1 or not len(values):
        raise ValueError("timestamps must be a non-empty one-dimensional array")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    index = max(0, math.ceil(len(values) * quantile) - 1)
    values.partition(index)
    return int(values[index])


def numeric_quantiles(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        str(probability): float(np.quantile(array, probability))
        for probability in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def counter_quantiles(counts: Counter[int]) -> dict[str, int]:
    """Nearest-rank quantiles without expanding millions of session sizes."""
    total = sum(counts.values())
    if not total:
        return {}
    ordered = sorted(counts.items())
    result: dict[str, int] = {}
    for probability in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
        target = 1 if probability == 0 else math.ceil(total * probability)
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative >= target:
                result[str(probability)] = value
                break
    return result


def session_sizes(events: Sequence[Event], maximum_gap: int) -> list[int]:
    if maximum_gap < 0:
        raise ValueError("session gap cannot be negative")
    if not events:
        return []
    ordered = sorted(events, key=lambda row: row[0])
    sizes: list[int] = []
    current = 1
    for previous, following in zip(ordered, ordered[1:]):
        if following[0] - previous[0] <= maximum_gap:
            current += 1
        else:
            sizes.append(current)
            current = 1
    sizes.append(current)
    return sizes


def eligible_events_for_user(
    events: Sequence[Event],
    *,
    candidate_pool: set[int],
    train_cutoff: int,
    validation_cutoff: int,
    session_gap: int,
    positive_rating_min: float,
    minimum_positive_history: int,
) -> dict[str, int]:
    """Count eligible singleton-session targets without selecting outcomes."""
    ordered = sorted(events, key=lambda row: row[0])
    batches: list[list[Event]] = []
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        batches.append(ordered[index:end])
        index = end

    counts = {"development": 0, "test": 0}
    positive_history = 0
    for batch_index, batch in enumerate(batches):
        timestamp = batch[0][0]
        previous_gap = math.inf if batch_index == 0 else timestamp - batches[batch_index - 1][0][0]
        next_gap = math.inf if batch_index + 1 == len(batches) else batches[batch_index + 1][0][0] - timestamp
        if len(batch) == 1 and previous_gap > session_gap and next_gap > session_gap:
            _, movie_id, rating = batch[0]
            if (
                rating >= positive_rating_min
                and movie_id in candidate_pool
                and positive_history >= minimum_positive_history
            ):
                if train_cutoff <= timestamp < validation_cutoff:
                    counts["development"] += 1
                elif timestamp >= validation_cutoff:
                    counts["test"] += 1
        positive_history += sum(rating >= positive_rating_min for _, _, rating in batch)
    return counts


class RatingAccumulator:
    def __init__(self, thresholds: Sequence[int]) -> None:
        self.thresholds = tuple(int(value) for value in thresholds)
        self.user_counts: list[int] = []
        self.unique_timestamp_ratios: list[float] = []
        self.fast_60_fractions: list[float] = []
        self.interval_counts: Counter[str] = Counter()
        self.session_size_counts = {threshold: Counter() for threshold in self.thresholds}
        self.sessions_per_user = {threshold: [] for threshold in self.thresholds}
        self.multi_session_events = Counter()

    def add_user(self, events: Sequence[Event]) -> None:
        if not events:
            return
        ordered = sorted(events, key=lambda row: row[0])
        timestamps = [row[0] for row in ordered]
        self.user_counts.append(len(ordered))
        self.unique_timestamp_ratios.append(len(set(timestamps)) / len(timestamps))
        fast_60 = 0
        for previous, following in zip(timestamps, timestamps[1:]):
            delta = following - previous
            self.interval_counts["total"] += 1
            for name, limit in (
                ("same_second", 0),
                ("le_10s", 10),
                ("le_60s", 60),
                ("le_300s", 300),
                ("le_1h", 3600),
                ("le_1d", 86400),
            ):
                self.interval_counts[name] += int(delta <= limit)
            fast_60 += int(delta <= 60)
        self.fast_60_fractions.append(fast_60 / max(1, len(ordered) - 1))
        for threshold in self.thresholds:
            sizes = session_sizes(ordered, threshold)
            self.session_size_counts[threshold].update(sizes)
            self.sessions_per_user[threshold].append(len(sizes))
            self.multi_session_events[threshold] += sum(size for size in sizes if size > 1)

    def result(self, rows: int) -> dict[str, Any]:
        total_intervals = self.interval_counts["total"]
        intervals = {
            key: {"count": value, "rate": value / total_intervals if total_intervals else 0.0}
            for key, value in self.interval_counts.items()
            if key != "total"
        }
        sessions: dict[str, Any] = {}
        for threshold in self.thresholds:
            sizes = self.session_size_counts[threshold]
            total_sessions = sum(sizes.values())
            multi_sessions = sum(count for size, count in sizes.items() if size > 1)
            sessions[str(threshold)] = {
                "sessions": total_sessions,
                "size_quantiles_nearest_rank": counter_quantiles(sizes),
                "sessions_per_user_quantiles": numeric_quantiles(self.sessions_per_user[threshold]),
                "sessions_size_gt_1_rate": multi_sessions / total_sessions if total_sessions else 0.0,
                "events_in_sessions_size_gt_1_rate": self.multi_session_events[threshold] / rows if rows else 0.0,
                "max_session_size": max(sizes, default=0),
            }
        return {
            "ratings_per_user_quantiles": numeric_quantiles(self.user_counts),
            "unique_timestamp_ratio_per_user_quantiles": numeric_quantiles(self.unique_timestamp_ratios),
            "per_user_fraction_intervals_le_60s_quantiles": numeric_quantiles(self.fast_60_fractions),
            "users_ge_50pct_intervals_le_60s": sum(value >= 0.5 for value in self.fast_60_fractions),
            "users_ge_80pct_intervals_le_60s": sum(value >= 0.8 for value in self.fast_60_fractions),
            "total_consecutive_intervals": total_intervals,
            "intervals": intervals,
            "sessionization_by_gap_seconds": sessions,
        }


def scan_ratings(
    path: Path, *, expected_rows: int, thresholds: Sequence[int], positive_rating_min: float
) -> tuple[dict[str, Any], set[int]]:
    timestamps = np.empty(expected_rows, dtype=np.int64)
    accumulator = RatingAccumulator(thresholds)
    rating_counts: Counter[str] = Counter()
    first_positive_timestamp: dict[int, int] = {}
    rated_movies: set[int] = set()
    current_user: int | None = None
    current_events: list[Event] = []
    previous_user = previous_movie = None
    duplicate_user_movie = 0
    rows = 0
    minimum_timestamp = math.inf
    maximum_timestamp = 0

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != RATING_HEADER:
            raise ValueError(f"{path} does not have the MovieLens rating schema")
        for raw in reader:
            if len(raw) != len(RATING_HEADER):
                raise ValueError(f"malformed rating row {rows + 2}")
            user_id, movie_id, timestamp = int(raw[0]), int(raw[1]), int(raw[3])
            rating = float(raw[2])
            if rows >= expected_rows:
                raise ValueError("ratings file contains more rows than the official contract")
            if previous_user is not None and (user_id < previous_user or (user_id == previous_user and movie_id < previous_movie)):
                raise ValueError("ratings file is not ordered by userId then movieId")
            if previous_user == user_id and previous_movie == movie_id:
                duplicate_user_movie += 1
            if current_user is None:
                current_user = user_id
            if user_id != current_user:
                accumulator.add_user(current_events)
                current_events = []
                current_user = user_id
            current_events.append((timestamp, movie_id, rating))
            timestamps[rows] = timestamp
            rows += 1
            rated_movies.add(movie_id)
            rating_counts[str(rating)] += 1
            if rating >= positive_rating_min:
                first_positive_timestamp[movie_id] = min(first_positive_timestamp.get(movie_id, timestamp), timestamp)
            minimum_timestamp = min(minimum_timestamp, timestamp)
            maximum_timestamp = max(maximum_timestamp, timestamp)
            previous_user, previous_movie = user_id, movie_id
    accumulator.add_user(current_events)
    if rows != expected_rows:
        raise ValueError(f"ratings row count mismatch: expected {expected_rows}, got {rows}")
    return {
        "rows": rows,
        "users": len(accumulator.user_counts),
        "rated_movies": len(rated_movies),
        "duplicate_user_movie": duplicate_user_movie,
        "rating_distribution": dict(sorted(rating_counts.items(), key=lambda pair: float(pair[0]))),
        "timestamp_min": int(minimum_timestamp),
        "timestamp_max": maximum_timestamp,
        "timestamps": timestamps,
        "temporal_behavior": accumulator.result(rows),
        "first_positive_timestamp": first_positive_timestamp,
    }, rated_movies


def count_auxiliary_rows(path: Path, expected_header: Sequence[str], *, limit: int | None = None) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != list(expected_header):
            raise ValueError(f"{path} schema mismatch")
        rows = 0
        for _ in reader:
            if limit is not None and rows >= limit:
                break
            rows += 1
        return rows


def scan_movies(path: Path, *, limit: int | None = None) -> tuple[dict[str, int], set[int]]:
    movie_ids: set[int] = set()
    no_genres = duplicate_ids = rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MOVIE_HEADER:
            raise ValueError(f"{path} schema mismatch")
        for row in reader:
            if limit is not None and rows >= limit:
                break
            movie_id = int(row["movieId"])
            duplicate_ids += int(movie_id in movie_ids)
            movie_ids.add(movie_id)
            no_genres += int(not row["genres"] or row["genres"] == "(no genres listed)")
            rows += 1
    return {"rows": rows, "unique_movie_ids": len(movie_ids), "duplicate_ids": duplicate_ids, "no_genres": no_genres}, movie_ids


def count_eligibility(
    path: Path,
    *,
    candidate_pool: set[int],
    train_cutoff: int,
    validation_cutoff: int,
    session_gap: int,
    positive_rating_min: float,
    minimum_positive_history: int,
) -> dict[str, int]:
    event_counts = Counter()
    user_counts = Counter()
    current_user: int | None = None
    current_events: list[Event] = []

    def finish(events: Sequence[Event]) -> None:
        counts = eligible_events_for_user(
            events,
            candidate_pool=candidate_pool,
            train_cutoff=train_cutoff,
            validation_cutoff=validation_cutoff,
            session_gap=session_gap,
            positive_rating_min=positive_rating_min,
            minimum_positive_history=minimum_positive_history,
        )
        for partition, count in counts.items():
            event_counts[partition] += count
            user_counts[partition] += int(count > 0)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != RATING_HEADER:
            raise ValueError(f"{path} schema mismatch")
        for raw in reader:
            user_id = int(raw[0])
            event = (int(raw[3]), int(raw[1]), float(raw[2]))
            if current_user is None:
                current_user = user_id
            if user_id != current_user:
                finish(current_events)
                current_events = []
                current_user = user_id
            current_events.append(event)
    finish(current_events)
    return {
        "development_events": event_counts["development"],
        "development_users": user_counts["development"],
        "test_events": event_counts["test"],
        "test_users": user_counts["test"],
    }


def smoke_parse(config: Mapping[str, Any], rows: int) -> dict[str, Any]:
    if not 20 <= rows <= 100:
        raise ValueError("smoke must parse 20-100 rows per file")
    dataset = config["dataset"]
    files = {
        "ratings": (project_path(dataset["ratings_file"]), RATING_HEADER),
        "tags": (project_path(dataset["tags_file"]), TAG_HEADER),
        "links": (project_path(dataset["links_file"]), LINK_HEADER),
    }
    parsed = {name: count_auxiliary_rows(path, header, limit=rows) for name, (path, header) in files.items()}
    movies, _ = scan_movies(project_path(dataset["movies_file"]), limit=rows)
    parsed["movies"] = movies["rows"]
    if any(value != rows for value in parsed.values()):
        raise ValueError(f"smoke row count mismatch: {parsed}")
    return {"decision": "pass", "rows_per_file": rows, "parsed": parsed, "artifact_written": False}


def verify_md5(config: Mapping[str, Any]) -> dict[str, str]:
    dataset = config["dataset"]
    expected = {str(name): str(value) for name, value in config["expected"]["md5"].items()}
    checksum_path = project_path(dataset["checksums_file"])
    listed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name.strip()] = digest
    if listed != expected:
        raise ValueError("checksums.txt does not match the locked official MD5 contract")
    resolved: dict[str, str] = {}
    for name, wanted in expected.items():
        got = md5_file(checksum_path.parent / name)
        if got != wanted:
            raise ValueError(f"MD5 mismatch for {name}: expected {wanted}, got {got}")
        resolved[name] = got
    return resolved


def run_full(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    dataset, expected = config["dataset"], config["expected"]
    split, protocol = config["temporal_split"], config["protocol"]
    ratings_path = project_path(dataset["ratings_file"])
    movies_path = project_path(dataset["movies_file"])
    tags_path = project_path(dataset["tags_file"])
    links_path = project_path(dataset["links_file"])
    official_md5 = verify_md5(config)
    ratings, rated_movies = scan_ratings(
        ratings_path,
        expected_rows=int(expected["ratings_rows"]),
        thresholds=protocol["session_audit_thresholds_seconds"],
        positive_rating_min=float(protocol["positive_rating_min"]),
    )
    timestamps = ratings.pop("timestamps")
    first_positive = ratings.pop("first_positive_timestamp")
    train_cutoff = timestamp_at_quantile(timestamps, float(split["train_quantile"]))
    validation_cutoff = timestamp_at_quantile(timestamps, float(split["validation_quantile"]))
    if train_cutoff >= validation_cutoff:
        raise ValueError("temporal cutoffs collapsed")
    candidate_pool = {movie for movie, timestamp in first_positive.items() if timestamp < train_cutoff}
    split_counts = {
        "train": int(np.sum(timestamps < train_cutoff)),
        "development": int(np.sum((timestamps >= train_cutoff) & (timestamps < validation_cutoff))),
        "test": int(np.sum(timestamps >= validation_cutoff)),
    }
    movies, metadata_movies = scan_movies(movies_path)
    tags_rows = count_auxiliary_rows(tags_path, TAG_HEADER)
    links_rows = count_auxiliary_rows(links_path, LINK_HEADER)
    if ratings["users"] != int(expected["users"]):
        raise ValueError(f"user count mismatch: {ratings['users']}")
    for actual, key in ((movies["rows"], "movies_rows"), (tags_rows, "tags_rows"), (links_rows, "links_rows")):
        if actual != int(expected[key]):
            raise ValueError(f"{key} mismatch: expected {expected[key]}, got {actual}")
    eligibility = count_eligibility(
        ratings_path,
        candidate_pool=candidate_pool,
        train_cutoff=train_cutoff,
        validation_cutoff=validation_cutoff,
        session_gap=int(protocol["session_gap_seconds"]),
        positive_rating_min=float(protocol["positive_rating_min"]),
        minimum_positive_history=int(protocol["minimum_positive_history"]),
    )
    required_users = int(protocol["required_primary_users"])
    decision = "pass" if eligibility["test_users"] >= required_users else "hard_stop"
    input_paths = {
        "ratings": ratings_path,
        "movies": movies_path,
        "tags": tags_path,
        "links": links_path,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "recommendation_outcomes_read": False,
        "raw_integrity": {
            "official_md5": official_md5,
            "sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        },
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            **{name: {"path": str(path.relative_to(PROJECT_ROOT))} for name, path in input_paths.items()},
        },
        "ratings": {
            **ratings,
            "timestamp_min_utc": utc(ratings["timestamp_min"]),
            "timestamp_max_utc": utc(ratings["timestamp_max"]),
        },
        "metadata": {
            **movies,
            "rated_movies": len(rated_movies),
            "rated_movies_missing_metadata": len(rated_movies - metadata_movies),
            "metadata_movies_never_rated": len(metadata_movies - rated_movies),
            "tags_rows": tags_rows,
            "links_rows": links_rows,
        },
        "temporal_split": {
            "train_quantile": float(split["train_quantile"]),
            "validation_quantile": float(split["validation_quantile"]),
            "train_cutoff": train_cutoff,
            "train_cutoff_utc": utc(train_cutoff),
            "validation_cutoff": validation_cutoff,
            "validation_cutoff_utc": utc(validation_cutoff),
            "event_counts": split_counts,
        },
        "protocol_feasibility": {
            "positive_candidate_pool_movies": len(candidate_pool),
            "session_gap_seconds": int(protocol["session_gap_seconds"]),
            "minimum_positive_history": int(protocol["minimum_positive_history"]),
            "positive_rating_min": float(protocol["positive_rating_min"]),
            "required_primary_users": required_users,
            "eligible_singleton_session": eligibility,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MovieLens 32M for temporal adaptation")
    parser.add_argument("--config", default="configs/temporal_movielens32m/m0_audit.yaml")
    parser.add_argument("--smoke", action="store_true", help="parse a few rows from every source and write nothing")
    parser.add_argument("--smoke-rows", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="replace the derived full-audit artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.smoke:
        print(json.dumps(smoke_parse(config, args.smoke_rows), indent=2, sort_keys=True))
        return
    output_path = project_path(config["dataset"]["audit_output"])
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; use --force only for an intentional rerun")
    result = run_full(config, config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "ratings": result["ratings"]["rows"],
        "users": result["ratings"]["users"],
        "temporal_split": result["temporal_split"],
        "protocol_feasibility": result["protocol_feasibility"],
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(output_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
