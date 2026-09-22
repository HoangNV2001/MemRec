"""Streaming MovieLens 32M adapter for the frozen temporal study."""
from __future__ import annotations

import bisect
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

from src.temporal_books.common import stable_order


RatingEvent = tuple[int, str, float]
RATING_HEADER = ["userId", "movieId", "rating", "timestamp"]
MOVIE_HEADER = ["movieId", "title", "genres"]


def iter_user_ratings(path: Path, *, limit_users: int | None = None) -> Iterator[tuple[str, list[RatingEvent]]]:
    """Yield one timestamp-sorted history at a time from the official ordering."""
    current_user: str | None = None
    events: list[RatingEvent] = []
    yielded = 0
    previous_user = previous_movie = -1
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != RATING_HEADER:
            raise ValueError(f"{path} schema mismatch")
        for row_number, raw in enumerate(reader, start=2):
            if len(raw) != 4:
                raise ValueError(f"malformed rating row {row_number}")
            user_value, movie_value = int(raw[0]), int(raw[1])
            if user_value < previous_user or (user_value == previous_user and movie_value < previous_movie):
                raise ValueError("ratings are not ordered by userId then movieId")
            user_id, movie_id = str(user_value), sys.intern(str(movie_value))
            if current_user is None:
                current_user = user_id
            if user_id != current_user:
                events.sort()
                yield current_user, events
                yielded += 1
                if limit_users is not None and yielded >= limit_users:
                    return
                current_user, events = user_id, []
            events.append((int(raw[3]), movie_id, float(raw[2])))
            previous_user, previous_movie = user_value, movie_value
    if current_user is not None and (limit_users is None or yielded < limit_users):
        events.sort()
        yield current_user, events


def load_movies(path: Path) -> Dict[str, Dict[str, str]]:
    movies: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MOVIE_HEADER:
            raise ValueError(f"{path} schema mismatch")
        for row in reader:
            movie_id = str(int(row["movieId"]))
            if movie_id in movies:
                raise ValueError(f"duplicate movieId: {movie_id}")
            movies[movie_id] = {
                "title": row["title"].strip() or f"Movie {movie_id}",
                "genres": row["genres"].strip() or "(no genres listed)",
            }
    return movies


def positive_candidate_pool(
    path: Path, *, train_cutoff: int, positive_rating_min: float
) -> list[str]:
    values: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != RATING_HEADER:
            raise ValueError(f"{path} schema mismatch")
        for raw in reader:
            if int(raw[3]) < train_cutoff and float(raw[2]) >= positive_rating_min:
                values.add(int(raw[1]))
    return [str(value) for value in sorted(values)]


def exact_timestamp_batches(events: Sequence[RatingEvent]) -> list[list[RatingEvent]]:
    batches: list[list[RatingEvent]] = []
    index = 0
    while index < len(events):
        end = index + 1
        while end < len(events) and events[end][0] == events[index][0]:
            end += 1
        batches.append(list(events[index:end]))
        index = end
    return batches


def eligible_singleton_targets(
    user_id: str,
    events: Sequence[RatingEvent],
    *,
    start: int,
    end: int | None,
    candidate_pool: set[str],
    session_gap_seconds: int,
    positive_rating_min: float,
    minimum_positive_history: int,
) -> list[Dict[str, Any]]:
    batches = exact_timestamp_batches(events)
    positive_history = 0
    targets: list[Dict[str, Any]] = []
    for index, batch in enumerate(batches):
        timestamp = batch[0][0]
        previous_gap = float("inf") if index == 0 else timestamp - batches[index - 1][0][0]
        next_gap = float("inf") if index + 1 == len(batches) else batches[index + 1][0][0] - timestamp
        in_window = timestamp >= start and (end is None or timestamp < end)
        if in_window and len(batch) == 1 and previous_gap > session_gap_seconds and next_gap > session_gap_seconds:
            _, movie_id, rating = batch[0]
            if (
                rating >= positive_rating_min
                and movie_id in candidate_pool
                and positive_history >= minimum_positive_history
            ):
                targets.append(
                    {
                        "user_id": user_id,
                        "timestamp": timestamp,
                        "gold_item_id": movie_id,
                        "rating": rating,
                        "previous_gap_seconds": None if previous_gap == float("inf") else int(previous_gap),
                        "next_gap_seconds": None if next_gap == float("inf") else int(next_gap),
                    }
                )
        positive_history += sum(rating >= positive_rating_min for _, _, rating in batch)
    return targets


def select_one_target_per_user(targets: Sequence[Mapping[str, Any]], *, salt: str) -> Dict[str, Any]:
    if not targets:
        raise ValueError("cannot select from an empty target list")
    return dict(
        min(
            targets,
            key=lambda row: stable_order(
                f"{salt}\0{row['user_id']}\0{row['timestamp']}\0{row['gold_item_id']}"
            ),
        )
    )


def strict_history(events: Sequence[RatingEvent], target_time: int) -> list[RatingEvent]:
    end = bisect.bisect_left(events, (target_time, "", float("-inf")))
    return list(events[:end])


def historical_burstiness(events: Sequence[RatingEvent], cutoff: int) -> float:
    timestamps = [timestamp for timestamp, _, _ in events if timestamp < cutoff]
    if len(timestamps) < 2:
        return 0.0
    return sum(following - previous <= 60 for previous, following in zip(timestamps, timestamps[1:])) / (
        len(timestamps) - 1
    )


def movie_memory(metadata: Mapping[str, str]) -> str:
    genres = metadata["genres"].replace("|", ", ")
    return f"Title: {metadata['title']}. Genres: {genres}."
