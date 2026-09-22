import csv

from src.temporal_movielens.data import (
    eligible_singleton_targets,
    historical_burstiness,
    iter_user_ratings,
    movie_memory,
    select_one_target_per_user,
    strict_history,
)


def test_iter_user_ratings_sorts_each_user_by_timestamp(tmp_path):
    path = tmp_path / "ratings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["userId", "movieId", "rating", "timestamp"])
        writer.writerow([1, 10, 4.0, 20])
        writer.writerow([1, 20, 5.0, 10])
        writer.writerow([2, 10, 3.0, 30])
    assert list(iter_user_ratings(path)) == [
        ("1", [(10, "20", 5.0), (20, "10", 4.0)]),
        ("2", [(30, "10", 3.0)]),
    ]


def test_singleton_target_and_strict_context_contract():
    events = [
        (1, "1", 4.0),
        (2, "2", 4.0),
        (3, "3", 4.0),
        (4, "4", 4.0),
        (5, "5", 4.0),
        (1000, "10", 4.5),
        (2000, "20", 5.0),
    ]
    targets = eligible_singleton_targets(
        "u",
        events,
        start=900,
        end=None,
        candidate_pool={"10", "20"},
        session_gap_seconds=300,
        positive_rating_min=4.0,
        minimum_positive_history=5,
    )
    assert [target["gold_item_id"] for target in targets] == ["10", "20"]
    selected = select_one_target_per_user(targets, salt="fixed")
    history = strict_history(events, selected["timestamp"])
    assert all(timestamp < selected["timestamp"] for timestamp, _, _ in history)
    assert selected["gold_item_id"] not in {movie for _, movie, _ in history}


def test_burstiness_and_movie_memory_are_dataset_native():
    events = [(0, "1", 4.0), (30, "2", 4.0), (100, "3", 4.0)]
    assert historical_burstiness(events, 200) == 0.5
    assert movie_memory({"title": "Film (2000)", "genres": "Drama|Crime"}) == (
        "Title: Film (2000). Genres: Drama, Crime."
    )
