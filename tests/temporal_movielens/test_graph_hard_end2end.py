import pytest

from src.temporal_movielens.graph_hard_end2end import validate_prepared_events


def config():
    return {"study": {"candidates_per_event": 10}}


def event():
    return {
        "user_id": "u1",
        "timestamp": 100,
        "gold_item_id": "9",
        "candidate_item_ids": [str(index) for index in range(10)],
        "negative_item_ids": [str(index) for index in range(9)],
        "candidate_history_overlap": [],
        "history": [{"timestamp": 10, "item_id": "known", "rating": 4.5}],
    }


def test_m7_prepared_event_contract_accepts_unique_strict_past_candidates():
    validate_prepared_events([event()], config())


@pytest.mark.parametrize("mutation", ["duplicate", "history_collision", "future_history"])
def test_m7_prepared_event_contract_rejects_leakage_and_invalid_candidates(mutation):
    value = event()
    if mutation == "duplicate":
        value["candidate_item_ids"][-1] = "8"
    elif mutation == "history_collision":
        value["history"][0]["item_id"] = "0"
        value["candidate_history_overlap"] = ["0"]
    else:
        value["history"][0]["timestamp"] = 100
    with pytest.raises(RuntimeError):
        validate_prepared_events([value], config())


def test_m7_prepared_event_contract_requires_one_event_per_user():
    first = event()
    second = {**event(), "timestamp": 200}
    with pytest.raises(RuntimeError, match="one-event-per-user"):
        validate_prepared_events([first, second], config())
