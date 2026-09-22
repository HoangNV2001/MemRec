from src.temporal_movielens.graph import event_seeds, score_view


def test_movie_event_seeds_are_recent_unique_and_strictly_prepared():
    event = {
        "history": [
            {"item_id": "a"},
            {"item_id": "b"},
            {"item_id": "a"},
            {"item_id": "c"},
        ]
    }
    assert event_seeds(event, 3) == ["c", "a", "b"]


def test_graph_scoring_never_requires_gold_label():
    event = {
        "user_id": "u",
        "timestamp": 100,
        "candidate_item_ids": ["a", "b"],
        "history": [{"item_id": "seed"}],
    }
    graph = {"seed": [("a",)], "a": [("b",)]}
    rows = score_view(
        [event],
        graph,
        {"seed_items": 6, "restart_probability": 0.15, "max_walk_steps": 64},
        walks=1000,
    )
    assert rows[0]["gold_label_used_for_scoring"] is False
    assert rows[0]["one_step_scores"]["a"] == 1.0
    assert rows[0]["ppr_scores"]["b"] > 0
