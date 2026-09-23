from src.temporal_movielens.hard_headroom import (
    graph_hard_candidate_event,
    headroom_metrics,
    one_step_all_scores,
)


def study_config():
    return {
        "study": {
            "run_id": "toy-hnv",
            "candidates_per_event": 10,
            "candidate_salt": "candidate",
            "candidate_order_salt": "order",
        },
        "graph": {
            "bootstrap_resamples": 100,
            "bootstrap_seed": 7,
            "headroom_min_delta": 0.02,
        },
    }


def test_all_item_one_step_scores_preserve_transition_weights():
    graph = {"s1": [("a", "b"), ("a",)], "s2": [("b",)]}
    assert one_step_all_scores(["s1", "s2"], graph) == {"a": 0.375, "b": 0.625}


def test_graph_hard_candidates_are_deterministic_and_do_not_require_gold_reachability():
    event = {
        "user_id": "u",
        "timestamp": 100,
        "gold_item_id": "gold",
        "history": [{"item_id": "known"}],
        "_known_item_ids": ["known"],
    }
    exact = {str(index): 1.0 for index in range(12)}
    exact["only_exact"] = 1.0
    session = {str(index): 2.0 for index in range(12)}
    session["only_session"] = 1.0
    pool = {str(index) for index in range(12)} | {"gold", "known", "only_exact", "only_session"}
    first = graph_hard_candidate_event(event, exact, session, pool, study_config()["study"])
    second = graph_hard_candidate_event(event, exact, session, pool, study_config()["study"])
    assert first == second
    assert first is not None
    assert len(first["candidate_item_ids"]) == len(set(first["candidate_item_ids"])) == 10
    assert set(first["negative_item_ids"]) <= set(exact) & set(session)
    assert "gold" in first["candidate_item_ids"]
    assert "gold" not in exact and "gold" not in session


def test_headroom_gate_compares_ppr_directly_against_one_step():
    candidates = [str(index) for index in range(10)]
    event = {
        "user_id": "u",
        "timestamp": 100,
        "gold_item_id": "9",
        "candidate_item_ids": candidates,
        "negative_item_ids": candidates[:9],
    }
    one = {item_id: float(10 - index) for index, item_id in enumerate(candidates)}
    ppr = dict(one)
    ppr["9"] = 20.0
    row = {
        "event_key": "u:100",
        "candidate_item_ids": candidates,
        "views": {
            "exact": {"one_step_scores": one, "ppr_scores": ppr},
            "session_300s": {"one_step_scores": one, "ppr_scores": ppr},
        },
    }
    result = headroom_metrics([event], [row], study_config())
    assert result["gate"]["exact"] == "pass"
    assert result["gate"]["session_300s"] == "pass"
    assert result["gate"]["decision"] == "promote_both_views"
    assert result["coverage"]["exact"]["negative_one_step_rate"] == 1.0
