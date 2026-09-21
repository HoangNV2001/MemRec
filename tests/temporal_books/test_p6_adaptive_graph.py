import numpy as np

from src.temporal_books.p6_adaptive_graph import (
    FEATURE_NAMES,
    adaptive_ranking,
    feature_vector,
    first_novel_positive_targets,
    fit_pairwise_model,
)


def graph_row(shallow_score=0.0, deep_score=0.0):
    shallow_paths = int(shallow_score > 0)
    deep_paths = shallow_paths + int(deep_score > shallow_score)
    return {
        "layer3": {
            "score": shallow_score,
            "best_path_score": shallow_score,
            "selected_paths": shallow_paths,
            "independent_anchors": shallow_paths,
            "candidate_peers": shallow_paths,
            "min_item_layers": 3 if shallow_score else None,
        },
        "layer5": {
            "score": deep_score,
            "best_path_score": deep_score,
            "selected_paths": deep_paths,
            "independent_anchors": deep_paths,
            "candidate_peers": deep_paths,
            "min_item_layers": 3 if shallow_score else (5 if deep_score else None),
        },
    }


def test_features_separate_shallow_and_deep_increment():
    shallow = feature_vector(graph_row(0.2, 0.2))
    deep_only = feature_vector(graph_row(0.0, 0.2))
    assert len(shallow) == len(FEATURE_NAMES)
    assert shallow[0] == 1.0 and shallow[6] == 0.0
    assert deep_only[0] == 0.0 and deep_only[6] == 1.0
    assert np.isfinite(shallow).all()


def test_pairwise_model_learns_shallow_support():
    events = {}
    rows = []
    for index in range(20):
        key = f"u{index}:1"
        events[key] = {"user_id": f"u{index}", "timestamp": 1, "gold_item_id": "g", "candidate_item_ids": ["g", "n"]}
        rows.append(
            {
                "event_key": key,
                "candidate_item_ids": ["g", "n"],
                "candidate_features": {"g": graph_row(0.2, 0.2), "n": graph_row(0.0, 0.2)},
            }
        )
    model = fit_pairwise_model(rows, events, {"pairwise_ridge_l2": 0.1, "pairwise_newton_steps": 50})
    assert model["weights"][0] > 0
    assert model["training_pairs"] == 20


def test_margin_gate_falls_back_to_local_and_can_activate():
    candidates = ["a", "b", "c"]
    local = ["a", "b", "c"]
    inactive, _, used = adaptive_ranking(candidates, local, {"a": 0.0, "b": 0.2, "c": 0.1}, alpha=1.0, margin_gate=0.5)
    assert inactive == local and not used
    active, _, used = adaptive_ranking(candidates, local, {"a": 0.0, "b": 2.0, "c": 0.1}, alpha=1.0, margin_gate=0.5)
    assert active[0] == "b" and used


def test_target_selection_does_not_count_same_timestamp_as_history():
    histories = {
        "bad": [(10, f"i{index}", 5.0) for index in range(6)],
        "good": [(index, f"h{index}", 5.0) for index in range(1, 7)] + [(10, "gold", 5.0)],
    }
    values = first_novel_positive_targets(
        histories,
        start=10,
        end=None,
        candidate_pool={"gold", *{f"i{index}" for index in range(6)}},
        minimum_history=5,
        excluded_users=set(),
    )
    assert values == [{"user_id": "good", "timestamp": 10, "gold_item_id": "gold"}]
