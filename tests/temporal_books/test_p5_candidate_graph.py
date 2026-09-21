from src.temporal_books.p5_candidate_graph import aggregate_paths, candidate_paths, residual_ranking


def test_five_layers_finds_path_that_three_layers_cannot_reach():
    histories = {
        "source": [(10, "anchor", 5.0)],
        "u1": [(11, "anchor", 5.0), (12, "i2", 5.0)],
        "u2": [(13, "i2", 5.0), (14, "i3", 5.0)],
        "u3": [(15, "i3", 5.0), (16, "i4", 5.0)],
        "u4": [(17, "i4", 5.0), (18, "candidate", 5.0)],
    }
    item_events = {}
    for user, events in histories.items():
        for timestamp, item, rating in events:
            item_events.setdefault(item, []).append((timestamp, user, rating))
    for events in item_events.values():
        events.sort()
    paths = candidate_paths(
        histories,
        item_events,
        source_user="source",
        target_time=100,
        candidate_item="candidate",
        minimum_rating=4.0,
        half_life_days=730.0,
        length_decay=0.7,
        hub_penalty_weight=0.5,
        source_anchor_limit=6,
        peers_per_item=8,
        items_per_peer=8,
        beam_width=64,
        max_item_layers=5,
    )
    assert aggregate_paths(paths, max_item_layers=3, max_paths=8)["score"] == 0
    assert aggregate_paths(paths, max_item_layers=5, max_paths=8)["score"] > 0


def test_residual_graph_score_can_promote_supported_candidate():
    candidates = ["a", "b", "c"]
    ranking = residual_ranking(candidates, ["a", "b", "c"], {"a": 0.0, "b": 1.0, "c": 0.0}, alpha=1.0)
    assert ranking[0] == "b"
