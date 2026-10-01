"""Synthetic requests for infrastructure checks without Books outcome labels."""

import random

from .request import RankRequest


def synthetic_rank_requests(count: int = 20, seed: int = 42) -> list[RankRequest]:
    if not 20 <= count <= 30:
        raise ValueError("Infrastructure smoke requires 20–30 fixtures")
    topics = ["astronomy", "history", "cooking", "gardening", "mathematics",
              "poetry", "travel", "computing", "biography", "mystery"]
    requests = []
    for index in range(count):
        order = list(range(10))
        random.Random(seed + index).shuffle(order)
        requests.append(RankRequest.from_stage_rr_inputs(
            user_id=index,
            instruction="Rank the candidate books by the user's preferences.",
            candidates=[{"id": item, "title": f"An introduction to {topics[item]}"}
                        for item in order],
            item_mems={item: f"A book about {topics[item]} for general readers."
                       for item in order},
            retrieval_bundle={"facets": [{
                "facet": f"Interest in {topics[index % 10]}",
                "confidence": 0.8, "supporting_neighbors": [],
            }]},
            snapshot_id="synthetic-format-smoke-no-outcome-labels",
            upstream_empty_facets_prompt=True,
        ))
    return requests
