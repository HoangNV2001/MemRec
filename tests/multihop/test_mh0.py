import csv
import json
from pathlib import Path

import pytest

from src.multihop import mh0


def write_interactions(path: Path) -> None:
    rows = [
        {"user_id": 1, "item_id": 10, "rating": 5, "timestamp": 1},
        {"user_id": 1, "item_id": 11, "rating": 5, "timestamp": 2},
        {"user_id": 1, "item_id": 12, "rating": 5, "timestamp": 3},
        {"user_id": 1, "item_id": 13, "rating": 5, "timestamp": 4},
        {"user_id": 2, "item_id": 10, "rating": 5, "timestamp": 1},
        {"user_id": 2, "item_id": 20, "rating": 5, "timestamp": 2},
        {"user_id": 2, "item_id": 21, "rating": 5, "timestamp": 3},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_pre_target_history_excludes_validation_and_test_items(tmp_path):
    inter = tmp_path / "books.inter"
    write_interactions(inter)

    user_items, item_users, edge_count = mh0.read_pre_target_history(inter)

    assert user_items == {1: [10, 11], 2: [10]}
    assert item_users == {10: [1, 2], 11: [1]}
    assert edge_count == 3


def test_control_row_keeps_ranking_values_outside_stage_r_context():
    snapshot = mh0.GraphSnapshot()
    snapshot.users[1] = mh0.UserState(
        user_id=1,
        user_memory="likes mystery",
        neighbors_text="**Collaborative Neighbors:**\n1. [Item-10] Mystery (score=1.0)",
        neighbor_ids=["Item-10"],
        n_train_items=4,
    )
    prompt = mh0.build_prompt(
        user_id=1,
        user_memory="likes mystery",
        neighbors_text=snapshot.users[1].neighbors_text,
        n_facets=7,
    )
    record = {
        "user_id": 1,
        "prompt": prompt,
        "candidates": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "gold_item_id": 100,
        "candidate_titles": {"100": "A sufficiently long target title that is not in the prompt"},
        "instruction": "rank books",
        "candidate_memories": {"100": "target memory"},
    }

    row = mh0.control_row("val", record, snapshot)

    assert "candidates" not in row["stage_r_context"]
    assert "gold_item_id" not in row["stage_r_context"]
    assert "instruction" not in row["stage_r_context"]
    assert row["ranking_context"]["gold_item_id"] == 100
    assert row["stage_r_context"]["K_u"] == 1
    assert row["stage_r_context"]["T_u"] > 0


def test_known_legacy_input_profile_is_recognized():
    assert mh0.input_profile(dict(mh0.KNOWN_INPUT_PROFILES["legacy_pre_m2_backfill"])) == (
        "legacy_pre_m2_backfill"
    )

