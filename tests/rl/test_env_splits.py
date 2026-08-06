"""
Unit tests for the frozen-environment plumbing: deterministic sampling, split
construction, and snapshot round-tripping.

No dataset files, no API, no GPU — everything runs against small fakes so this
stays inside the "<5 min smoke test" rule (RL_PLAN.md §10.5).
"""
import numpy as np
import pytest

from src.rl.env import GraphSnapshot, UserState, neighbor_id_strings
from src.rl.splits import (
    assert_disjoint,
    build_splits,
    eligible_users,
    sample_candidates,
    user_rng,
)


class FakeDataset:
    """Minimal stand-in for RecDataset covering only what splits.py touches."""

    def __init__(self, n_users=50, n_items=500):
        self.n_items = n_items
        self.train_data = {u: list(range(u, u + 5)) for u in range(n_users)}
        self.valid_data = {u: n_items - 2 - u for u in range(n_users)}
        self.test_data = {u: n_items - 1 - u for u in range(n_users)}
        self.user_negatives = {u: np.arange(100, 400) for u in range(n_users)}

    def get_user_all_items(self, user_id):
        return self.train_data[user_id] + [self.valid_data[user_id], self.test_data[user_id]]


# --- determinism ------------------------------------------------------------

def test_candidates_are_deterministic_across_calls():
    pool = list(range(1000, 2000))
    a = sample_candidates(7, target_item=42, negative_pool=pool)
    b = sample_candidates(7, target_item=42, negative_pool=pool)
    assert a == b


def test_candidates_differ_across_users():
    pool = list(range(1000, 2000))
    a = sample_candidates(7, target_item=42, negative_pool=pool)
    b = sample_candidates(8, target_item=42, negative_pool=pool)
    assert a != b, "per-user streams must be decorrelated"


def test_candidates_contain_target_exactly_once():
    pool = list(range(1000, 2000))
    cands = sample_candidates(7, target_item=42, negative_pool=pool, n_candidates=10)
    assert len(cands) == 10
    assert cands.count(42) == 1
    assert len(set(cands)) == 10


def test_candidates_are_shuffled_not_target_first():
    """The M0 code shuffles for a reason; regression guard on position bias."""
    pool = list(range(1000, 2000))
    positions = [
        sample_candidates(u, target_item=42, negative_pool=pool).index(42)
        for u in range(200)
    ]
    assert len(set(positions)) > 5, "target position is not being varied"


def test_warmup_and_eval_draw_different_negatives():
    """
    Warm-up's Stage-R sees the candidate block, so its distractors end up shaping
    M_u. If the eval list reused them, the frozen state would be conditioned on the
    items it is later scored against.
    """
    from src.rl.splits import EVAL_CANDIDATE_SALT, WARMUP_CANDIDATE_SALT

    pool = list(range(1000, 3000))
    warm = sample_candidates(7, target_item=42, negative_pool=pool, salt=WARMUP_CANDIDATE_SALT)
    evl = sample_candidates(7, target_item=99, negative_pool=pool, salt=EVAL_CANDIDATE_SALT)
    overlap = (set(warm) - {42}) & (set(evl) - {99})
    assert len(overlap) <= 1, f"warm-up and eval negatives overlap on {sorted(overlap)}"


def test_salt_is_deterministic():
    pool = list(range(1000, 2000))
    a = sample_candidates(7, target_item=42, negative_pool=pool, salt=1)
    b = sample_candidates(7, target_item=42, negative_pool=pool, salt=1)
    assert a == b


def test_candidates_reject_undersized_pool():
    with pytest.raises(ValueError):
        sample_candidates(1, target_item=0, negative_pool=[1, 2], n_candidates=10)


def test_user_rng_is_independent_of_call_order():
    first = user_rng(123).randint(0, 10_000, size=5).tolist()
    _ = user_rng(999).randint(0, 10_000, size=50)
    second = user_rng(123).randint(0, 10_000, size=5).tolist()
    assert first == second


# --- splits -----------------------------------------------------------------

def test_build_splits_is_disjoint_and_sized():
    ds = FakeDataset(n_users=50)
    splits = build_splits(ds, test_user_ids=[0, 1, 2, 3, 4], n_train=20, n_val=5)
    assert len(splits["test"]) == 5
    assert len(splits["train"]) == 20
    assert len(splits["val"]) == 5
    assert_disjoint(splits)


def test_build_splits_is_reproducible():
    ds = FakeDataset(n_users=50)
    kwargs = dict(test_user_ids=[0, 1, 2], n_train=10, n_val=5)
    assert build_splits(ds, **kwargs) == build_splits(ds, **kwargs)


def test_build_splits_raises_when_not_enough_users():
    ds = FakeDataset(n_users=20)
    with pytest.raises(ValueError, match="eligible"):
        build_splits(ds, test_user_ids=[0, 1], n_train=100, n_val=50)


def test_assert_disjoint_detects_overlap():
    with pytest.raises(AssertionError, match="overlap"):
        assert_disjoint({"train": [1, 2, 3], "val": [3, 4], "test": [5]})


def test_eligible_users_excludes_empty_history():
    ds = FakeDataset(n_users=10)
    ds.train_data[3] = []
    assert 3 not in eligible_users(ds)


# --- snapshot ---------------------------------------------------------------

def test_neighbor_id_strings_use_repo_convention():
    pruned = {"neighbors": [{"type": "item", "id": 5}, {"type": "user", "id": 9}]}
    assert neighbor_id_strings(pruned) == ["Item-5", "User-9"]


def test_snapshot_roundtrip(tmp_path):
    snap = GraphSnapshot(meta={"dataset": "test"})
    snap.users[1] = UserState(1, "likes fantasy", "1. [Item-5] Dune", ["Item-5"], 12)
    snap.item_memories[5] = "A desert epic."

    path = tmp_path / "snap.json"
    snap.save(str(path))
    loaded = GraphSnapshot.load(str(path))

    assert len(loaded) == 1
    assert loaded.state(1).user_memory == "likes fantasy"
    assert loaded.state(1).neighbor_ids == ["Item-5"]
    assert loaded.state(1).n_train_items == 12
    assert loaded.item_memory(5) == "A desert epic."
    assert loaded.item_memory(999) == ""
    assert loaded.meta["dataset"] == "test"


def test_neighbour_budget_matches_original_stage_r():
    """
    The candidate-blind state must not get a bigger neighbour budget than the
    prompted Stage-R baseline, or "GRPO > prompted" is confounded by context size.
    """
    from src.memory import SnippetPacker
    from src.rl.env import CANDIDATE_BLOCK_RESERVE

    def budget(packer, has_candidates):
        return (packer.tau
                - (CANDIDATE_BLOCK_RESERVE if has_candidates else 0)
                - 200   # user memory reserve
                - 600)  # output reserve

    original = SnippetPacker(tau=1800)
    rl_side = SnippetPacker(tau=1800 - CANDIDATE_BLOCK_RESERVE)
    assert budget(rl_side, has_candidates=False) == budget(original, has_candidates=True)


def test_snapshot_rejects_wrong_version(tmp_path):
    import json

    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"version": 999, "users": {}, "item_memories": {}}))
    with pytest.raises(ValueError, match="version"):
        GraphSnapshot.load(str(path))
