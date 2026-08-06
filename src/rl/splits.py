"""
Deterministic user splits and candidate sampling for the RL environment.

Two things in here exist specifically to fix reproducibility problems that the
original eval path has (see docs/RESULTS.md, bug #5):

1. Candidate negatives are sampled from a ``RandomState`` seeded by
   ``(seed, user_id)`` -- never by thread id. The same user always gets the same
   10 candidates regardless of worker count, run order, or process.
2. The train/val/test user split is derived by sorting + seeded shuffling, so it
   is stable across machines.

Nothing here touches the original eval path (RL_PLAN.md §10.4).
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import json
import numpy as np

# Multiplier used to decorrelate per-user streams. Any large odd number works;
# fixed here so the candidate sets are reproducible forever.
_USER_SEED_MULT = 1_000_003


def user_rng(user_id: int, seed: int = 42, salt: int = 0) -> np.random.RandomState:
    """Per-user deterministic RNG. Independent of threading and iteration order."""
    return np.random.RandomState((seed * _USER_SEED_MULT + user_id * 7919 + salt) % (2 ** 32))


# Warm-up and evaluation must not draw the same distractors. Warm-up's Stage-R
# sees the candidate block, so its negatives leak into the facets that Stage-W
# turns into M_u; if the eval list reused them, the frozen state would be shaped
# by the very items it is later scored against. Different salts, independent draws.
WARMUP_CANDIDATE_SALT = 0
EVAL_CANDIDATE_SALT = 1


def sample_candidates(
    user_id: int,
    target_item: int,
    negative_pool: Sequence[int],
    n_candidates: int = 10,
    seed: int = 42,
    salt: int = WARMUP_CANDIDATE_SALT,
) -> List[int]:
    """
    Build one candidate list: 1 positive + (n_candidates - 1) negatives, shuffled.
    Deterministic given (user_id, target_item, negative_pool, seed, salt).

    Raises ValueError if the pool is too small -- callers should filter such users
    out of the split rather than silently producing a short list.
    """
    pool = np.asarray(negative_pool)
    n_neg = n_candidates - 1
    if len(pool) < n_neg:
        raise ValueError(
            f"user {user_id}: negative pool has {len(pool)} items, need {n_neg}"
        )

    rng = user_rng(user_id, seed=seed, salt=salt)
    negatives = rng.choice(pool, size=n_neg, replace=False).tolist()
    candidates = [int(target_item)] + [int(x) for x in negatives]
    rng.shuffle(candidates)
    return candidates


def build_candidates_for_users(
    dataset,
    user_ids: Sequence[int],
    n_candidates: int = 10,
    seed: int = 42,
    salt: int = EVAL_CANDIDATE_SALT,
) -> Dict[int, List[int]]:
    """
    Deterministic candidate list per user, for the *test* item.

    Shared by the snapshot builder (which needs to know which item memories to
    freeze) and the dataset builder (which writes them into the jsonl), so both
    see the identical lists. Salted away from the warm-up draw by default.
    """
    out: Dict[int, List[int]] = {}
    for user_id in user_ids:
        user_id = int(user_id)
        target = dataset.test_data.get(user_id)
        if target is None:
            continue
        positives = set(dataset.get_user_all_items(user_id))
        pool = [i for i in dataset.user_negatives.get(user_id, []) if i not in positives]
        try:
            out[user_id] = sample_candidates(
                user_id=user_id,
                target_item=int(target),
                negative_pool=pool,
                n_candidates=n_candidates,
                seed=seed,
                salt=salt,
            )
        except ValueError:
            continue
    return out


def eligible_users(dataset, min_train_items: int = 1) -> List[int]:
    """
    Users usable as RL environment states: they have a held-out test item and a
    non-empty training history (otherwise the graph gives them no neighbours).

    Returned sorted, so downstream shuffling is deterministic.
    """
    users = []
    for user_id in dataset.test_data:
        if len(dataset.train_data.get(user_id, [])) >= min_train_items:
            users.append(int(user_id))
    return sorted(users)


def build_splits(
    dataset,
    test_user_ids: Sequence[int],
    n_train: int = 1200,
    n_val: int = 150,
    seed: int = 42,
    min_train_items: int = 1,
) -> Dict[str, List[int]]:
    """
    Build user-disjoint splits.

    ``test_user_ids`` is pinned from outside (we reuse the M0 1k eval sample so the
    RL test set and the M0 baseline table cover the same users). Train and val are
    drawn from the remaining eligible users.
    """
    pinned_test = [int(u) for u in test_user_ids]
    eligible = eligible_users(dataset, min_train_items=min_train_items)
    eligible_set = set(eligible)

    test = [u for u in pinned_test if u in eligible_set]
    remaining = sorted(eligible_set - set(test))

    need = n_train + n_val
    if len(remaining) < need:
        raise ValueError(
            f"need {need} non-test users for train+val, only {len(remaining)} eligible"
        )

    rng = np.random.RandomState(seed)
    picked = rng.choice(remaining, size=need, replace=False)
    picked = [int(x) for x in picked]

    splits = {
        "train": sorted(picked[:n_train]),
        "val": sorted(picked[n_train:]),
        "test": sorted(test),
    }
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: Dict[str, List[int]]) -> None:
    """Hard assertion that no user appears in two splits (M1 DoD)."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                raise AssertionError(
                    f"splits '{a}' and '{b}' overlap on {len(overlap)} users, "
                    f"e.g. {sorted(overlap)[:5]}"
                )


def save_splits(splits: Dict[str, List[int]], path: str, meta: Optional[Dict] = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "splits": {k: list(v) for k, v in splits.items()}}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {k: [int(u) for u in v] for k, v in payload["splits"].items()}
