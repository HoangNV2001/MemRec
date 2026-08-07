"""
jsonl -> ``datasets.Dataset`` for TRL.

TRL's ``GRPOTrainer`` needs a ``prompt`` column and passes every other column
through to the reward function. That is exactly the layout ``build_dataset.py``
writes, so loading is thin -- the work here is the *filters*, which are the
curriculum and dynamic-sampling knobs from RL_PLAN.md §6.4.

``datasets`` is imported lazily so the M1 CPU tests (and the reward unit tests at
M2) do not require the HF stack to be installed.
"""
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import json

# Columns TRL must not see as model input, but the reward function needs.
REWARD_COLUMNS = (
    "user_id", "candidates", "gold_item_id", "M_u", "neighbors", "neighbor_snippets",
    "r_null", "baseline_h1", "instruction", "candidate_titles",
    "candidate_memories", "neighbor_ids", "n_train_items",
)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_records(records: Sequence[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def filter_by_difficulty(
    records: Sequence[Dict[str, Any]],
    lo: float = 0.2,
    hi: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Curriculum band of RL_PLAN.md §6.4: keep only users whose baseline difficulty
    sits in [lo, hi]. Users the frozen ranker already nails, or never gets, give no
    useful gradient.

    Difficulty is read from ``baseline_p_gold`` -- the frozen ranker's probability
    on the gold candidate with no collaborative memory -- **not** from
    ``baseline_h1``.

    Why not ``baseline_h1``, which is what §6.4 names. The ranker is frozen and
    deterministic, so a user's Hit@1 is either 0.0 or 1.0; no user can ever land
    inside a band like [0.2, 0.8], and filtering on it would return an **empty**
    training set. That failure is silent and would only surface after a rented
    session had already been paid for. ``p_gold`` is continuous and expresses the
    same intent ("drop the users the ranker already nails and the ones it never
    gets"), so the band means what §6.4 wanted it to mean. See M2 Part B in
    docs/RESULTS.md.

    Records with no difficulty field yet (before the M2 back-fill) are kept
    untouched -- filtering on a field that does not exist would empty the set for
    a different reason.
    """
    out = []
    for r in records:
        score = r.get("baseline_p_gold")
        if score is None and r.get("baseline_h1") is not None:
            raise ValueError(
                "records carry 'baseline_h1' but not 'baseline_p_gold', so the "
                "curriculum band would be applied to a binary field and drop every "
                "user. Re-run: python -m src.rl.backfill_baselines"
            )
        if score is None or lo <= float(score) <= hi:
            out.append(r)
    return out


def load_dataset(
    path: str,
    difficulty_band: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
):
    """
    Load one split as a ``datasets.Dataset``.

    Args:
        path: jsonl written by ``src.rl.build_dataset``.
        difficulty_band: ``(lo, hi)`` curriculum band, or None for no filtering.
        limit: keep only the first N records (smoke runs).
    """
    from datasets import Dataset  # lazy: keeps CPU-only M1 tests dependency-free

    records = load_records(path)
    if difficulty_band is not None:
        records = filter_by_difficulty(records, *difficulty_band)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"no records left after filtering: {path}")
    return Dataset.from_list(records)


def backfill(path: str, values: Dict[int, Dict[str, Any]]) -> int:
    """
    Write ``r_null`` / ``baseline_h1`` into an existing jsonl in place.

    ``values`` maps user_id -> partial record. Used by the single batch job after
    M2 (RL_PLAN.md M1 + M2-B), so the expensive fields are computed once.
    """
    records = load_records(path)
    n = 0
    for r in records:
        patch = values.get(int(r["user_id"]))
        if patch:
            r.update(patch)
            n += 1
    write_records(records, path)
    return n
