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
    "user_id", "candidates", "gold_item_id", "M_u", "neighbors",
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

    Records with ``baseline_h1`` still null (i.e. before the M2 back-fill) are kept
    untouched -- filtering on a field that does not exist yet would silently empty
    the training set.
    """
    out = []
    for r in records:
        score = r.get("baseline_h1")
        if score is None:
            out.append(r)
        elif lo <= float(score) <= hi:
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
