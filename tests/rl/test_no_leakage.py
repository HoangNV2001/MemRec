"""
M1 DoD: the RL prompt must not contain the answer.

RL_PLAN.md §7 M1: "grep gold item title/id trong `prompt` ra 0 kết quả trên toàn
bộ train set", plus a coded assertion that the splits are user-disjoint.

Three separate leakage channels are checked, because they fail for different
reasons and a single "no leakage" assertion would hide which one broke:

1. **gold item id** -- would mean the frozen graph or the packer put the held-out
   item into the neighbour table.
2. **gold item title** -- same, but via text rather than id (also catches
   duplicate catalogue entries for the same book).
3. **candidate ids / instruction** -- would break candidate-blindness (§5.3). The
   InstructRec instruction paraphrases the target book, so it must stay on the
   Stage-ReRank side only.

Runs in seconds on CPU; no API, no GPU.
"""
from pathlib import Path

import json
import re

import pytest

from src.rl.dataset import load_records
from src.rl.splits import assert_disjoint, load_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_FILE = PROJECT_ROOT / "data/rl/user_splits_books.json"
PREFIX = PROJECT_ROOT / "data/rl/stager_books"
SPLIT_NAMES = ("train", "val", "test")

# Titles shorter than this are too generic for a substring test to be meaningful
# ("Bones", "Nan", ...); the id check still covers those records.
MIN_TITLE_CHARS = 25

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NON_WORD.sub(" ", str(text).lower()).strip()


def _split_path(name: str) -> Path:
    return Path(f"{PREFIX}_{name}.jsonl")


def _require(name: str):
    path = _split_path(name)
    if not path.exists():
        pytest.skip(f"{path} not built yet — run scripts/rl/01_build_dataset.sh")
    return load_records(str(path))


# --- splits -----------------------------------------------------------------

def test_splits_are_user_disjoint():
    if not SPLITS_FILE.exists():
        pytest.skip("splits not built yet — run scripts/rl/00_build_snapshot.sh")
    splits = load_splits(str(SPLITS_FILE))
    assert_disjoint(splits)          # raises with details on overlap
    assert set(splits) == set(SPLIT_NAMES)
    for name in SPLIT_NAMES:
        assert len(splits[name]) == len(set(splits[name])), f"{name} has duplicate user ids"


def test_jsonl_users_match_splits_and_are_disjoint():
    per_split = {name: _require(name) for name in SPLIT_NAMES}
    ids = {name: {r["user_id"] for r in recs} for name, recs in per_split.items()}
    for name, recs in per_split.items():
        assert len(ids[name]) == len(recs), f"{name}.jsonl has duplicate user_id rows"
    assert_disjoint({k: sorted(v) for k, v in ids.items()})


# --- leakage ----------------------------------------------------------------

def _id_in(prompt: str, item_id: int) -> bool:
    """``Item-2125`` must not match ``Item-21254``; guard the trailing digits."""
    return re.search(rf"Item-{item_id}(?![0-9])", prompt) is not None


@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_gold_item_id_absent_from_prompt(split):
    violations = [
        (r["user_id"], r["gold_item_id"]) for r in _require(split)
        if _id_in(r["prompt"], r["gold_item_id"])
    ]
    assert not violations, f"{split}: gold item id in prompt for {len(violations)} users: {violations[:5]}"


@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_gold_item_title_absent_from_prompt(split):
    records = _require(split)
    violations, checked = [], 0
    for r in records:
        title = r["candidate_titles"].get(str(r["gold_item_id"]), "")
        norm_title = _norm(title)
        if len(norm_title) < MIN_TITLE_CHARS:
            continue
        checked += 1
        if norm_title in _norm(r["prompt"]):
            violations.append((r["user_id"], title[:80]))
    assert not violations, (
        f"{split}: gold title appears in prompt for {len(violations)}/{checked} checked users: "
        f"{violations[:3]}"
    )
    assert checked > 0, f"{split}: no title was long enough to check — test is vacuous"


@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_prompt_is_candidate_blind(split):
    """No candidate id may appear in the prompt — not just the gold one (§5.3)."""
    violations = []
    for r in _require(split):
        leaked = [c for c in r["candidates"] if _id_in(r["prompt"], c)]
        if leaked:
            violations.append((r["user_id"], leaked))
    assert not violations, f"{split}: candidate ids in prompt for {len(violations)} users: {violations[:5]}"


@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_instruction_absent_from_prompt(split):
    """
    The InstructRec instruction paraphrases the target book. It belongs to
    Stage-ReRank; if it reached Stage-R the policy could copy the answer.
    """
    violations = []
    for r in _require(split):
        instruction = _norm(r.get("instruction", ""))
        if len(instruction) < MIN_TITLE_CHARS:
            continue
        if instruction[:120] in _norm(r["prompt"]):
            violations.append(r["user_id"])
    assert not violations, f"{split}: instruction leaked into prompt for users {violations[:5]}"


# --- record shape -----------------------------------------------------------

@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_record_schema(split):
    required = {
        "user_id", "prompt", "candidates", "gold_item_id", "M_u", "neighbors",
        "r_null", "baseline_h1",
    }
    for r in _require(split):
        missing = required - set(r)
        assert not missing, f"{split}: user {r['user_id']} missing {missing}"
        assert r["gold_item_id"] in r["candidates"], (
            f"{split}: user {r['user_id']} gold item not among its candidates"
        )
        assert len(r["candidates"]) == len(set(r["candidates"])), (
            f"{split}: user {r['user_id']} has duplicate candidates"
        )
        assert r["prompt"].strip(), f"{split}: user {r['user_id']} has empty prompt"


@pytest.mark.slow
def test_candidates_are_reproducible_from_the_dataset():
    """
    Re-deriving candidates from the real negative pool must reproduce the stored
    lists exactly -- same items, same order.

    This is the guard against the M0 bug (docs/RESULTS.md #5) where negatives were
    seeded by thread id, so no two runs scored the same exam. It has to go through
    ``RecDataset`` because the draw depends on the full pool, not just on the nine
    negatives that happened to be selected.
    """
    inter = PROJECT_ROOT / "data/processed/instructrec-books/instructrec-books.inter"
    if not inter.exists():
        pytest.skip("processed dataset not available")

    from src.data import RecDataset
    from src.rl.splits import build_candidates_for_users

    records = {r["user_id"]: r for r in _require("test")[:100]}
    dataset = RecDataset(str(inter), seed=42)
    regenerated = build_candidates_for_users(dataset, list(records), n_candidates=10, seed=42)

    for user_id, r in records.items():
        assert regenerated[user_id] == r["candidates"], (
            f"user {user_id}: candidates are not reproducible\n"
            f"  stored:      {r['candidates']}\n"
            f"  regenerated: {regenerated[user_id]}"
        )


def test_no_record_survives_the_leakage_screen():
    """
    Cross-check against src/rl/leakage.py, the screen build_dataset applies.

    The assertions above re-implement the checks independently; this one makes
    sure the shipped screen and the tests agree on the result.
    """
    from src.rl.leakage import gold_leak_reason

    for split in SPLIT_NAMES:
        offenders = [
            (r["user_id"], gold_leak_reason(r)) for r in _require(split)
            if gold_leak_reason(r)
        ]
        assert not offenders, f"{split}: {len(offenders)} leaked records survived: {offenders[:5]}"
