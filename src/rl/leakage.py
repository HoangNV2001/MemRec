"""
Gold-answer leakage screening for RL records.

The frozen graph is built from ``train_data`` only, so a user's held-out test item
can never be their own neighbour by id. It can still reach the prompt by *title*:
the Books catalogue contains duplicate entries for the same book (different
``item_id``, different edition or format), and if a user has one of those in their
history, the other one's title is printed in the neighbour table.

That is a real leak -- the policy could write a facet naming the answer without
ever "seeing" the answer -- so contaminated users are dropped at dataset-build
time. Measured on instructrec-books: 23/2350 users (~1%), all via the neighbour
table, none via M_u.

``tests/rl/test_no_leakage.py`` re-implements this check independently and asserts
zero survivors, so a regression in the packer or pruner still trips the DoD.
"""
from typing import Dict, Optional

import re

# Titles shorter than this are too generic for substring matching to mean anything
# ("Bones", "Nan"); the id check still covers those records.
MIN_TITLE_CHARS = 25

_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _NON_WORD.sub(" ", str(text).lower()).strip()


def _id_present(prompt: str, item_id: int) -> bool:
    """``Item-2125`` must not match ``Item-21254`` -- guard the trailing digits."""
    return re.search(rf"Item-{item_id}(?![0-9])", prompt) is not None


def gold_leak_reason(record: Dict) -> Optional[str]:
    """
    Return a short description of how ``record['prompt']`` leaks the answer, or
    None if it is clean.

    Checks, in order of severity:
      * gold item id printed in the prompt (would mean the frozen graph is wrong);
      * any candidate id printed (would break candidate-blindness, §5.3);
      * gold item title printed (the duplicate-catalogue channel above).
    """
    prompt = record.get("prompt", "")
    gold = record.get("gold_item_id")

    if gold is not None and _id_present(prompt, int(gold)):
        return "gold_item_id in prompt"

    for cand in record.get("candidates", []):
        if _id_present(prompt, int(cand)):
            return f"candidate id {cand} in prompt"

    title = record.get("candidate_titles", {}).get(str(gold), "")
    norm_title = normalize(title)
    if len(norm_title) >= MIN_TITLE_CHARS and norm_title in normalize(prompt):
        return "gold item title in prompt (duplicate catalogue entry)"

    return None
