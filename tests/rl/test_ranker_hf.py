"""
Exercise the real `hf` ranker path against a tiny model on CPU.

Skipped unless a model is available offline, so it never blocks the normal test
run. Its job is the one thing the stub cannot cover: that the *actual* forward
pass used on the GPU is wired correctly. In particular that we never materialise
full-sequence logits -- at batch 64 x 3072 tokens x 151936 vocab that is 60 GB in
bf16, which OOMs before the first reward exists.

    HF_TEST_MODEL=hf-internal-testing/tiny-random-Qwen2ForCausalLM \
        pytest tests/rl/test_ranker_hf.py -q
"""
import os

import pytest

from src.rl.reward.ranker import LETTERS, FrozenRanker

MODEL = os.environ.get("HF_TEST_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL, reason="set HF_TEST_MODEL to a small causal LM to run the hf path"
)

CANDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
TITLES = {str(c): f"Book {c}" for c in CANDS}


@pytest.fixture(scope="module")
def ranker():
    return FrozenRanker(mode="hf", model_name=MODEL, device="cpu")


def test_hf_ranker_produces_a_total_ranking(ranker):
    out = ranker.score(CANDS, TITLES, m_collab=[{"facet": "fantasy", "confidence": 0.9}])
    assert sorted(out.ranking) == sorted(CANDS)
    assert sum(out.scores.values()) == pytest.approx(1.0, abs=1e-5)


def test_letter_tokens_are_distinct(ranker):
    """A collision would make two candidates share a logit and silently corrupt
    every reward for the rest of the run."""
    ranker._ensure_loaded()
    ids = ranker._letter_token_ids
    assert len(ids) == len(LETTERS)
    assert len(set(ids)) == len(ids)


def test_last_position_logits_shape_is_batch_by_vocab(ranker):
    """The whole point of the slice: (B, V), never (B, S, V)."""
    import torch

    ranker._ensure_loaded()
    texts = ["hello world", "a much longer prompt that pads differently"]
    batch = ranker._tokenizer(texts, return_tensors="pt", padding=True).to("cpu")
    with torch.no_grad():
        logits = ranker._last_position_logits(batch)
    assert logits.dim() == 2
    assert logits.shape[0] == 2
    assert logits.shape[1] == ranker._model.config.vocab_size


def test_left_padding_so_last_index_is_the_real_final_token(ranker):
    """
    With right padding the final index would be a pad token for every short prompt
    and the reward would be read off padding. Batched and unbatched must agree.
    """
    ranker._ensure_loaded()
    assert ranker._tokenizer.padding_side == "left"

    short = dict(candidates=CANDS, candidate_titles=TITLES, user_id=1)
    long = dict(candidates=CANDS, candidate_titles=TITLES, user_id=2,
                m_collab=[{"facet": "x " * 200, "confidence": 0.5}])

    alone = ranker.score(**short)
    together = ranker.score_batch([short, long])[0]
    assert alone.ranking == together.ranking
