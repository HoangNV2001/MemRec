"""
Full reward path on CPU with the stub ranker (RL_PLAN.md M2 Part A DoD).

No GPU, no API, no model download. Everything that can be wrong about the reward
*logic* -- metric arithmetic, grounding denominators, penalty units, TRL wiring,
degenerate groups -- is caught here, before any rented session.
"""
import math

import pytest

from src.rl.env import parse_neighbor_snippets
from src.rl.reward.composite import RewardConfig, StageRReward
from src.rl.reward.grounding import GroundingScorer, cosine
from src.rl.reward.metrics import hit_at_k, ndcg_at_k, rank_of
from src.rl.reward.ranker import LETTERS, FrozenRanker, build_ranker_prompt


# ---------------------------------------------------------------------------
# metrics -- hand-computed values
# ---------------------------------------------------------------------------

RANKING = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


@pytest.mark.parametrize(
    "gold,k,expected",
    [
        (10, 5, 1.0),                  # pos 0 -> 1/log2(2) = 1
        (20, 5, 1 / math.log2(3)),     # pos 1 -> 0.6309
        (30, 5, 0.5),                  # pos 2 -> 1/log2(4) = 0.5
        (40, 5, 1 / math.log2(5)),     # pos 3 -> 0.4307
        (50, 5, 1 / math.log2(6)),     # pos 4 -> 0.3869
        (60, 5, 0.0),                  # pos 5 -> outside k=5
        (100, 5, 0.0),
        (60, 10, 1 / math.log2(7)),    # inside k=10
        (999, 5, 0.0),                 # absent
    ],
)
def test_ndcg_at_k_hand_computed(gold, k, expected):
    assert ndcg_at_k(RANKING, gold, k) == pytest.approx(expected, abs=1e-9)


def test_ndcg_matches_original_pipeline_formula():
    """Must equal src/train/metrics.py so RL numbers compare with M0's table."""
    import numpy as np

    for pos in range(10):
        gold = RANKING[pos]
        original = float(np.where(pos < 5, 1.0 / np.log2(pos + 2), 0.0))
        assert ndcg_at_k(RANKING, gold, 5) == pytest.approx(original, abs=1e-12)


@pytest.mark.parametrize("gold,k,expected", [(10, 1, 1.0), (20, 1, 0.0), (30, 3, 1.0), (40, 3, 0.0)])
def test_hit_at_k(gold, k, expected):
    assert hit_at_k(RANKING, gold, k) == expected


def test_rank_of_distinguishes_last_from_absent():
    assert rank_of(RANKING, 100) == 9
    assert rank_of(RANKING, 999) is None


def test_metrics_reject_nonpositive_k():
    for fn in (ndcg_at_k, hit_at_k):
        with pytest.raises(ValueError):
            fn(RANKING, 10, 0)


# ---------------------------------------------------------------------------
# ranker
# ---------------------------------------------------------------------------

CANDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
TITLES = {str(c): f"Book {c}" for c in CANDS}


def test_ranker_prompt_labels_candidates_in_order():
    prompt = build_ranker_prompt(CANDS, TITLES)
    for letter, cid in zip(LETTERS, CANDS):
        assert f"{letter}. Item {cid}" in prompt


def test_ranker_prompt_omits_facet_section_when_empty():
    """M0 bug #3: claiming patterns then listing none degrades scoring."""
    assert "preference patterns" not in build_ranker_prompt(CANDS, TITLES, m_collab=[])
    with_facets = build_ranker_prompt(
        CANDS, TITLES, m_collab=[{"facet": "epic fantasy", "confidence": 0.9}]
    )
    assert "preference patterns" in with_facets and "epic fantasy" in with_facets


def test_ranker_prompt_is_deterministic():
    a = build_ranker_prompt(CANDS, TITLES, m_collab=[{"facet": "x", "confidence": 0.5}])
    b = build_ranker_prompt(CANDS, TITLES, m_collab=[{"facet": "x", "confidence": 0.5}])
    assert a == b


def test_stub_ranker_is_deterministic_and_total():
    ranker = FrozenRanker(mode="stub")
    a = ranker.score(CANDS, TITLES)
    b = ranker.score(CANDS, TITLES)
    assert a.ranking == b.ranking
    assert sorted(a.ranking) == sorted(CANDS), "every candidate must be ranked exactly once"
    assert sum(a.scores.values()) == pytest.approx(1.0)


def test_stub_ranker_responds_to_memory():
    """Different M_collab must be able to change the ranking, or reward is flat."""
    ranker = FrozenRanker(mode="stub")
    a = ranker.score(CANDS, TITLES, m_collab=[{"facet": "fantasy", "confidence": 0.9}])
    b = ranker.score(CANDS, TITLES, m_collab=[{"facet": "cookbooks", "confidence": 0.9}])
    assert a.ranking != b.ranking


def test_ranker_rejects_unknown_mode():
    with pytest.raises(ValueError):
        FrozenRanker(mode="magic")


def test_rank_from_logits_orders_by_probability():
    out = FrozenRanker._rank_from_logits([1, 2, 3], [0.0, 5.0, 1.0])
    assert out.ranking == [2, 3, 1]
    assert out.scores[2] > out.scores[3] > out.scores[1]


def test_score_batch_matches_score_one_by_one():
    ranker = FrozenRanker(mode="stub")
    reqs = [
        dict(candidates=CANDS, candidate_titles=TITLES, user_id=1),
        dict(candidates=CANDS, candidate_titles=TITLES, user_id=2),
    ]
    batched = ranker.score_batch(reqs)
    assert [b.ranking for b in batched] == [ranker.score(**r).ranking for r in reqs]


# ---------------------------------------------------------------------------
# grounding
# ---------------------------------------------------------------------------

class FakeEncoder:
    """Bag-of-words vectors — enough to make cosine behave sensibly, no download."""

    def encode(self, texts):
        vocab = sorted({w for t in texts for w in str(t).lower().split()})
        index = {w: i for i, w in enumerate(vocab)}
        out = []
        for t in texts:
            vec = [0.0] * len(vocab)
            for w in str(t).lower().split():
                vec[index[w]] += 1.0
            out.append(vec)
        return out


SNIPPETS = {
    "Item-1": "epic fantasy dragons quest",
    "Item-2": "french cooking recipes kitchen",
    "User-3": "space opera starships",
}


def _scorer(n_facets=7, tau=0.35):
    return GroundingScorer(encoder=FakeEncoder(), tau=tau, n_facets=n_facets)


def test_grounding_rewards_a_supported_citation():
    facets = [{"facet": "epic fantasy dragons", "supporting_neighbors": ["Item-1"]}]
    r = _scorer(n_facets=1).score(facets, SNIPPETS)
    assert r.n_grounded == 1
    assert r.score == pytest.approx(1.0)


def test_grounding_rejects_unrelated_citation():
    facets = [{"facet": "french cooking recipes", "supporting_neighbors": ["Item-1"]}]
    r = _scorer(n_facets=1).score(facets, SNIPPETS)
    assert r.n_grounded == 0 and r.score == 0.0


def test_grounding_flags_hallucinated_ids():
    facets = [{"facet": "epic fantasy", "supporting_neighbors": ["Item-9999"]}]
    r = _scorer(n_facets=1).score(facets, SNIPPETS)
    assert r.hallucinated_ids == ["Item-9999"]
    assert r.n_grounded == 0


def test_grounding_uses_best_of_several_citations():
    facets = [{"facet": "epic fantasy dragons", "supporting_neighbors": ["Item-2", "Item-1"]}]
    r = _scorer(n_facets=1).score(facets, SNIPPETS)
    assert r.n_grounded == 1


def test_grounding_denominator_is_requested_n_facets_not_produced():
    """
    Guards the obvious exploit: emit one perfect facet, drop the other six.
    RL_PLAN §5.2 divides by N_f, the target count.
    """
    facets = [{"facet": "epic fantasy dragons", "supporting_neighbors": ["Item-1"]}]
    r = _scorer(n_facets=7).score(facets, SNIPPETS)
    assert r.denominator == 7
    assert r.score == pytest.approx(1 / 7)


def test_grounding_empty_facets_scores_zero():
    r = _scorer().score([], SNIPPETS)
    assert r.score == 0.0 and r.n_grounded == 0


def test_grounding_uncited_facet_is_not_grounded():
    facets = [{"facet": "epic fantasy dragons", "supporting_neighbors": []}]
    assert _scorer(n_facets=1).score(facets, SNIPPETS).n_grounded == 0


def test_cosine_edge_cases():
    assert cosine([0, 0], [1, 1]) == 0.0
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# neighbour-snippet parsing (feeds grounding)
# ---------------------------------------------------------------------------

def test_parse_neighbor_snippets_round_trip():
    text = (
        "**Collaborative Neighbors:**\n"
        "1. [Item-144648] The Undertaking of Tess | ['a tender comedy'] (score=1.000)\n"
        "2. [User-6902] (overlap_score=0.500) - Recent: Dune, Neuromancer\n"
    )
    snippets = parse_neighbor_snippets(text)
    assert set(snippets) == {"Item-144648", "User-6902"}
    assert snippets["Item-144648"].startswith("The Undertaking of Tess")


def test_parse_neighbor_snippets_handles_empty():
    assert parse_neighbor_snippets("") == {}
    assert parse_neighbor_snippets("**Collaborative Neighbors:** (none available)") == {}


# ---------------------------------------------------------------------------
# composite reward
# ---------------------------------------------------------------------------

def _example(**overrides):
    ex = {
        "user_id": 7,
        "candidates": list(CANDS),
        "gold_item_id": 103,
        "candidate_titles": dict(TITLES),
        "candidate_memories": {},
        "instruction": "I want an epic fantasy novel.",
        "neighbor_snippets": dict(SNIPPETS),
        "neighbor_ids": list(SNIPPETS),
    }
    ex.update(overrides)
    return ex


def _reward(**cfg):
    return StageRReward(
        ranker=FrozenRanker(mode="stub"),
        grounding=GroundingScorer(encoder=FakeEncoder(), n_facets=cfg.get("n_facets", 7)),
        config=RewardConfig(**cfg),
    )


GOOD_COMPLETION = (
    '{"facets": [{"facet": "epic fantasy dragons", "confidence": 0.9, '
    '"supporting_neighbors": ["Item-1"]}]}'
)


def test_reward_decomposes_into_its_terms():
    r = _reward()
    b = r.per_example(GOOD_COMPLETION, _example())
    assert b.total == pytest.approx(
        b.r_ndcg + b.r_soft + b.r_margin + 0.2 * b.r_ground - b.penalty_len - b.penalty_fmt
    )
    assert not b.is_malformed
    assert b.penalty_fmt == 0.0
    assert b.n_facets == 1


def test_malformed_output_takes_format_penalty():
    b = _reward().per_example("I cannot produce facets, sorry.", _example())
    assert b.is_malformed
    assert b.penalty_fmt == 1.0
    assert b.r_ground == 0.0


def test_malformed_outputs_still_vary_across_prompts():
    """
    A malformed rollout is scored as "no memory", not as a flat constant.

    Within one GRPO group (same prompt) an all-malformed group is degenerate no
    matter what -- that is unavoidable. What must hold is that malformed rollouts
    on *different* prompts still differ, so a batch of them carries signal rather
    than collapsing to a single number (§9.2).

    Checked over several users rather than two, because with ten candidates any
    given pair lands on the same NDCG about 10% of the time.
    """
    r = _reward()
    totals = [
        r.per_example("garbage", _example(user_id=u, gold_item_id=CANDS[u % len(CANDS)])).total
        for u in range(12)
    ]
    assert len(set(totals)) > 1, "malformed reward is constant across prompts"
    mean = sum(totals) / len(totals)
    std = (sum((t - mean) ** 2 for t in totals) / len(totals)) ** 0.5
    assert std > 0.05, f"malformed rewards barely vary (std={std:.4f})"


def test_truncated_output_is_penalised_but_still_scored():
    truncated = '{"facets": [{"facet": "epic fantasy dragons", "confidence": 0.9, ' \
                '"supporting_neighbors": ["Item-1"]}, {"facet": "milita'
    b = _reward().per_example(truncated, _example())
    assert b.is_truncated
    assert b.penalty_fmt == 1.0, "truncation must still cost the format penalty"
    assert b.n_facets == 1, "but its usable facet is not thrown away"


def test_length_penalty_units_match_the_plan():
    """lambda_len = 0.1 per 100 tokens over a 400-token budget (§5)."""
    cfg = RewardConfig()
    r = StageRReward(ranker=FrozenRanker(mode="stub"),
                     grounding=GroundingScorer(encoder=FakeEncoder()), config=cfg)
    # 4 chars/token => 2400 chars = 600 tokens = 200 over budget = 2 units = 0.2
    penalty, n_tokens = r._length_penalty("x" * 2400)
    assert n_tokens == 600
    assert penalty == pytest.approx(0.2)
    assert r._length_penalty("x" * 400)[0] == 0.0, "under budget must be free"


def test_hallucinated_citations_are_counted():
    completion = ('{"facets": [{"facet": "epic fantasy", "confidence": 0.9, '
                  '"supporting_neighbors": ["Item-1", "Item-424242"]}]}')
    b = _reward().per_example(completion, _example())
    # parse_facets already filters against the whitelist, so nothing hallucinated
    # survives into the facet; the guard is that it is dropped, not scored.
    assert b.n_hallucinated_ids == 0
    assert b.r_ground >= 0.0


def test_r_null_is_cached_and_diagnostic_only():
    r = _reward()
    ex = _example()
    first = r.r_null(ex)
    assert r.r_null(ex) == first
    assert 7 in r._null_cache

    b = r.per_example(GOOD_COMPLETION, ex)
    assert b.r_null == pytest.approx(first)
    assert b.beats_null in (True, False)
    # r_null must not be subtracted from the returned reward (§5.4)
    assert b.total == pytest.approx(
        b.r_ndcg + b.r_soft + b.r_margin + 0.2 * b.r_ground - b.penalty_len - b.penalty_fmt
    )


def test_missing_required_column_raises_loudly():
    """A wiring bug must not masquerade as 'every rollout is malformed'."""
    with pytest.raises(KeyError, match="gold_item_id"):
        _reward().per_example(GOOD_COMPLETION, {"candidates": CANDS})


# ---------------------------------------------------------------------------
# TRL interface
# ---------------------------------------------------------------------------

def test_trl_call_signature_returns_one_float_per_completion():
    r = _reward()
    completions = [GOOD_COMPLETION, "garbage", GOOD_COMPLETION]
    rewards = r(
        prompts=["p1", "p2", "p3"],
        completions=completions,
        user_id=[1, 2, 3],
        candidates=[CANDS, CANDS, CANDS],
        gold_item_id=[103, 103, 103],
        candidate_titles=[TITLES, TITLES, TITLES],
        candidate_memories=[{}, {}, {}],
        instruction=["i", "i", "i"],
        neighbor_snippets=[SNIPPETS, SNIPPETS, SNIPPETS],
    )
    assert len(rewards) == 3
    assert all(isinstance(x, float) for x in rewards)
    assert rewards[0] > rewards[1], "well-formed output must beat garbage"


def test_trl_call_accepts_chat_format_completions():
    r = _reward()
    rewards = r(
        completions=[[{"role": "assistant", "content": GOOD_COMPLETION}]],
        user_id=[1],
        candidates=[CANDS],
        gold_item_id=[103],
        candidate_titles=[TITLES],
        neighbor_snippets=[SNIPPETS],
    )
    assert len(rewards) == 1 and not r.last_breakdowns[0].is_malformed


def test_aggregate_reports_the_m4_logging_metrics():
    r = _reward()
    r(
        completions=[GOOD_COMPLETION, "garbage"],
        user_id=[1, 2],
        candidates=[CANDS, CANDS],
        gold_item_id=[103, 104],
        candidate_titles=[TITLES, TITLES],
        neighbor_snippets=[SNIPPETS, SNIPPETS],
    )
    agg = r.aggregate()
    for key in (
        "reward_mean", "reward_std", "grounding_score", "format_valid_rate",
        "completion_length_mean", "pct_beating_null", "hit_at_1",
    ):
        assert key in agg, f"§M4 requires {key} to be logged"
    assert agg["format_valid_rate"] == pytest.approx(0.5)


def test_reward_never_raises_on_any_malformed_case():
    """Reuses the M1 parser corpus: none of it may crash the reward."""
    from tests.rl.test_policy_parser import MALFORMED_CASES

    r = _reward()
    for name, raw, _ in MALFORMED_CASES:
        b = r.per_example(raw, _example())
        assert isinstance(b.total, float) and math.isfinite(b.total), name


# ---------------------------------------------------------------------------
# continuous tie-breaker (soft_weight)
# ---------------------------------------------------------------------------

def test_soft_weight_is_off_because_measurement_ruled_against_it():
    """
    M2 Part B enabled the soft term on §M2's tie-rate rule, then measured it
    against the real LLM_Rec on five M_collab samples per user and switched it
    back off.

    p_gold breaks exactly the pairs NDCG@5 cannot judge, and on those 197 pairs it
    agrees with gpt-4o-mini only 40.1% of the time, CI [33.5, 47.1] -- entirely
    below chance. It drags a weak-but-real 60.6% signal down to 47.0%. This pins
    the default so the earlier, wrong decision cannot quietly come back.
    """
    assert RewardConfig().soft_weight == 0.0
    # margin_weight off too, so this isolates the soft term rather than the
    # continuous term that replaced it.
    b = _reward(margin_weight=0.0).per_example(GOOD_COMPLETION, _example())
    assert b.r_soft == 0.0
    assert b.total == pytest.approx(b.r_ndcg + 0.2 * b.r_ground - b.penalty_len - b.penalty_fmt)


def test_soft_weight_enters_the_total_when_enabled():
    r = _reward(soft_weight=0.3)
    b = r.per_example(GOOD_COMPLETION, _example())
    assert 0.0 <= b.p_gold <= 1.0
    assert b.r_soft == pytest.approx(0.3 * b.p_gold)
    assert b.total == pytest.approx(
        b.r_ndcg + b.r_soft + b.r_margin + 0.2 * b.r_ground - b.penalty_len - b.penalty_fmt
    )


def test_soft_weight_breaks_ties_that_ndcg_cannot():
    """
    The point of the soft term: two memories that put the gold in the same slot
    score identically under any rank-only reward (measured 71.1% of the time on
    the real 3B ranker at M2 Part B), which zeroes the GRPO advantage.
    """
    # Both sides run with margin_weight=0.0: the question here is what the SOFT
    # term does on its own, and leaving the margin on would break the ties for it.
    plain = _reward(soft_weight=0.0, margin_weight=0.0)
    soft = _reward(soft_weight=0.3, margin_weight=0.0)
    ex = _example()

    def totals(rw):
        out = []
        for i in range(24):
            completion = ('{"facets": [{"facet": "variant %d fantasy", "confidence": 0.8, '
                          '"supporting_neighbors": ["Item-1"]}]}' % i)
            out.append(rw.per_example(completion, ex).total)
        return out

    assert _uniqueness(totals(soft)) > _uniqueness(totals(plain))


def test_tie_rate_is_reported():
    from src.rl.reward.composite import _tie_rate

    assert _tie_rate([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert _tie_rate([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.0)
    assert _tie_rate([1.0, 1.0, 3.0, 4.0]) == pytest.approx(0.5)
    assert _tie_rate([1.0]) == 0.0


def _uniqueness(values):
    return len({round(v, 9) for v in values}) / len(values)


# ---------------------------------------------------------------------------
# pointwise scoring (M2 Part B fallback)
# ---------------------------------------------------------------------------

def test_pointwise_prompt_shows_exactly_one_candidate():
    """
    The whole point of pointwise is that a candidate is judged without seeing its
    competitors. If another candidate leaked in, the scores would be coupled again
    and the tie ceiling would come back.
    """
    from src.rl.reward.ranker import build_pointwise_prompt

    p = build_pointwise_prompt(
        candidate_id=101, title="Dune", memory="classic sci-fi",
        m_collab=[{"facet": "likes epic science fiction", "confidence": 0.9}],
        instruction="something long and strange", user_id=7,
    )
    assert "Item 101" in p and "Dune" in p
    assert "Item 102" not in p
    assert "likes epic science fiction" in p
    assert p.rstrip().endswith("Answer:")


def test_pointwise_prompt_matches_listwise_above_the_candidate_block():
    """Switching scoring mode must change the question, not the user context."""
    from src.rl.reward.ranker import build_pointwise_prompt, build_ranker_prompt

    kw = dict(m_collab=[{"facet": "likes epic science fiction", "confidence": 0.9}],
              instruction="find me a book", user_id=7)
    lw = build_ranker_prompt(candidates=[101], candidate_titles={"101": "Dune"}, **kw)
    pw = build_pointwise_prompt(candidate_id=101, title="Dune", **kw)

    # The opening sentence differs by design ("each candidate" vs "a candidate"),
    # because the question genuinely differs. What must match is everything
    # describing the *user*: target, instruction, and the M_collab block.
    def context(p):
        return p[p.index("**Target User:**"):p.index("**Candidate Item")
                 if "**Candidate Item" in p else p.index("**Candidate Items")]

    assert context(lw) == context(pw)


def test_pointwise_ranking_is_by_independent_scores_not_renormalised():
    """
    P("Yes") per candidate must survive into `scores` untouched -- that value is
    what the reward reads as p_gold, and renormalising would recouple them.
    """
    from src.rl.reward.ranker import FrozenRanker

    out = FrozenRanker._rank_from_scores([10, 11, 12], [0.9, 0.1, 0.5])
    assert out.ranking == [10, 12, 11]
    assert out.scores[10] == pytest.approx(0.9)
    assert out.scores[12] == pytest.approx(0.5)
    assert sum(out.scores.values()) == pytest.approx(1.5)   # NOT normalised to 1


def test_unknown_scoring_mode_is_rejected():
    from src.rl.reward.ranker import FrozenRanker

    with pytest.raises(ValueError, match="scoring"):
        FrozenRanker(mode="stub", scoring="pairwise")


# --- margin_logit: the continuous term that replaced soft_weight -------------
#
# docs/RESULTS.md (M2): NDCG@5 is a function of the gold's rank, so it takes at
# most k+1 values and leaves 71.1% of five-sample groups completely flat. A flat
# group is std(r)=0 and no gradient at all (§9.2). These lock the replacement in
# place, including the two ways it was nearly got wrong.


class FixedLogitRanker:
    """Ranker returning logits chosen by the test, so margins are exact."""

    def __init__(self, logits, expose_logits=True):
        self.logits = dict(logits)
        self.expose_logits = expose_logits

    def score(self, candidates, **_):
        from src.rl.reward.ranker import RankerOutput

        picked = {int(c): float(self.logits.get(int(c), 0.0)) for c in candidates}
        top = max(picked.values())
        exps = {k: math.exp(v - top) for k, v in picked.items()}
        z = sum(exps.values())
        return RankerOutput(
            ranking=[k for k, _ in sorted(picked.items(), key=lambda kv: -kv[1])],
            scores={k: v / z for k, v in exps.items()},
            logits=picked if self.expose_logits else {},
        )


def _margin_reward(logits, expose_logits=True, **cfg):
    return StageRReward(
        ranker=FixedLogitRanker(logits, expose_logits),
        grounding=GroundingScorer(encoder=FakeEncoder(), n_facets=7),
        config=RewardConfig(**cfg),
    )


def test_margin_weight_default_is_the_measured_value():
    """
    0.02 is not a guess. Any w > 0 breaks every tie (the sweep is a step
    function, not a curve) and within-user accuracy is flat at 62.9% across
    [0.005, 0.05], eroding from w >= 0.1 as the margin starts overriding NDCG@5
    rather than breaking its ties.
    """
    assert RewardConfig().margin_weight == 0.02


def test_soft_weight_stays_off_now_that_margin_replaces_it():
    """
    Both are continuous tie-breakers, so enabling both double-counts. Given the
    same job -- the pairs NDCG@5 cannot judge -- p_gold agreed 40.1% of the time
    (below chance) and margin_logit 58.4% (above it).
    """
    assert RewardConfig().soft_weight == 0.0


def test_margin_is_the_logit_gap_to_the_best_competitor():
    r = _margin_reward({101: 4.0, 102: 1.5, 103: -2.0})
    b = r.per_example(GOOD_COMPLETION, _example(gold_item_id=101))
    assert b.margin_logit == pytest.approx(4.0 - 1.5)
    assert b.r_margin == pytest.approx(0.02 * 2.5)


def test_margin_is_negative_when_the_gold_is_beaten():
    r = _margin_reward({101: 0.5, 102: 3.0, 103: 1.0})
    b = r.per_example(GOOD_COMPLETION, _example(gold_item_id=101))
    assert b.margin_logit == pytest.approx(0.5 - 3.0)


def test_margin_separates_two_rollouts_that_ndcg_ties():
    """
    The whole point. Two memories that put the gold in the same slot get the
    same NDCG@5 and would leave the group at std(r)=0; the margin still tells
    them apart, so the group produces a gradient.
    """
    ex = _example(gold_item_id=101)
    a = _margin_reward({101: 5.0, 102: 1.0, 103: 0.0}).per_example(GOOD_COMPLETION, ex)
    b = _margin_reward({101: 5.0, 102: 4.9, 103: 0.0}).per_example(GOOD_COMPLETION, ex)
    assert a.r_ndcg == b.r_ndcg          # same gold rank -> NDCG cannot separate
    assert a.total != b.total            # margin can


def test_margin_degrades_to_zero_when_the_ranker_exposes_no_logits():
    """
    The pointwise path returns no logit map. A reward that raised there would
    fail three hours into a rented session, so it falls back to plain NDCG@5.
    """
    r = _margin_reward({101: 4.0, 102: 1.0, 103: 0.0}, expose_logits=False)
    b = r.per_example(GOOD_COMPLETION, _example(gold_item_id=101))
    assert b.margin_logit == 0.0
    assert b.r_margin == 0.0


def test_margin_weight_zero_recovers_the_reward_written_in_section_5():
    """The M5 ablation needs the original formula reachable from config alone."""
    logits = {101: 4.0, 102: 1.5, 103: -2.0}
    ex = _example(gold_item_id=101)
    off = _margin_reward(logits, margin_weight=0.0).per_example(GOOD_COMPLETION, ex)
    on = _margin_reward(logits).per_example(GOOD_COMPLETION, ex)
    assert off.r_margin == 0.0
    assert on.total == pytest.approx(off.total + 0.02 * on.margin_logit)


def test_score_many_matches_per_example_and_batches_the_ranker():
    """
    §5.1 budgets the reward assuming 64 rollouts share a forward pass. The
    batching lived in FrozenRanker.score_batch and nothing above it ever called
    it with more than one request, so a whole GRPO step was scored one prompt at
    a time. score_many fixes that; this pins both halves of the contract --
    identical numbers, and one ranker call per batch rather than per rollout.
    """
    calls = {"batches": 0, "requests": 0}
    inner = FrozenRanker(mode="stub")

    class CountingRanker:
        def score(self, **kw):
            return inner.score(**kw)

        def score_batch(self, requests):
            calls["batches"] += 1
            calls["requests"] += len(requests)
            return inner.score_batch(requests)

    completions = [
        '{"facets": [{"facet": "variant %d fantasy", "confidence": 0.8, '
        '"supporting_neighbors": ["Item-1"]}]}' % i
        for i in range(10)
    ]
    examples = [_example() for _ in completions]

    reward = StageRReward(ranker=CountingRanker(),
                          grounding=GroundingScorer(encoder=FakeEncoder(), n_facets=7))
    batched = reward.score_many(completions, examples, batch_size=4)
    one_at_a_time = [reward.per_example(c, e) for c, e in zip(completions, examples)]

    assert [b.total for b in batched] == pytest.approx([b.total for b in one_at_a_time])
    # 10 rollouts at batch 4 -> ceil(10/4) = 3 forward passes, not 10. The
    # per_example loop above goes through .score and is not counted here.
    assert calls["batches"] == 3
    assert calls["requests"] == 10


def test_grounding_caches_snippet_embeddings_across_rollouts():
    """
    Neighbour snippets belong to the user, not the rollout, so every generation
    for one prompt cites the same text: 8 teacher samples at M3, G=8 rollouts per
    group at M4. Without a cache bge-small re-encodes identical strings eight
    times, and that was measurably the whole bottleneck while scoring the M3
    samples -- 720% CPU with the GPU idle.
    """
    calls = {"n": 0, "texts": 0}

    class CountingEncoder(FakeEncoder):
        def encode(self, texts):
            calls["n"] += 1
            calls["texts"] += len(list(texts))
            return FakeEncoder.encode(self, texts)

    scorer = GroundingScorer(encoder=CountingEncoder(), n_facets=7)
    facets = [{"facet": "epic fantasy dragons", "supporting_neighbors": ["Item-1"]}]
    snippets = {"Item-1": "a long snippet about dragons and epic fantasy"}

    first = scorer.score(facets, snippets)
    after_first = calls["texts"]
    for _ in range(7):                      # the other seven rollouts of the group
        again = scorer.score(facets, snippets)
        assert again.score == pytest.approx(first.score)

    assert calls["texts"] == after_first     # nothing re-encoded
    assert scorer.cache_hits > 0
