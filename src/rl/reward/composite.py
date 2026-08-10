"""
Composite Stage-R reward (RL_PLAN.md §5).

    r = r_ndcg + lambda_g * r_ground - lambda_len * over_length - lambda_fmt * is_malformed

Exposed as a TRL-compatible callable: ``fn(prompts, completions, **columns) ->
list[float]``, where ``columns`` are the extra jsonl fields TRL forwards from the
dataset (``candidates``, ``gold_item_id``, ``neighbor_snippets``, ...).

Design notes that are easy to get wrong and expensive to discover mid-run:

* **Malformed generations still get scored.** A completion that fails to parse
  takes the format penalty and an ``r_ndcg`` computed with empty ``M_collab`` --
  which is exactly ``r_null``. Returning a flat constant instead would make every
  malformed rollout identical, and a group of them would have ``std = 0`` and
  contribute no gradient at all (§9.2).
* **Nothing here may raise.** A single exception aborts the training step. Every
  component is defensive, and ``per_example`` catches anything that slips through.
* **``r_null`` is diagnostic only.** Subtracting it changes no gradient -- it is
  constant per prompt and GRPO already mean-centres within the group (§5.4). It
  is recorded because ``% rollouts beating null`` is the best health metric we have.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.rl.policy import parse_facets
from src.rl.reward.grounding import GroundingScorer
from src.rl.reward.metrics import hit_at_k, ndcg_at_k, rank_of
from src.rl.reward.ranker import FrozenRanker


@dataclass
class RewardConfig:
    """Starting values from RL_PLAN.md §5."""

    lambda_ground: float = 0.2
    lambda_len: float = 0.1        # per 100 tokens over budget
    lambda_fmt: float = 1.0
    length_budget_tokens: int = 400
    length_penalty_unit: int = 100
    ndcg_k: int = 5                # §5.1: NDCG@5, not Hit@1
    n_facets: int = 7
    chars_per_token: float = 4.0   # crude but consistent with SnippetPacker

    # Continuous tie-breaker: weight on the ranker's softmax probability of the
    # gold candidate. Set to 0.0 to recover exactly the reward written in §5.
    #
    # OFF. It was briefly enabled at 0.3 -- §M2's rule is "tie rate > 50% -> turn
    # on at 0.3", and the 3B ranker ties on 71% of pairs -- and then measured
    # properly against the real LLM_Rec with five M_collab samples per user
    # (docs/RESULTS.md, M2 Part B). It makes the reward WORSE, and the rule that
    # switched it on was looking at the wrong number.
    #
    # Decomposition over the 296 within-user pairs gpt-4o-mini can separate:
    #
    #   NDCG@5 decides                99 pairs   60.6%  CI [50.8, 69.7]  > chance
    #   NDCG@5 ties, p_gold decides  197 pairs   40.1%  CI [33.5, 47.1]  BELOW chance
    #   combined                     296 pairs   47.0%                   = chance
    #
    # p_gold is asked to break exactly the pairs NDCG@5 cannot judge, and on those
    # it is anti-informative -- its whole interval sits under 50%. So it does not
    # rescue the degenerate groups, it fills them with noise pointed the wrong way,
    # and it drags a weak-but-real 60.6% signal down to chance. The effect is
    # independent of w (any w > 0 breaks every tie), so it cannot be tuned away.
    #
    # This leaves the tie problem below UNSOLVED, which is a real blocker for M4 --
    # see docs/RESULTS.md. Do not re-enable this as a fix for it.
    #
    # Why it exists. NDCG@k is a function of the gold's *rank* alone, so it takes
    # at most k+1 distinct values and two different memories that land the gold in
    # the same slot score identically. Measured on the 149 val users with
    # gpt-4o-mini (docs/RESULTS.md, M2 Part A): two independently sampled M_collab
    # tie 80.5% of the time under NDCG@5, and 74.5% under NDCG@10 -- 74% is the
    # hard ceiling, because that is how often both samples put the gold in the same
    # position. Inside a GRPO group a tie means std(r)=0, advantage 0, no gradient
    # (§9.2), and dynamic sampling (§6.4) would filter far past the 60% alarm
    # threshold in §M4's kill criteria.
    #
    # p_gold is continuous, so it almost never ties, while NDCG stays the dominant
    # term and the reported metric stays interpretable. This is the same reasoning
    # §5.1 already used to reject Hit@1 in favour of NDCG@5, carried one step
    # further. M2 Part B made that measurement, and it came out against -- see the
    # note above. Kept as a knob because the *idea* is sound; this particular
    # continuous term is what failed.
    soft_weight: float = 0.0

    # Weight on the ranker's *logit* margin between the gold candidate and its
    # best competitor. This is the continuous term that works, and it replaces
    # `soft_weight` above rather than joining it (docs/RESULTS.md, M2).
    #
    # Why a continuous term is needed at all. NDCG@k is a function of the gold's
    # rank alone, so it takes at most k+1 values, and two memories that land the
    # gold in the same slot score identically. That ceiling belongs to the *task*,
    # not to whoever is scoring: gpt-4o-mini ties 80.1% of within-user pairs and
    # gpt-5.6-luna, a much stronger model, ties 79.1%. On Qwen3.5-4B it leaves
    # 71.1% of five-sample groups completely flat -- std(r)=0, advantage 0, no
    # gradient (§9.2), i.e. M4 would train on under a third of its data.
    #
    # Measured on 149 val users x 5 teacher samples, against the real LLM_Rec:
    #
    #   NDCG@5 decides                 72/102  = 70.6%  [61.1, 78.6]
    #   NDCG@5 ties -> margin decides  101/173 = 58.4%  [50.9, 65.5]
    #   combined                       173/275 = 62.9%  [57.1, 68.4]
    #   degenerate groups              71.1% -> 0.0%
    #
    # That middle row is the whole difference from `soft_weight`: given the same
    # job, p_gold came back at 40.1%, below chance. margin_logit is above it.
    #
    # Why the *logit* margin and not the probability margin. This ranker puts
    # ~99% of the letter mass on a single token, so p(gold) - max p(other)
    # saturates near +-1 and what is left moving is floating-point noise: it
    # separates 99.9% of pairs against a 98.3% noise floor, which is 1.6 points
    # of real signal dressed up as a perfect score. The logit form is unbounded
    # and gives 90.4% against 48.6%.
    #
    # Why 0.02. A weight sweep is a step function, not a curve -- any w > 0 breaks
    # every tie, and accuracy is flat at 62.9% across [0.005, 0.05], eroding only
    # from w >= 0.1 as the margin starts overriding NDCG instead of breaking its
    # ties. 0.02 sits in the middle of the flat region, and keeps the magnitudes
    # ordered too: NDCG's real-vs-corrupted headroom is +0.131 against the
    # margin's +1.09, so 0.02 * 1.09 stays the smaller term.
    #
    # Set to 0.0 to recover exactly the reward written in §5, for the M5 ablation.
    margin_weight: float = 0.02


@dataclass
class RewardBreakdown:
    """Every term kept separately -- §9 requires them logged, not just the total."""

    total: float = 0.0
    r_ndcg: float = 0.0
    r_ground: float = 0.0
    penalty_len: float = 0.0
    penalty_fmt: float = 0.0
    # diagnostics
    is_malformed: bool = False
    is_truncated: bool = False
    n_facets: int = 0
    gold_rank: Optional[int] = None
    hit_at_1: float = 0.0
    completion_tokens: int = 0
    n_hallucinated_ids: int = 0
    r_null: Optional[float] = None
    beats_null: Optional[bool] = None
    p_gold: float = 0.0            # ranker probability mass on the gold candidate
    r_soft: float = 0.0            # soft_weight * p_gold, 0 unless enabled
    margin_logit: float = 0.0      # logit(gold) - max logit(other candidate)
    r_margin: float = 0.0          # margin_weight * margin_logit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StageRReward:
    """Callable reward for TRL's ``GRPOTrainer(reward_funcs=[...])``."""

    def __init__(
        self,
        ranker: Optional[FrozenRanker] = None,
        grounding: Optional[GroundingScorer] = None,
        config: Optional[RewardConfig] = None,
    ):
        self.config = config or RewardConfig()
        self.ranker = ranker or FrozenRanker(mode="stub")
        self.grounding = grounding or GroundingScorer(n_facets=self.config.n_facets)
        self._null_cache: Dict[int, float] = {}
        self.last_breakdowns: List[RewardBreakdown] = []
        # Ranker forward-pass batch. §5.1 assumes 64; 32 is what a 24 GB card
        # holds for a 4B ranker at ~1200-token prompts (48 is already past
        # saturation, 64 OOMs). Raise on an H100.
        self.ranker_batch_size = 32

    __name__ = "stage_r_reward"   # TRL uses this for logging column names

    # -- helpers ------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.config.chars_per_token)

    def _length_penalty(self, completion: str) -> tuple:
        n_tokens = self._estimate_tokens(completion)
        over = max(0, n_tokens - self.config.length_budget_tokens)
        units = over / self.config.length_penalty_unit
        return self.config.lambda_len * units, n_tokens

    def r_null(self, example: Dict) -> float:
        """
        NDCG@k with no collaborative memory. Deterministic, so cached per user.

        Diagnostic only (§5.4) -- it never enters the returned reward.
        """
        user_id = int(example.get("user_id", -1))
        if user_id in self._null_cache:
            return self._null_cache[user_id]
        out = self.ranker.score(
            candidates=example["candidates"],
            candidate_titles=example.get("candidate_titles", {}),
            candidate_memories=example.get("candidate_memories", {}),
            m_collab=None,
            instruction=example.get("instruction"),
            user_id=user_id,
        )
        value = ndcg_at_k(out.ranking, int(example["gold_item_id"]), self.config.ndcg_k)
        self._null_cache[user_id] = value
        return value

    # -- single example -----------------------------------------------------

    def per_example(self, completion: str, example: Dict) -> RewardBreakdown:
        """
        Score one rollout.

        Never raises on *model* output -- a bad generation is a reward signal.
        Does raise on a missing dataset column, because that is a wiring bug that
        would otherwise show up as "every rollout is malformed" three hours into
        a rented session.
        """
        for required in ("candidates", "gold_item_id"):
            if required not in example:
                raise KeyError(
                    f"reward is missing required dataset column {required!r}. "
                    f"TRL only forwards columns present in the Dataset; check "
                    f"src/rl/build_dataset.py. Got: {sorted(example)}"
                )
        try:
            return self._per_example(completion, example)
        except Exception:  # noqa: BLE001 - one bad rollout must not kill a step
            return RewardBreakdown(
                total=-self.config.lambda_fmt,
                penalty_fmt=self.config.lambda_fmt,
                is_malformed=True,
                completion_tokens=self._estimate_tokens(completion or ""),
            )

    def _prepare(self, completion: str, example: Dict):
        """Parse the completion and build the ranker request it implies."""
        cfg = self.config
        snippets = example.get("neighbor_snippets") or {}
        allowed_ids = list(snippets) or example.get("neighbor_ids") or None
        parsed = parse_facets(completion, valid_node_ids=allowed_ids, max_facets=cfg.n_facets)
        request = dict(
            candidates=example["candidates"],
            candidate_titles=example.get("candidate_titles", {}),
            candidate_memories=example.get("candidate_memories", {}),
            # A malformed completion is scored as "no memory at all" rather than
            # as a constant, so groups of malformed rollouts still differ by prompt.
            m_collab=parsed.facets if parsed.facets else None,
            instruction=example.get("instruction"),
            user_id=example.get("user_id"),
        )
        return parsed, request, snippets

    def score_many(self, completions: Sequence[str], examples: Sequence[Dict],
                   batch_size: int = 32) -> List[RewardBreakdown]:
        """
        Score many rollouts with **batched** ranker forward passes.

        ``per_example`` calls the ranker one prompt at a time, which is what the
        TRL entry point used to do for a whole step. §5.1 sizes the reward budget
        assuming 64 rollouts share a forward pass ("Batch được 64 rollout cùng
        lúc"), and that assumption did not survive the composite: the batching
        lived in ``FrozenRanker.score_batch`` and nothing above it ever called it
        with more than one request. Measured on Qwen3.5-4B, batch 1 runs at ~1/s
        against ~2.8/s at batch 32, so a 400-step M4 run was going to pay roughly
        three times the reward cost §11.3 budgeted.

        Everything outside the ranker -- parsing, grounding, penalties -- stays
        per example, so the returned breakdowns are identical to ``per_example``
        up to the ranker's own batch-composition noise.
        """
        prepared, requests = [], []
        for completion, example in zip(completions, examples):
            try:
                parsed, request, snippets = self._prepare(_as_text(completion), example)
            except Exception:  # noqa: BLE001 - fall back to the guarded path
                prepared.append(None)
                continue
            prepared.append((parsed, snippets))
            requests.append((len(prepared) - 1, request))

        outs: Dict[int, Any] = {}
        for start in range(0, len(requests), batch_size):
            chunk = requests[start:start + batch_size]
            for (idx, _), out in zip(chunk, self.ranker.score_batch([r for _, r in chunk])):
                outs[idx] = out

        results = []
        for i, (completion, example) in enumerate(zip(completions, examples)):
            if prepared[i] is None or i not in outs:
                results.append(self.per_example(_as_text(completion), example))
                continue
            parsed, snippets = prepared[i]
            try:
                results.append(self._finish(_as_text(completion), example,
                                            parsed, snippets, outs[i]))
            except Exception:  # noqa: BLE001 - one bad rollout must not kill a step
                results.append(RewardBreakdown(
                    total=-self.config.lambda_fmt,
                    penalty_fmt=self.config.lambda_fmt,
                    is_malformed=True,
                    completion_tokens=self._estimate_tokens(_as_text(completion) or ""),
                ))
        return results

    def _per_example(self, completion: str, example: Dict) -> RewardBreakdown:
        parsed, request, snippets = self._prepare(completion, example)
        out = self.ranker.score(**request)
        return self._finish(completion, example, parsed, snippets, out)

    def _finish(self, completion: str, example: Dict, parsed, snippets, out) -> RewardBreakdown:
        cfg = self.config
        gold = int(example["gold_item_id"])
        r_ndcg = ndcg_at_k(out.ranking, gold, cfg.ndcg_k)
        p_gold = float(out.scores.get(gold, 0.0))
        r_soft = cfg.soft_weight * p_gold
        # Continuous tie-breaker. Defaults to 0.0 when the ranker exposes no
        # logits (the pointwise path) so the reward degrades to plain NDCG@5
        # rather than raising three hours into a rented session.
        others = [v for i, v in out.logits.items() if i != gold]
        margin_logit = float(out.logits[gold] - max(others)) if (gold in out.logits and others) else 0.0
        r_margin = cfg.margin_weight * margin_logit

        ground = self.grounding.score(parsed.facets, snippets) if parsed.facets else None
        r_ground = ground.score if ground else 0.0

        penalty_len, n_tokens = self._length_penalty(completion or "")
        # `is_valid` is False for truncated-but-salvaged output too, which is
        # intended: truncation should still cost the format penalty (§9.4).
        penalty_fmt = 0.0 if parsed.is_valid else cfg.lambda_fmt

        total = (r_ndcg + r_soft + r_margin + cfg.lambda_ground * r_ground
                 - penalty_len - penalty_fmt)

        null_value = example.get("r_null")
        if null_value is None:
            null_value = self.r_null(example) if example.get("_compute_null", True) else None

        return RewardBreakdown(
            total=total,
            r_ndcg=r_ndcg,
            r_ground=r_ground,
            penalty_len=penalty_len,
            penalty_fmt=penalty_fmt,
            is_malformed=not parsed.facets,
            is_truncated=parsed.truncated,
            n_facets=parsed.n_facets,
            gold_rank=rank_of(out.ranking, gold),
            hit_at_1=hit_at_k(out.ranking, gold, 1),
            completion_tokens=n_tokens,
            n_hallucinated_ids=len(ground.hallucinated_ids) if ground else 0,
            r_null=null_value,
            beats_null=(r_ndcg > null_value) if null_value is not None else None,
            p_gold=p_gold,
            r_soft=r_soft,
            margin_logit=margin_logit,
            r_margin=r_margin,
        )

    # -- TRL entry point ----------------------------------------------------

    def __call__(
        self,
        prompts: Optional[Sequence[str]] = None,
        completions: Optional[Sequence[str]] = None,
        **columns: Any,
    ) -> List[float]:
        """
        TRL reward signature. ``columns`` holds the passthrough dataset fields,
        each a list aligned with ``completions``.
        """
        completions = list(completions or [])
        examples = [
            {key: values[i] for key, values in columns.items()
             if isinstance(values, (list, tuple)) and i < len(values)}
            for i in range(len(completions))
        ]
        # Batched: a GRPO step scores G x prompts rollouts at once and §5.1 budgets
        # them sharing forward passes. See score_many.
        breakdowns = self.score_many(completions, examples, batch_size=self.ranker_batch_size)

        self.last_breakdowns = breakdowns
        return [b.total for b in breakdowns]

    def aggregate(self, breakdowns: Optional[Sequence[RewardBreakdown]] = None) -> Dict[str, float]:
        """Batch-level metrics for the §M4 logging checklist."""
        rows = list(breakdowns if breakdowns is not None else self.last_breakdowns)
        if not rows:
            return {}
        n = len(rows)
        beating = [b for b in rows if b.beats_null is not None]
        return {
            "reward_mean": sum(b.total for b in rows) / n,
            "reward_std": _std([b.total for b in rows]),
            "r_ndcg_mean": sum(b.r_ndcg for b in rows) / n,
            "p_gold_mean": sum(b.p_gold for b in rows) / n,
            "margin_logit_mean": sum(b.margin_logit for b in rows) / n,
            # Fraction of rollouts sharing their reward with another rollout in the
            # batch. The single best early-warning for §9.2 group degeneracy.
            "reward_tie_rate": _tie_rate([b.total for b in rows]),
            "grounding_score": sum(b.r_ground for b in rows) / n,
            "format_valid_rate": sum(1.0 for b in rows if not b.is_malformed and not b.is_truncated) / n,
            "completion_length_mean": sum(b.completion_tokens for b in rows) / n,
            "hit_at_1": sum(b.hit_at_1 for b in rows) / n,
            "pct_beating_null": (sum(1.0 for b in beating if b.beats_null) / len(beating)) if beating else 0.0,
            "hallucinated_ids_mean": sum(b.n_hallucinated_ids for b in rows) / n,
        }


def _as_text(completion: Any) -> str:
    """TRL passes plain strings, or chat-format ``[{'role', 'content'}]``."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")


def _tie_rate(values: Sequence[float], atol: float = 1e-9) -> float:
    """Fraction of values that are not unique. 1.0 means the batch is degenerate."""
    n = len(values)
    if n < 2:
        return 0.0
    rounded = [round(v / atol) for v in values]
    counts: Dict[int, int] = {}
    for v in rounded:
        counts[v] = counts.get(v, 0) + 1
    return sum(c for c in counts.values() if c > 1) / n


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5
