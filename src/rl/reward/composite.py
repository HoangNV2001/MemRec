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
    # gold candidate. Default 0.0 == exactly the reward written in §5.
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
    # further. Left off by default: M2 Part B measures the real ranker's tie rate
    # and decides.
    soft_weight: float = 0.0


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

    def _per_example(self, completion: str, example: Dict) -> RewardBreakdown:
        cfg = self.config
        snippets = example.get("neighbor_snippets") or {}
        allowed_ids = list(snippets) or example.get("neighbor_ids") or None

        parsed = parse_facets(completion, valid_node_ids=allowed_ids, max_facets=cfg.n_facets)

        # A malformed completion is scored as "no memory at all" rather than as a
        # constant, so groups of malformed rollouts still differ by prompt.
        m_collab = parsed.facets if parsed.facets else None

        out = self.ranker.score(
            candidates=example["candidates"],
            candidate_titles=example.get("candidate_titles", {}),
            candidate_memories=example.get("candidate_memories", {}),
            m_collab=m_collab,
            instruction=example.get("instruction"),
            user_id=example.get("user_id"),
        )
        gold = int(example["gold_item_id"])
        r_ndcg = ndcg_at_k(out.ranking, gold, cfg.ndcg_k)
        p_gold = float(out.scores.get(gold, 0.0))
        r_soft = cfg.soft_weight * p_gold

        ground = self.grounding.score(parsed.facets, snippets) if parsed.facets else None
        r_ground = ground.score if ground else 0.0

        penalty_len, n_tokens = self._length_penalty(completion or "")
        # `is_valid` is False for truncated-but-salvaged output too, which is
        # intended: truncation should still cost the format penalty (§9.4).
        penalty_fmt = 0.0 if parsed.is_valid else cfg.lambda_fmt

        total = r_ndcg + r_soft + cfg.lambda_ground * r_ground - penalty_len - penalty_fmt

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
        breakdowns = []
        for i, completion in enumerate(completions):
            example = {key: values[i] for key, values in columns.items()
                       if isinstance(values, (list, tuple)) and i < len(values)}
            breakdowns.append(self.per_example(_as_text(completion), example))

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
