"""Reward components for Stage-R GRPO (RL_PLAN.md §5)."""
from src.rl.reward.metrics import hit_at_k, ndcg_at_k, rank_of
from src.rl.reward.grounding import GroundingScorer
from src.rl.reward.ranker import FrozenRanker, RankerOutput
from src.rl.reward.composite import RewardConfig, RewardBreakdown, StageRReward

__all__ = [
    "hit_at_k",
    "ndcg_at_k",
    "rank_of",
    "GroundingScorer",
    "FrozenRanker",
    "RankerOutput",
    "RewardConfig",
    "RewardBreakdown",
    "StageRReward",
]
