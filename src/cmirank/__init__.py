"""CPU-safe building blocks for collaborative-memory iterative ranking.

This package does not train a policy or alter the MemRec baseline by default.
"""

from .request import RankRequest
from .rewards import mpss_rewards, ndcg_at_k

__all__ = ["RankRequest", "mpss_rewards", "ndcg_at_k"]
