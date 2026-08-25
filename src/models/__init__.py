"""Recommendation models."""

from .llm_client import LLMClient

__all__ = ["LLMClient", "MemRecAgent", "LLMReranker", "VectorReranker"]


def __getattr__(name):
    """Avoid importing torch-dependent agents when only LLMClient is needed."""
    if name == "MemRecAgent":
        from .memrec_agent import MemRecAgent
        return MemRecAgent
    if name == "LLMReranker":
        from .reranker_llm import LLMReranker
        return LLMReranker
    if name == "VectorReranker":
        from .reranker_vector import VectorReranker
        return VectorReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
