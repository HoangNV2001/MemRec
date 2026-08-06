"""
Grounding reward -- anti-hallucination guard (RL_PLAN.md §5.2).

    r_ground = (# facets with >=1 valid source_id AND cos_sim(facet, snippet) >= tau) / N_f

Two deliberate choices worth stating, because both are easy to get subtly wrong:

**Denominator is the requested ``n_facets``, not the number produced.** Dividing
by the produced count would hand the policy a trivial exploit: emit one
well-grounded facet, score 1.0, and drop the other six. The plan writes ``/ N_f``
and it means the target count.

**Similarity is measured against the neighbour *snippet*, not against ``M_v``.**
The snippet is what the policy actually read (``SnippetPacker`` renders it from
static item metadata). Scoring against storage-side memory would judge the policy
on text it never saw, and would break outright for neighbour users that are not
among the snapshot's 2350 and therefore have no ``M_u``.

The encoder is injectable so the whole reward path is testable on CPU without
downloading a model.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import math

DEFAULT_TAU = 0.35
DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"


class Encoder(Protocol):
    """Anything that maps texts to unit-comparable vectors."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


@dataclass
class GroundingDetail:
    """Per-facet outcome, for diagnostics and the M5 qualitative read."""

    facet_index: int
    cited: List[str] = field(default_factory=list)
    valid_cited: List[str] = field(default_factory=list)
    best_similarity: float = 0.0
    grounded: bool = False


@dataclass
class GroundingResult:
    score: float
    n_grounded: int
    n_facets_seen: int
    denominator: int
    hallucinated_ids: List[str] = field(default_factory=list)
    details: List[GroundingDetail] = field(default_factory=list)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


class SentenceTransformerEncoder:
    """Lazy wrapper around bge-small. Loads on first use; runs fine on CPU."""

    def __init__(self, model_name: str = DEFAULT_ENCODER, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def encode(self, texts: Sequence[str]):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model.encode(list(texts), normalize_embeddings=True)


class GroundingScorer:
    """Scores how well a facet list is supported by the neighbours it cites."""

    def __init__(
        self,
        encoder: Optional[Encoder] = None,
        tau: float = DEFAULT_TAU,
        n_facets: int = 7,
    ):
        self.encoder = encoder
        self.tau = tau
        self.n_facets = n_facets

    def _encode(self, texts: Sequence[str]):
        if self.encoder is None:
            self.encoder = SentenceTransformerEncoder()
        return self.encoder.encode(texts)

    def score(
        self,
        facets: Sequence[Dict],
        neighbor_snippets: Dict[str, str],
    ) -> GroundingResult:
        """
        Args:
            facets: parsed facets, each with ``facet`` text and
                ``supporting_neighbors`` (already normalised by
                ``src.rl.policy.parse_facets``).
            neighbor_snippets: ``{node_id: snippet}`` for this user's N'_k(u),
                from ``src.rl.env.parse_neighbor_snippets``.
        """
        denominator = max(1, self.n_facets)
        if not facets:
            return GroundingResult(0.0, 0, 0, denominator)

        # One encode call for everything: facets plus every snippet actually cited.
        cited_per_facet: List[List[str]] = []
        needed_snippets: List[str] = []
        hallucinated: List[str] = []
        for facet in facets:
            valid = []
            for node_id in facet.get("supporting_neighbors", []) or []:
                if node_id in neighbor_snippets:
                    valid.append(node_id)
                    if node_id not in needed_snippets:
                        needed_snippets.append(node_id)
                elif node_id not in hallucinated:
                    hallucinated.append(node_id)
            cited_per_facet.append(valid)

        facet_texts = [str(f.get("facet", "")) for f in facets]
        snippet_texts = [neighbor_snippets[n] for n in needed_snippets]

        vectors = self._encode(facet_texts + snippet_texts) if snippet_texts else self._encode(facet_texts)
        facet_vecs = vectors[: len(facet_texts)]
        snippet_vecs = {n: vectors[len(facet_texts) + i] for i, n in enumerate(needed_snippets)}

        details: List[GroundingDetail] = []
        n_grounded = 0
        for i, facet in enumerate(facets):
            best = 0.0
            for node_id in cited_per_facet[i]:
                best = max(best, cosine(facet_vecs[i], snippet_vecs[node_id]))
            grounded = bool(cited_per_facet[i]) and best >= self.tau
            n_grounded += int(grounded)
            details.append(
                GroundingDetail(
                    facet_index=i,
                    cited=list(facet.get("supporting_neighbors", []) or []),
                    valid_cited=cited_per_facet[i],
                    best_similarity=best,
                    grounded=grounded,
                )
            )

        return GroundingResult(
            score=n_grounded / denominator,
            n_grounded=n_grounded,
            n_facets_seen=len(facets),
            denominator=denominator,
            hallucinated_ids=hallucinated,
            details=details,
        )
