"""Exact Stage-ReRank input capture, independent of reward/target labels.

The digest uses sorted-key UTF-8 JSON without Unicode normalization.  This
preserves the text actually supplied to the baseline prompt; callers must not
silently normalize or truncate fields before comparing request digests.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class RankRequest:
    """The *baseline-visible* semantic input at the Stage-ReRank boundary.

    Target IDs, reward labels, candidate-sampler metadata, endpoint credentials,
    and runtime timestamps are deliberately absent.  ``snapshot_id`` is kept
    for provenance but excluded from the semantic digest: two requests with
    byte-identical prompt inputs should hash the same even if replayed later.
    """

    user_id: int
    instruction: str | None
    candidates: tuple[dict[str, Any], ...]
    item_memories: dict[int, str]
    collaborative_facets: tuple[dict[str, Any], ...]
    vanilla_mode: bool = False
    upstream_empty_facets_prompt: bool = False
    snapshot_id: str | None = None

    @classmethod
    def from_stage_rr_inputs(
        cls,
        *,
        user_id: int,
        retrieval_bundle: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        item_mems: Mapping[int, str] | None = None,
        instruction: str | None = None,
        vanilla_mode: bool = False,
        upstream_empty_facets_prompt: bool = False,
        snapshot_id: str | None = None,
    ) -> "RankRequest":
        candidate_rows = tuple(deepcopy(dict(row)) for row in candidates)
        ids = [int(row["id"]) for row in candidate_rows]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("RankRequest requires nonempty, unique candidates")
        if any(row["id"] != item_id for row, item_id in zip(candidate_rows, ids)):
            raise ValueError("Candidate IDs must be integer-valued")
        memories = {int(item_id): deepcopy(value)
                    for item_id, value in (item_mems or {}).items()}
        if any(not isinstance(value, str) for value in memories.values()):
            raise ValueError("Item memories must be text, as expected by LLMReranker")
        if set(memories) - set(ids):
            raise ValueError("Item memory refers to an out-of-set candidate")
        facets = tuple(deepcopy(dict(row)) for row in retrieval_bundle.get("facets", []))
        request = cls(
            user_id=int(user_id), instruction=instruction,
            candidates=candidate_rows, item_memories=memories,
            collaborative_facets=facets, vanilla_mode=bool(vanilla_mode),
            upstream_empty_facets_prompt=bool(upstream_empty_facets_prompt),
            snapshot_id=snapshot_id,
        )
        # Fail before promotion if a captured field cannot be replayed or hashed.
        _canonical_json(request.semantic_payload())
        return request

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RankRequest":
        """Load a persisted request; reject extra reward/target fields."""
        allowed = {
            "user_id", "instruction", "candidates", "item_memories",
            "collaborative_facets", "vanilla_mode", "upstream_empty_facets_prompt",
            "snapshot_id", "rank_request_sha256",
        }
        if set(row) - allowed:
            raise ValueError(f"Unexpected RankRequest fields: {sorted(set(row) - allowed)}")
        request = cls.from_stage_rr_inputs(
            user_id=int(row["user_id"]), instruction=row.get("instruction"),
            candidates=row["candidates"],
            item_mems={int(k): v for k, v in row.get("item_memories", {}).items()},
            retrieval_bundle={"facets": row.get("collaborative_facets", [])},
            vanilla_mode=bool(row.get("vanilla_mode", False)),
            upstream_empty_facets_prompt=bool(row.get("upstream_empty_facets_prompt", False)),
            snapshot_id=row.get("snapshot_id"),
        )
        expected = row.get("rank_request_sha256")
        if expected is not None and expected != request.sha256():
            raise ValueError("RankRequest digest mismatch")
        return request

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "instruction": self.instruction,
            "candidates": deepcopy(list(self.candidates)),
            "item_memories": {str(k): v for k, v in self.item_memories.items()},
            "collaborative_facets": deepcopy(list(self.collaborative_facets)),
            "vanilla_mode": self.vanilla_mode,
            "upstream_empty_facets_prompt": self.upstream_empty_facets_prompt,
        }

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.semantic_payload()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "snapshot_id": self.snapshot_id,
                "rank_request_sha256": self.sha256()}

    def baseline_prompt_kwargs(self) -> dict[str, Any]:
        """Arguments that reproduce the existing LLMReranker prompt exactly."""
        return {
            "user_id": self.user_id,
            "facets": deepcopy(list(self.collaborative_facets)),
            "candidates": deepcopy(list(self.candidates)),
            "item_mems": deepcopy(self.item_memories),
            "instruction": self.instruction,
            "vanilla_mode": self.vanilla_mode,
            "upstream_empty_facets_prompt": self.upstream_empty_facets_prompt,
        }
