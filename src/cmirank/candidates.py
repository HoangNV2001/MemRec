"""Training-only, deterministic 3+3+3 negatives; never evaluation retrieval.

The positive anchors popularity/semantic sampling, as declared in the protocol.
Only shuffled item identities are policy inputs. Component labels, similarity,
and the positive identity are reward/audit-side data, not prompt features.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
from typing import Collection, Mapping, Sequence

import numpy as np


def validate_candidate_contract(config: Mapping) -> None:
    """Reject a declared contract the v1 implementation cannot execute."""
    fixed = {
        "schema_version": 1,
        "composition": {"uniform": 3, "popularity_matched": 3, "semantic_hard": 3},
        "selection_order": ["uniform", "popularity_matched", "semantic_hard"],
        "eligible_pool": "static_metadata_with_nonempty_title",
        "metadata_duplicate_policy": "collapse_identical_asin_title_description_fail_on_conflict",
        "popularity": "interaction_count_in_common_1797_query_prefix_snapshot",
        "popularity_bucket": "floor_log2_count_plus_one",
        "popularity_fallback": "nearest_available_bucket_then_lower_bucket",
        "shuffle": "sha256_seed_episode_id_slot_not_target_identity_or_component",
    }
    encoder_fixed = {
        "pooling": "attention_mask_mean_then_l2_normalization",
        "text_template": "Title: {title}\nDescription: {description}",
        "whitespace": "collapse", "dtype": "float32", "device": "cpu",
    }
    if (any(config.get(key) != value for key, value in fixed.items())
            or any(config.get("encoder", {}).get(key) != value
                   for key, value in encoder_fixed.items())
            or not config.get("seed")
            or config.get("index_build", {}).get("smoke_items") != 20):
        raise ValueError("Unsupported candidate contract; do not relabel a v1 run")


def digest_key(seed: str, episode_id: str, namespace: str, value: int) -> bytes:
    """Stable across processes, unlike Python hash()."""
    return hashlib.sha256(
        f"{seed}\0{episode_id}\0{namespace}\0{value}".encode("utf-8")
    ).digest()


def popularity_from_snapshot(histories: Mapping[int, Sequence[int]]) -> Counter:
    """Count allowed interactions, not a user's withheld suffix or outcomes."""
    return Counter(int(item) for items in histories.values() for item in items)


def popularity_bucket(count: int) -> int:
    if count < 0:
        raise ValueError("Popularity cannot be negative")
    return (int(count) + 1).bit_length() - 1


def item_text(title: str, description: str, *, max_characters: int) -> str:
    if not title.strip() or max_characters < 16:
        raise ValueError("Need a nonempty title and a bounded text template")
    title = " ".join(title.split())
    description = " ".join(description.split())
    return f"Title: {title}\nDescription: {description}"[:max_characters]


@dataclass(frozen=True)
class CandidateSet:
    episode_id: str
    candidate_ids: tuple[int, ...]
    positive_item_id: int
    negative_components: Mapping[str, tuple[int, ...]]

    def policy_input(self) -> dict:
        """Allowlist: never serialize the whole audit dataclass to the policy."""
        return {"episode_id": self.episode_id,
                "candidate_ids": list(self.candidate_ids)}

    def reward_audit(self) -> dict:
        return {"positive_item_id": self.positive_item_id,
                "positive_position": self.candidate_ids.index(self.positive_item_id),
                "negative_components": {key: list(value)
                                        for key, value in self.negative_components.items()}}


def shuffle_candidates(items: Sequence[int], *, seed: str, episode_id: str) -> tuple[int, ...]:
    """Shuffle all identities identically, independently of target/component."""
    canonical = sorted(items)
    if len(canonical) != 10 or len(set(canonical)) != 10:
        raise ValueError("Need exactly ten unique candidate identities")
    slots = sorted(range(10), key=lambda slot: (
        digest_key(seed, episode_id, "shuffle_slot", slot), slot))
    return tuple(canonical[slot] for slot in slots)


class UniformWarmupSampler:
    """One uniform-nine policy for *all* warm-up events; no encoder/labels.

    A positive may have an empty title, but must have a known metadata identity.
    The negative pool remains the same frozen eligible catalog as PPO sampling.
    """

    def __init__(self, item_ids: Sequence[int], catalog_ids: Collection[int], *, seed: str):
        ids = tuple(int(item) for item in item_ids)
        catalog = frozenset(int(item) for item in catalog_ids)
        if (not seed or len(ids) < 10 or ids != tuple(sorted(set(ids)))
                or any(item < 0 for item in catalog) or not set(ids) <= catalog):
            raise ValueError("Need a sorted unique catalog subset and explicit seed")
        self.item_ids, self.catalog_ids, self.seed = ids, catalog, seed

    def sample(self, episode_id: str, positive: int, allowed_prefix: Collection[int], *,
               additionally_forbidden: Collection[int] = ()) -> CandidateSet:
        if not episode_id or positive not in self.catalog_ids:
            raise ValueError("Warm-up positive lacks a known static metadata identity")
        if positive in allowed_prefix:
            raise ValueError("Warm-up positive is not novel to its graph prefix")
        blocked = {positive, *allowed_prefix, *additionally_forbidden}
        eligible = (item for item in self.item_ids if item not in blocked)
        uniform = heapq.nsmallest(9, eligible, key=lambda item: (
            digest_key(self.seed, episode_id, "uniform", item), item))
        if len(uniform) != 9:
            raise ValueError("Not enough eligible unique warm-up negatives")
        ordered = shuffle_candidates([positive, *uniform], seed=self.seed, episode_id=episode_id)
        return CandidateSet(episode_id, ordered, positive, {"uniform": tuple(uniform)})


class MixedCandidateSampler:
    """Frozen catalog vectors must align with sorted, unique item IDs.

    No random fallback for semantic negatives. Incomplete/nonfinite/nonunit
    embeddings are errors, never a reason to silently change the sampler.
    """

    def __init__(self, item_ids: Sequence[int], vectors: np.ndarray,
                 popularity: Mapping[int, int], *, seed: str):
        ids = np.asarray(item_ids, dtype=np.int64)
        if (not seed or ids.ndim != 1 or len(ids) < 10
                or np.any(ids < 0) or np.any(ids[1:] <= ids[:-1])):
            raise ValueError("Need a seed and sorted unique nonnegative catalog IDs")
        if (vectors.ndim != 2 or vectors.shape[0] != len(ids)
                or vectors.shape[1] < 1 or vectors.dtype != np.float32
                or not np.isfinite(vectors).all()):
            raise ValueError("Embedding index must be finite float32 with matching rows")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Embedding index must have unit L2 norm")
        if any(count < 0 for count in popularity.values()):
            raise ValueError("Popularity cannot be negative")
        self.item_ids = ids
        self.vectors = vectors
        self.popularity = dict(popularity)
        self.seed = seed
        self.positions = {int(item): pos for pos, item in enumerate(ids)}
        self.buckets = np.array([popularity_bucket(popularity.get(int(item), 0))
                                 for item in ids], dtype=np.int32)

    def sample(self, episode_id: str, positive: int,
               allowed_prefix: Collection[int], *,
               additionally_forbidden: Collection[int] = ()) -> CandidateSet:
        if not episode_id or positive not in self.positions:
            raise ValueError("Episode ID and positive catalog metadata are required")
        if positive in allowed_prefix:
            raise ValueError("Positive is not novel to its allowed prefix")
        blocked = set(int(item) for item in allowed_prefix)
        blocked.update(int(item) for item in additionally_forbidden)
        blocked.add(positive)
        eligible = [int(item) for item in self.item_ids if item not in blocked]
        if len(eligible) < 9:
            raise ValueError("Not enough eligible unique negatives")

        uniform = heapq.nsmallest(3, eligible, key=lambda item: (
            digest_key(self.seed, episode_id, "uniform", item), item))
        blocked.update(uniform)
        bucket = popularity_bucket(self.popularity.get(positive, 0))
        popular = heapq.nsmallest(3, (item for item in eligible if item not in blocked),
                         key=lambda item: (
                             abs(int(self.buckets[self.positions[item]]) - bucket),
                             int(self.buckets[self.positions[item]]),
                             digest_key(self.seed, episode_id, "popularity", item), item))
        blocked.update(popular)
        scores = self.vectors @ self.vectors[self.positions[positive]]
        # Stable score ordering; hash-break exact ties. O(pool) argpartition
        # avoids sorting the whole catalog by vector score for every episode.
        eligible_positions = np.array([self.positions[item] for item in eligible
                                       if item not in blocked], dtype=np.int64)
        eligible_scores = scores[eligible_positions]
        threshold = np.partition(eligible_scores, -3)[-3]
        shortlist = eligible_positions[eligible_scores >= threshold]
        semantic = sorted((int(self.item_ids[pos]) for pos in shortlist),
                          key=lambda item: (
                              -float(scores[self.positions[item]]),
                              digest_key(self.seed, episode_id, "semantic_tie", item), item))[:3]
        # Shuffle slot numbers after canonically sorting *all* IDs: no positive
        # marker, target identity, or negative-component label enters this key.
        ordered = shuffle_candidates([positive, *uniform, *popular, *semantic],
                                     seed=self.seed, episode_id=episode_id)
        return CandidateSet(episode_id, ordered, positive,
                            {"uniform": tuple(uniform),
                             "popularity_matched": tuple(popular),
                             "semantic_hard": tuple(semantic)})
