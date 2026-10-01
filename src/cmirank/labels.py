"""Opaque candidate labels and exact bidirectional mapping."""

from __future__ import annotations

from typing import Sequence


def make_labels(candidate_ids: Sequence[int]) -> tuple[str, ...]:
    if not candidate_ids or len(candidate_ids) > 100:
        raise ValueError("Expected between 1 and 100 candidates")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate candidate ID")
    return tuple(f"C{index:02d}" for index in range(len(candidate_ids)))


def label_to_item(candidate_ids: Sequence[int]) -> dict[str, int]:
    labels = make_labels(candidate_ids)
    return dict(zip(labels, candidate_ids))
