"""Static metadata identity audit for a frozen sampler text index."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .candidates import item_text


def read_metadata_texts(source: Path, *, max_characters: int) -> tuple[dict[int, str], dict]:
    """Collapse only identical repeated baseline fields; conflict is an error.

    The baseline overwrites dictionary entries on repeated IDs. Identical
    ASIN/title/description rows are therefore semantically equivalent to one
    row; conflicting rows must be investigated, not selected by last-row wins.
    """
    csv.field_size_limit(16 * 1024 * 1024)
    fingerprints: dict[int, bytes] = {}
    texts: dict[int, str] = {}
    rows = duplicate_rows = 0
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"item_id", "asin", "title", "description"}.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected static metadata schema")
        for row in reader:
            rows += 1
            item_id = int(row["item_id"])
            if item_id < 0:
                raise ValueError("Negative metadata item identity")
            fields = [row["asin"], row["title"], row["description"]]
            if any(not isinstance(value, str) for value in fields):
                raise ValueError("Malformed static metadata row")
            fingerprint = hashlib.sha256(json.dumps(
                fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).digest()
            if item_id in fingerprints:
                if fingerprints[item_id] != fingerprint:
                    raise ValueError(f"Conflicting metadata rows for item_id {item_id}")
                duplicate_rows += 1
                continue
            fingerprints[item_id] = fingerprint
            if row["title"].strip():
                texts[item_id] = item_text(row["title"], row["description"],
                                           max_characters=max_characters)
    return texts, {"metadata_rows": rows, "metadata_unique_ids": len(fingerprints),
                   "identical_duplicate_rows_collapsed": duplicate_rows,
                   "conflicting_duplicate_rows": 0,
                   "eligible_catalog_items": len(texts),
                   "empty_title_items_excluded": len(fingerprints) - len(texts)}
