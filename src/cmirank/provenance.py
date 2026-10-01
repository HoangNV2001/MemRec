"""Small artifact hashes, compatible with the cluster's Python 3.10."""

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_json_dumps(value: object) -> str:
    """Serialize diagnostics deterministically, preserving set-valued fields.

    Transformers loading reports may contain sets of parameter names. Convert
    only these unordered collections to sorted arrays, never arbitrary objects
    to strings: unknown types must still fail rather than obscure diagnostics.
    """
    def encode_collection(item: object) -> list:
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                      default=encode_collection) + "\n"
