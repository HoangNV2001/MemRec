#!/usr/bin/env python3
"""Pinned official checkpoint download; CPU-only, authorized Slurm step only."""

import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Download requires an authorized Slurm step")
    root = Path("/mnt/data/users/anhnct/memrec-hnv")
    if not Path(os.environ.get("HF_HOME", "")).resolve().is_relative_to(root / "cache"):
        raise RuntimeError("HF cache is outside MemRec-owned storage")
    contract = json.loads(Path("configs/cmirank/policy_model_v1.json").read_text())
    revision = contract["model_revision"]
    destination = root / f"models/Qwen3.5-4B-{revision[:7]}-hnv"
    marker = destination / "download-complete-hnv.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["model_revision"] != revision:
            raise RuntimeError("Existing checkpoint has a different revision")
        if any(not (destination / name).is_file()
               or (destination / name).stat().st_size != size
               for name, size in saved["file_bytes"].items()):
            raise RuntimeError("Existing checkpoint is incomplete")
        print(json.dumps({"status": "already_downloaded", "model_revision": revision}))
        return
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise RuntimeError("Less than 40 GiB free; refusing preparation")
    snapshot_download(
        repo_id=contract["model_id"], revision=revision,
        local_dir=destination, max_workers=2,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt", "vocab.json", "LICENSE"],
    )
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    required = ["config.json", "tokenizer_config.json", "tokenizer.json", *files]
    if any(not (destination / name).is_file() for name in required):
        raise RuntimeError("Missing checkpoint file")
    if sum((destination / name).stat().st_size for name in files) < 7_000_000_000:
        raise RuntimeError("Checkpoint weight size is incomplete")
    record = {
        "model_id": contract["model_id"], "model_revision": revision,
        "file_bytes": {name: (destination / name).stat().st_size for name in required},
        "sha256": {name: sha256(destination / name) for name in required},
    }
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "downloaded", "model_revision": revision,
                      "weight_shards": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
