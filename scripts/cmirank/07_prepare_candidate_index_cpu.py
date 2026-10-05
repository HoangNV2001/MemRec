#!/usr/bin/env python3
"""Pinned metadata encoder: 20-item technical smoke before any full index.

CPU only, inside the authorized Slurm allocation. No user outcome or LLM call.
Output is a sampler index, not a MemRec memory cache or ranking result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cmirank.candidates import digest_key, validate_candidate_contract
from src.cmirank.metadata import read_metadata_texts
from src.cmirank.provenance import artifact_json_dumps, file_sha256

CLUSTER_ROOT = Path("/mnt/data/users/hoangnv242/memrec-hnv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--full-after-smoke", action="store_true")
    args = parser.parse_args()
    contract_path = ROOT / "configs/cmirank/candidate_sampler_v1.json"
    config = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_candidate_contract(config)
    run = args.run_dir.resolve()
    job = os.environ.get("SLURM_JOB_ID", "")
    if (not job.isdigit()
            or subprocess.check_output(["squeue", "-j", job, "-h", "-o", "%u %T %j"],
                                       text=True).strip() != "hoangnv242 RUNNING senvoice-pro-opt"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or not run.is_relative_to(CLUSTER_ROOT / "runs")
            or run.name != config["index_build"]["run_id"]):
        raise RuntimeError("Unexpected allocation, device visibility or run destination")
    if not Path(os.environ.get("HF_HOME", "/invalid")).resolve().is_relative_to(
            CLUSTER_ROOT / "cache"):
        raise RuntimeError("HF cache is outside project-owned storage")
    if any((run / name).exists() for name in ("manifest.json", "smoke.json", "item_ids.npy")):
        raise RuntimeError("Refusing to overwrite index artifacts")
    if shutil.disk_usage(CLUSTER_ROOT).free < 5 * 2**30:
        raise RuntimeError("Less than 5 GiB free for metadata index")
    run.mkdir(exist_ok=True)
    config_hash = file_sha256(contract_path)
    manifest = {
        "status": "RUNNING_METADATA_ENCODER_CPU_ONLY", "training_ready": False,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=ROOT, text=True).strip(),
        "allocation_job_id": job, "candidate_contract_sha256": config_hash,
        "model_id": config["encoder"]["model_id"],
        "model_revision": config["encoder"]["revision"],
        "device": "cpu", "gpu_requested": False,
        "original_candidate_lists_accessed": False,
        "user_interactions_accessed": False,
        "real_llm_requests": 0,
    }
    (run / "manifest.json").write_text(artifact_json_dumps(manifest))
    (run / "candidate-contract.json").write_text(artifact_json_dumps(config))
    started = time.monotonic()
    source = ROOT / "data/processed/instructrec-books/instructrec-books.meta"
    if file_sha256(source) != config["metadata_sha256"]:
        raise RuntimeError("Metadata hash differs from frozen input")
    texts, metadata_audit = read_metadata_texts(
        source, max_characters=config["encoder"]["max_text_characters"])
    ids = sorted(texts)
    if len(ids) < 20:
        raise RuntimeError("Need at least 20 catalog items")
    smoke_ids = sorted(ids, key=lambda item: (
        digest_key(config["seed"], "encoder-smoke", "catalog_item", item), item))[:20]
    manifest.update({"metadata_sha256": config["metadata_sha256"], **metadata_audit})
    (run / "manifest.json").write_text(artifact_json_dumps(manifest))
    print(json.dumps({"phase": "metadata_validated", "catalog_items": len(ids)}), flush=True)

    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    threads = config["index_build"]["cpu_threads"]
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < threads:
        raise RuntimeError("Insufficient CPUs for frozen encoder contract")
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise RuntimeError("Metadata indexing must not expose a GPU")
    model_dir = CLUSTER_ROOT / ("models/all-MiniLM-L6-v2-"
                               f"{config['encoder']['revision'][:7]}-hnv")
    snapshot_download(
        repo_id=config["encoder"]["model_id"], revision=config["encoder"]["revision"],
        local_dir=model_dir, max_workers=2,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model, loading = AutoModel.from_pretrained(
        model_dir, local_files_only=True, output_loading_info=True)
    # Some sentence checkpoints omit BERT's unused pooler. The selected
    # last_hidden_state + masked-mean path must load all actual encoder layers.
    allowed_missing = {"pooler.dense.weight", "pooler.dense.bias"}
    if (set(loading.get("missing_keys", ())) - allowed_missing
            or loading.get("unexpected_keys") or loading.get("mismatched_keys")
            or loading.get("error_msgs")):
        raise RuntimeError("Encoder checkpoint failed loading-integrity gate")
    (run / "encoder-loading.json").write_text(artifact_json_dumps(loading))
    model = model.float().cpu().eval()
    model.requires_grad_(False)
    if model.config.hidden_size != config["encoder"]["dimension"]:
        raise RuntimeError("Encoder dimension differs from contract")

    def encode(batch: list[int]) -> np.ndarray:
        inputs = tokenizer([texts[item] for item in batch], padding=True, truncation=True,
                           max_length=config["encoder"]["max_tokens"], return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()
        if (vectors.dtype != np.float32 or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)):
            raise RuntimeError("Nonfinite or nonunit metadata vector")
        return vectors

    smoke_started = time.monotonic()
    first = encode(smoke_ids)
    reordered = encode(list(reversed(smoke_ids)))[::-1]
    if not np.allclose(first, reordered, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Encoder output is not invariant to batch order")
    smoke = {
        "status": "ENCODER_TECHNICAL_SMOKE_PASS_NOT_RANKING_RESULT",
        "items": len(smoke_ids), "sample_item_ids": smoke_ids,
        "candidate_contract_sha256": config_hash,
        "sample_text_sha256": hashlib.sha256(
            json.dumps([texts[item] for item in smoke_ids], ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")).hexdigest(),
        "model_id": config["encoder"]["model_id"],
        "model_revision": config["encoder"]["revision"],
        "dimension": first.shape[1], "batch_order_max_abs_difference":
            float(np.max(np.abs(first - reordered))),
        "elapsed_seconds": time.monotonic() - smoke_started,
        "gpu_used": False, "training_ready": False,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "encoder_file_sha256": {path.name: file_sha256(path) for path in sorted(model_dir.iterdir())
                                if path.is_file()},
        "loading_diagnostics_sha256": file_sha256(run / "encoder-loading.json"),
    }
    (run / "smoke.json").write_text(artifact_json_dumps(smoke))
    print(json.dumps({"phase": "encoder_smoke_pass", "items": 20,
                      "seconds": smoke["elapsed_seconds"]}), flush=True)
    if not args.full_after_smoke:
        manifest["status"] = "ENCODER_SMOKE_ONLY_COMPLETE_NO_FULL_INDEX"
        (run / "manifest.json").write_text(artifact_json_dumps(manifest))
        return

    vectors_path = run / "vectors.npy"
    vectors = np.lib.format.open_memmap(
        vectors_path, mode="w+", dtype=np.float32,
        shape=(len(ids), config["encoder"]["dimension"]))
    positions = {item: index for index, item in enumerate(ids)}
    for item, vector in zip(smoke_ids, first):
        vectors[positions[item]] = vector
    smoke_set = set(smoke_ids)
    pending_ids = [item for item in ids if item not in smoke_set]
    batch_size = config["index_build"]["batch_size"]
    index_started = time.monotonic()
    for offset in range(0, len(pending_ids), batch_size):
        batch = pending_ids[offset:offset + batch_size]
        vectors[[positions[item] for item in batch]] = encode(batch)
        if offset % (batch_size * 32) == 0:
            print(json.dumps({"phase": "encoding", "completed": offset + len(batch) + 20,
                              "total": len(ids), "seconds": time.monotonic() - index_started}),
                  flush=True)
    vectors.flush()
    del vectors
    np.save(run / "item_ids.npy", np.asarray(ids, dtype=np.int64), allow_pickle=False)
    manifest.update({
        "status": "COMPLETE_METADATA_INDEX_NOT_MEMORY_OR_PPO_PROMOTION",
        "index_shape": [len(ids), config["encoder"]["dimension"]],
        "dtype": "float32", "smoke_sha256": file_sha256(run / "smoke.json"),
        "item_ids_sha256": file_sha256(run / "item_ids.npy"),
        "vectors_sha256": file_sha256(vectors_path),
        "elapsed_seconds": time.monotonic() - started,
        "index_seconds": time.monotonic() - index_started,
    })
    (run / "manifest.json").write_text(artifact_json_dumps(manifest))
    print(json.dumps({"phase": "index_complete", "items": len(ids),
                      "seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
