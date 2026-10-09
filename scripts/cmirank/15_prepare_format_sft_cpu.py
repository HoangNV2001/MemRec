#!/usr/bin/env python3
"""Freeze synthetic SFT data/token masks before reclaiming any GPU."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.format_sft import (
    encode_completion, format_code_hashes, format_examples, load_format_config,
    synthetic_request, validate_encoded_row,
)
from src.cmirank.labels import make_labels
from src.cmirank.memory_smoke import object_sha256
from src.cmirank.policy_smoke import load_secondary_inputs
from src.cmirank.prompts import render_direct_prompt, render_step_prompt
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config, path = load_format_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private / "runs" / config["data_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 4
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Fresh scoped CPU preparation on exact clean source required")
    if importlib.metadata.version("transformers") != "5.13.0":
        raise ValueError("Existing isolated tokenizer environment changed")
    started = time.monotonic()
    import torch
    from transformers import AutoTokenizer
    if torch.cuda.is_initialized():
        raise ValueError("CPU preparation initialized CUDA")
    model = private / f"models/Qwen3.5-4B-{config['model_revision'][:7]}-hnv"
    marker = json.loads((model / "download-complete-hnv.json").read_text())
    if marker["model_revision"] != config["model_revision"] or not marker["sha256"]:
        raise ValueError("Pinned official checkpoint is not complete")
    for name, digest in marker["sha256"].items():
        if name != Path(name).name or file_sha256(model / name) != digest:
            raise ValueError("Official checkpoint files changed before SFT")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    run.mkdir()
    raw, tokenized = [], []
    for example in format_examples(config):
        prompt = example.prompt()
        row = {"example_id": example.example_id, "mode": example.mode,
               **encode_completion(tokenizer, prompt, example.completion,
                   max_input=config["max_input_tokens"], max_output=config["max_new_tokens"])}
        validate_encoded_row(row, config)
        raw.append({"example_id": example.example_id, "mode": example.mode,
            "rank_request": example.request.to_dict(), "active_labels": list(example.active_labels),
            "prompt": prompt, "completion": example.completion,
            "completion_role": "uniform_structural_tie_break_not_relevance_label"})
        tokenized.append(row)
        if len(tokenized) == config["smoke_train_examples"]:
            assert all(r["labels"][:r["prompt_tokens"]] == [-100] * r["prompt_tokens"] for r in tokenized)
            print("CPU structural/token-mask smoke: 20/20 PASS; preparing remaining fixed examples", flush=True)
    smoke_rows = tokenized[:config["smoke_train_examples"]]
    assert len(smoke_rows) == 20 and all(r["labels"][:r["prompt_tokens"]] == [-100] * r["prompt_tokens"] for r in smoke_rows)
    holdout = [synthetic_request(config["data_seed"], "holdout", i) for i in range(config["holdout_users"])]
    assert {r["rank_request"]["user_id"] for r in raw}.isdisjoint(r.user_id for r in holdout)
    real_config = json.loads((ROOT / config["real_diagnostic_contract"]).read_text())
    real, provenance = load_secondary_inputs(private / "runs" / real_config["memory_run_id"],
        private / "runs" / real_config["memory_review_id"] / "review.json", expected_review_sha=real_config["memory_review_sha256"])
    audit = []
    for cohort, requests in (("synthetic", holdout), ("secondary_books", real)):
        for request in requests:
            for mode, prompt in (("iterative", render_step_prompt(request, make_labels(range(10)))),
                                 ("direct", render_direct_prompt(request))):
                rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                n = len(tokenizer.encode(rendered, add_special_tokens=False))
                if n > config["max_input_tokens"]:
                    raise ValueError("Functional prompt too long; no truncation")
                audit.append({"cohort": cohort, "user_id": request.user_id, "mode": mode,
                              "tokens": n, "rendered_prompt_sha256": object_sha256(rendered)})
    for name, rows in (("train.jsonl", raw), ("train-tokenized.jsonl", tokenized),
                       ("synthetic-holdout.jsonl", [r.to_dict() for r in holdout])):
        (run / name).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows))
    (run / "diagnostic-token-audit.json").write_text(artifact_json_dumps(audit))
    report = {"status": "FORMAT_SFT_CPU_PREPARATION_PASS", "source_commit": commit,
        "config_sha256": file_sha256(path), "code_sha256": format_code_hashes(ROOT), "slurm_job_id": job,
        "checkpoint_marker_sha256": file_sha256(model / "download-complete-hnv.json"),
        "train_examples": len(raw), "smoke_mask_examples": len(smoke_rows), "holdout_users": len(holdout),
        "prompt_tokens_min": min(r["prompt_tokens"] for r in tokenized),
        "prompt_tokens_max": max(r["prompt_tokens"] for r in tokenized),
        "completion_tokens_min": min(r["completion_tokens"] for r in tokenized),
        "completion_tokens_max": max(r["completion_tokens"] for r in tokenized),
        "diagnostic_prompts_audited": len(audit), "official_eos_token_id": tokenizer.eos_token_id,
        "prompt_loss_mask_verified": True, "synthetic_split_disjoint": True, "official_checkpoint_hashes_verified": True,
        "books_training_data_accessed": False, "secondary_books_diagnostic_inputs_accessed": True,
        "secondary_books_provenance": provenance, "gpu_requested": False, "model_weights_loaded": False,
        "model_updates": 0, "ranking_metrics_computed": False, "training_ready": False,
        "artifact_sha256": {p.name: file_sha256(p) for p in run.iterdir() if p.is_file()},
        "elapsed_seconds": time.monotonic() - started}
    (run / "report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "train_examples", "smoke_mask_examples",
          "holdout_users", "prompt_tokens_min", "prompt_tokens_max", "completion_tokens_min",
          "completion_tokens_max", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
