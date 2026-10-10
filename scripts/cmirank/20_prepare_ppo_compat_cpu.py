#!/usr/bin/env python3
"""Verify full-role smoke inputs/kernel/checkpoint BEFORE any GPU handoff."""

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
from src.cmirank.format_sft import encode_completion, synthetic_request
from src.cmirank.labels import make_labels
from src.cmirank.ppo_compat import compat_code_hashes, load_compat_config, validate_compat_rows
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.prompts import render_step_prompt
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config, path = load_compat_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / config["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 4
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or os.environ.get("PYTHONPATH") != str(private / "overlays" / config["overlay_name"])):
        raise ValueError("Fresh exact-source private CPU preparation required")
    started = time.monotonic()
    for name, version in runtime["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned PPO environment changed")
    for key in ("runtime", "kernel", "sft", "sft_data"):
        directory = private / "runs" / config[f"{key}_run_id"]
        if file_sha256(directory / "report.json") != config[f"{key}_report_sha256"]:
            raise ValueError(f"Reviewed {key} receipt changed")
    kernel_dir = private / "runs" / config["kernel_run_id"]
    kernel = json.loads((kernel_dir / "report.json").read_text())
    if (kernel["status"] != "FA2_CPU_BUILD_AND_IMPORT_PASS_CUDA_AND_WORKER_GATES_REMAIN"
            or kernel["existing_environment_mutated"] or kernel["gpu_requested"]
            or kernel["overlay"] != os.environ["PYTHONPATH"]
            or kernel["wheel_sha256"] != config["kernel_wheel_sha256"]
            or file_sha256(kernel_dir / "wheels" / kernel["wheel_name"]) != config["kernel_wheel_sha256"]
            or any(file_sha256(kernel_dir / n) != sha for n, sha in kernel["artifact_sha256"].items())):
        raise ValueError("Native kernel/source/import receipt does not match")
    overlay = private / "overlays" / config["overlay_name"]
    checkpoint = private / "runs" / config["sft_run_id"] / "checkpoint"
    if set(p.name for p in checkpoint.iterdir() if p.is_file()) != set(config["checkpoint_files_sha256"]):
        raise ValueError("Warm initializer checkpoint file set changed")
    for name, sha in config["checkpoint_files_sha256"].items():
        if file_sha256(checkpoint / name) != sha:
            raise ValueError("Reviewed full SFT initializer weights changed")
    import torch
    import flash_attn
    from transformers import AutoConfig, AutoTokenizer
    from verl.utils.model import AutoModelForVision2Seq, get_hf_auto_model_class
    if torch.cuda.is_initialized() or flash_attn.__version__ != "2.8.3":
        raise ValueError("CPU preparation initialized CUDA or uses wrong kernel")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    hf_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if hf_config.model_type != "qwen3_5" or tokenizer.eos_token_id != 248046:
        raise ValueError("Official Qwen architecture/tokenizer changed")
    if (type(hf_config) not in AutoModelForVision2Seq._model_mapping.keys()
            or get_hf_auto_model_class(hf_config) is not AutoModelForVision2Seq):
        raise ValueError("Native VeRL actor/critic auto-model registrations do not accept the full Qwen config")
    old_data = private / "runs" / config["sft_data_run_id"]
    old_report = json.loads((old_data / "report.json").read_text())
    holdout_path = old_data / "synthetic-holdout.jsonl"
    if file_sha256(holdout_path) != old_report["artifact_sha256"][holdout_path.name]:
        raise ValueError("Fixed synthetic holdout changed")
    saved = [json.loads(line) for line in holdout_path.read_text().splitlines()]
    audit_path = old_data / "diagnostic-token-audit.json"
    if file_sha256(audit_path) != old_report["artifact_sha256"][audit_path.name]:
        raise ValueError("Reviewed frozen prefix token audit changed")
    old_audit = json.loads(audit_path.read_text())
    rows = []
    for i in range(config["samples"]):
        request = synthetic_request(config["synthetic_seed"], "holdout", i)
        if request.to_dict() != saved[i]:
            raise ValueError("Compatibility is not the existing disjoint synthetic holdout")
        prompt = render_step_prompt(request, make_labels(range(10)))
        row = {"user_id": request.user_id, "snapshot_id": request.snapshot_id,
               **encode_completion(tokenizer, prompt, f"<answer>C{i % 10:02d}</answer>",
                   max_input=config["max_input_tokens"], max_output=config["max_new_tokens"])}
        prior = next(x for x in old_audit if x["cohort"] == "synthetic"
                     and x["mode"] == "iterative" and x["user_id"] == request.user_id)
        if row["rendered_prompt_sha256"] != prior["rendered_prompt_sha256"] or row["prompt_tokens"] != prior["tokens"]:
            raise ValueError("New tokenizer/runtime changed the frozen inference prefix")
        rows.append(row)
    validate_compat_rows(rows, config)
    if torch.cuda.is_initialized():
        raise ValueError("CUDA initialized during CPU token preparation")
    run.mkdir()
    (run / "tokens.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    report = {"status": "FULL_ROLE_COMPAT_CPU_PREPARATION_PASS_GPU_GATES_REMAIN",
        "source_commit": commit, "config_sha256": file_sha256(path), "code_sha256": compat_code_hashes(ROOT),
        "slurm_job_id": job, "samples": len(rows), "head_dim": hf_config.text_config.head_dim,
        "overlay_sha256": {str(p.relative_to(overlay)): file_sha256(p) for p in overlay.rglob("*")
                           if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"},
        "checkpoint_files_sha256": config["checkpoint_files_sha256"], "prompt_prefix_parity_with_reviewed_SFT": True,
        "native_actor_and_critic_factory_config_registered": True,
        "checkpoint_stat": {name: {"size": (checkpoint / name).stat().st_size,
                            "mtime_ns": (checkpoint / name).stat().st_mtime_ns,
                            "inode": (checkpoint / name).stat().st_ino} for name in config["checkpoint_files_sha256"]},
        "max_prompt_tokens": max(r["prompt_tokens"] for r in rows), "gpu_requested": False,
        "model_weights_loaded": False, "books_outcomes_accessed": False, "training_ready": False,
        "artifact_sha256": {"tokens.jsonl": file_sha256(run / "tokens.jsonl")},
        "elapsed_seconds": time.monotonic() - started}
    (run / "report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "samples", "max_prompt_tokens", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
