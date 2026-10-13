#!/usr/bin/env python3
"""Source-bound20 CPU protocol proxies before generated-rollout GPU handoff."""

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.corrected_rollout import (corrected_code_hashes, corrected_server_options,
    generated_request, load_corrected_config, native_cpu_rehearsal)
from src.cmirank.ppo_compat import load_compat_config, validate_compat_rows
from src.cmirank.ppo_parity import verify_fla_overlay
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.cuda_toolchain import verify_cuda_toolchain
from src.cmirank.causal_conv import verify_causal_overlay
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_corrected_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / c["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 4
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or os.environ.get("PYTHONPATH") != str(private / "overlays" / compat["overlay_name"])):
        raise ValueError("Fresh exact-source CPU preparation required")
    started = time.monotonic()
    for n, v in runtime["versions"].items():
        if importlib.metadata.version(n) != v:
            raise ValueError("Pinned runtime changed")
    parent = private / "runs" / c["parent_prepare_run_id"]
    diagnostic = private / "runs" / c["parent_diagnostic_run_id"]
    if (file_sha256(parent / "report.json") != c["parent_prepare_report_sha256"]
            or file_sha256(diagnostic / "report.json") != c["parent_diagnostic_report_sha256"]
            or file_sha256(parent / "tokens.jsonl") != c["tokens_sha256"]):
        raise ValueError("Reviewed parents changed; preserve old FAILs")
    old = json.loads((parent / "report.json").read_text())
    cleanup = json.loads((diagnostic / "cleanup.json").read_text())
    if not cleanup["gpu_released"] or not cleanup["child_exited"] or not cleanup["allocation_still_running"]:
        raise ValueError("Parent cleanup must be verified")
    rows = list(map(json.loads, (parent / "tokens.jsonl").read_text().splitlines()))
    validate_compat_rows(rows, compat)
    verify_fla_overlay(private, c)
    verify_cuda_toolchain(private, ROOT, c)
    verify_causal_overlay(private, ROOT, c)
    checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
    if set(p.name for p in checkpoint.iterdir() if p.is_file()) != set(compat["checkpoint_files_sha256"]):
        raise ValueError("Only the unchanged reviewed initializer allowed")
    for n, sha in compat["checkpoint_files_sha256"].items():
        if file_sha256(checkpoint / n) != sha:
            raise ValueError("Reviewed weights changed")
    for n, sha in old["overlay_sha256"].items():
        if file_sha256(private / "overlays" / compat["overlay_name"] / n) != sha:
            raise ValueError("FA2 overlay changed")
    for n, sha in old["native_source_sha256"].items():
        if file_sha256(Path(sys.prefix) / n) != sha:
            raise ValueError("Pinned source changed")
    os.environ.update(c["server_environment"])
    import torch
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.cli_args import FlexibleArgumentParser, make_arg_parser
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import ModelConfig
    from vllm.v1.sample.sampler import Sampler
    from verl.workers.config.model import HFModelConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.rollout import RolloutConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != 248046:
        raise ValueError("Official tokenizer EOS changed")
    native_model = HFModelConfig(path=str(checkpoint), load_tokenizer=False,
        use_remove_padding=False, use_fused_kernels=False, enable_gradient_checkpointing=False)
    engine = FSDPEngineConfig(strategy="fsdp2", model_dtype="bfloat16", forward_only=True,
        use_torch_compile=False, use_dynamic_bsz=False, infer_micro_batch_size_per_gpu=1)
    if native_model.lora_rank or not engine.forward_only or engine.model_dtype != c["model_dtype"]:
        raise ValueError("Native no-update profile changed")
    cli = make_arg_parser(FlexibleArgumentParser())
    serving = cli.parse_args(["--model", str(checkpoint), *corrected_server_options(c)])
    if serving.logprobs_mode != "processed_logprobs" or serving.tensor_parallel_size != 1:
        raise ValueError("Sampled behavior probabilities/TP1 mode required")
    probes = []
    for row in rows:
        payload = generated_request(row, c)
        native = CompletionRequest(**payload).to_sampling_params(max_tokens=c["max_new_tokens"])
        if (native.temperature != 0.9 or native.top_p != 1.0 or native.top_k != -1
                or native.logprobs != 0 or native.ignore_eos or native.max_tokens != 64):
            raise ValueError("Native request altered the stochastic full-support distribution")
        probes.append(payload)
    rehearsal = native_cpu_rehearsal(c)
    if torch.cuda.is_initialized():
        raise ValueError("CPU preparation initialized CUDA")
    run.mkdir()
    (run / "tokens.jsonl").write_bytes((parent / "tokens.jsonl").read_bytes())
    (run / "requests.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in probes))
    classes = (CompletionRequest, EngineArgs, ModelConfig, Sampler, HFModelConfig,
               FSDPEngineConfig, RolloutConfig, FSDPEngineWithLMHead, compute_rollout_correction_weights)
    native_source = {**old["native_source_sha256"], **{str(Path(inspect.getfile(x)).relative_to(Path(sys.prefix))):
        file_sha256(Path(inspect.getfile(x))) for x in classes}}
    report = {"status": "CORRECTED_ROLLOUT_CPU_PREPARATION_PASS_GENERATED_GPU_GATES_REMAIN",
        "source_commit": commit, "config_sha256": file_sha256(path), "code_sha256": corrected_code_hashes(ROOT),
        "slurm_job_id": job, "samples": 20, "eos_token_id": tokenizer.eos_token_id,
        "native_schema_probes": len(probes), "cpu_rehearsal": rehearsal,
        "native_source_sha256": native_source, "overlay_sha256": old["overlay_sha256"],
        "checkpoint_stat": {n: {"size": (checkpoint/n).stat().st_size,
            "mtime_ns": (checkpoint/n).stat().st_mtime_ns, "inode": (checkpoint/n).stat().st_ino}
            for n in compat["checkpoint_files_sha256"]},
        "artifact_sha256": {n: file_sha256(run/n) for n in ("tokens.jsonl", "requests.jsonl")},
        "physical_requests": 0, "gpu_requested": False, "training_ready": False,
        "full_weights_loaded": False, "optimizer_updates": 0, "elapsed_seconds": time.monotonic()-started}
    (run / "report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "samples", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
