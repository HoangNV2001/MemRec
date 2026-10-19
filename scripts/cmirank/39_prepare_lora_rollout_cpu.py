#!/usr/bin/env python3
"""CPU-only20 sampler schemas, native checkpoint audit and exact PEFT export."""

import argparse
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.lora_joint import reviewed_lora_parent, native_model_config, native_engine_config
from src.cmirank.lora_checkpoint import checkpoint_budget
from src.cmirank.lora_rollout import (load_lora_rollout_config, reviewed_checkpoint_parent,
    checkpoint_directory, lora_server_options, rollout_code_hashes, export_adapter_cpu, rollout_host_budget)
from src.cmirank.optimizer_smoke import load_optimizer_config, verified_parents
from src.cmirank.corrected_rollout import generated_request, native_microbatch, native_cpu_rehearsal
from src.cmirank.host_memory import cpu_memory_budget
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_parity import completion_request
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_lora_rollout_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private/"runs"/c["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(sys.prefix).resolve() != private/"envs"/runtime["environment_name"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Fresh clean-source CUDA-hidden CPU export required")
    began = time.monotonic()
    for name, version in {**runtime["versions"], "peft": "0.21.2"}.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned native runtime changed")
    host = rollout_host_budget(job, c)
    if not host["assessment"]["within_estimated_budget"]:
        raise ValueError("Original CPU host admission margin missing")
    parent, previous = reviewed_checkpoint_parent(private, ROOT, c)
    scope = reviewed_lora_parent(private, ROOT, c)
    observations, _, source = verified_parents(private, load_optimizer_config(ROOT)[0])
    rows = [{**r, "snapshot_id": "synthetic-format-holdout-v1"} for r in observations]
    checkpoint = private/"runs"/compat["sft_run_id"]/"checkpoint"
    for name, sha in compat["checkpoint_files_sha256"].items():
        if file_sha256(checkpoint/name) != sha:
            raise ValueError("Frozen initializer changed")
    import torch
    import torch.distributed as dist
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != 248046:
        raise ValueError("Official EOS changed")
    serving = make_arg_parser(FlexibleArgumentParser()).parse_args([
        "--model", str(checkpoint), *lora_server_options(c, run/"actor-adapter")])
    if (not serving.enable_lora or serving.max_lora_rank != 32 or serving.max_loras != 1
            or serving.tensor_parallel_size != 1 or serving.logprobs_mode != "processed_logprobs"):
        raise ValueError("Actual native vLLM LoRA/TP/probability schema differs")
    requests = []
    for row in rows:
        data = native_microbatch(torch.tensor(row["input_ids"]), c, torch)
        packed, _ = FSDPEngineWithLMHead.prepare_model_inputs(
            SimpleNamespace(use_ulysses_sp=False, pad_to_length=False), data)
        if packed["input_ids"][0].tolist() != row["input_ids"]:
            raise ValueError("Native full-sequence packing changed frozen tokens")
        for model in (c["served_model_name"], c["served_adapter_name"]):
            payload = {**completion_request(row, c), "model": model}
            CompletionRequest(**payload)
            requests.append(payload)
        payload = {**generated_request(row, c), "model": c["served_adapter_name"]}
        params = CompletionRequest(**payload).to_sampling_params(max_tokens=c["max_new_tokens"])
        if params.temperature != 0.9 or params.top_p != 1.0 or params.logprobs != 0 or params.ignore_eos:
            raise ValueError("Actual native full-support sampling distribution changed")
        requests.append(payload)
    rehearsal = native_cpu_rehearsal(c)
    for role in ("actor", "critic"):
        if native_model_config(private, c, compat, scope, role).lora_rank != 32 or native_engine_config().model_dtype != "fp32":
            raise ValueError("Native FP32 LoRA training-role schemas changed")
    total = int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.total",
        "--format=csv,noheader,nounits"], text=True).strip())*2**20
    import shutil
    budget = checkpoint_budget(scope["roles"], total, host["allocation_memory"]["maximum_bytes"],
        host["stats_bytes"]["anon"], parent["peak_worker_rss_kib"]*1024,
        shutil.disk_usage(private/"runs").free, c)
    if not budget["within_estimated_budget"]:
        raise ValueError("Sequential native cold-restore staging exceeds unchanged resource caps")
    run.mkdir()
    def interrupted(signum, frame):
        raise InterruptedError(f"CPU adapter export interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # DTensor metadata is restored by ordinary torch.load under a bounded
        # one-rank CPU process group; never rewrite pickle/device-mesh metadata.
        dist.init_process_group("gloo", init_method="file://"+str(run/"cpu-export-rendezvous"), rank=0, world_size=1)
        hf = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        with init_empty_weights():
            base = Qwen3_5ForConditionalGeneration._from_config(hf, dtype=torch.float32).to(dtype=torch.float32)
            model = get_peft_model(base, LoraConfig(r=32, lora_alpha=32, lora_dropout=0,
                target_modules=sorted(scope["targets"]["actor"]), task_type="CAUSAL_LM", bias="none"))
        audit = json.loads((private/"runs"/c["checkpoint_parent_run_id"]/"checkpoint-actor-audit.json").read_text())
        export = export_adapter_cpu(checkpoint_directory(private, c)/"actor/model_world_size_1_rank_0.pt",
            model, scope["roles"]["actor"], audit, run/"actor-adapter")
        del model, base
        gc.collect()
        if torch.cuda.is_initialized():
            raise ValueError("CPU export initialized CUDA")
        (run/"tokens.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
        (run/"requests.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in requests))
        classes = (Qwen3_5ForConditionalGeneration, CompletionRequest, FSDPEngineWithLMHead, LoraConfig, torch.load)
        native_sources = {**previous["native_source_sha256"],
            **{str(Path(inspect.getfile(x)).relative_to(Path(sys.prefix))): file_sha256(Path(inspect.getfile(x))) for x in classes}}
        for name, sha in native_sources.items():
            if file_sha256(Path(sys.prefix)/name) != sha:
                raise ValueError("Reviewed native runtime source changed")
        native_vllm_model = Path(sys.prefix)/"lib/python3.10/site-packages/vllm/model_executor/models/qwen3_5.py"
        native_sources[str(native_vllm_model.relative_to(Path(sys.prefix)))] = file_sha256(native_vllm_model)
        result = {"status": "LORA_ROLLOUT_CPU20_EXPORT_PASS_GPU_GATE_REMAINS", "source_commit": commit,
            "config_sha256": file_sha256(path), "code_sha256": rollout_code_hashes(ROOT),
            "samples_per_role": 20, "request_schema_probes": len(requests), "adapter_export": export,
            "gpu_requested": False, "optimizer_updates": 0, "training_ready": False,
            "physical_requests": 0, "books_outcomes_accessed": False, "full_model_instantiated": False,
            "native_checkpoint_cpu_mmap_used": True, "cpu_rehearsal": rehearsal,
            "roles": scope["roles"], "targets": scope["targets"], "eos_token_id": tokenizer.eos_token_id,
            "host_budget": host, "host_budget_after": rollout_host_budget(job, c),
            "checkpoint_budget": budget,
            "native_source_sha256": native_sources, "overlay_sha256": source["overlay_sha256"],
            "checkpoint_stat": previous["checkpoint_stat"],
            "native_checkpoint_stat": {n: {"size": (checkpoint_directory(private, c)/n).stat().st_size,
                "mtime_ns": (checkpoint_directory(private, c)/n).stat().st_mtime_ns,
                "inode": (checkpoint_directory(private, c)/n).stat().st_ino} for n in c["native_checkpoint_sha256"]},
            "artifact_sha256": {str(p.relative_to(run)): file_sha256(p) for p in run.rglob("*")
                if p.is_file() and "rendezvous" not in p.name}, "elapsed_seconds": time.monotonic()-began}
        (run/"report.json").write_text(artifact_json_dumps(result))
        print(json.dumps({k: result[k] for k in ("status", "samples_per_role", "adapter_export", "elapsed_seconds")}), flush=True)
    except BaseException as error:
        (run/"failure.json").write_text(artifact_json_dumps({"error_type": type(error).__name__, "error": str(error),
            "gpu_requested": False, "training_ready": False, "elapsed_seconds": time.monotonic()-began}))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
