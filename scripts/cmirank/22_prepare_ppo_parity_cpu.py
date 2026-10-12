#!/usr/bin/env python3
"""Prepare twenty fixed numerical parity probes without GPU/model weights."""

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.ppo_compat import load_compat_config, validate_compat_rows
from src.cmirank.ppo_parity import completion_request, load_parity_config, parity_code_hashes, server_options
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config, path = load_parity_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / config["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 4
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or os.environ.get("PYTHONPATH") != str(private / "overlays" / compat["overlay_name"])):
        raise ValueError("Fresh exact-source private CPU parity preparation required")
    started = time.monotonic()
    for name, version in runtime["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned native runtime changed")
    role_dir = private / "runs" / config["parent_role_run_id"]
    old_dir = private / "runs" / config["parent_prepare_run_id"]
    if (file_sha256(role_dir / "report.json") != config["parent_role_report_sha256"]
            or file_sha256(old_dir / "report.json") != config["parent_prepare_report_sha256"]):
        raise ValueError("Reviewed parent compatibility receipts changed")
    role = json.loads((role_dir / "report.json").read_text())
    cleanup = json.loads((role_dir / "cleanup.json").read_text())
    handback = json.loads((role_dir / "keeper-handback.json").read_text())
    if (role["status"] != "FULL_4B_ROLE_COMPAT_PASS_CLEANUP_REVIEW_AND_PPO_WORKER_GATES_REMAIN"
            or any(role[k] != 20 for k in ("kernel_samples", "actor_reference_samples", "native_critic_samples"))
            or role["checkpoint_max_response_logprob_delta"] != 0 or not role["value_head_state_roundtrip"]
            or cleanup["process_exit_code"] or not cleanup["child_exited"] or not cleanup["gpu_released"]
            or not cleanup["allocation_still_running"] or handback["reserved_job_cancelled"] or handback["gpu1_signalled"]
            or handback["status"] != "KEEPER_GPU0_HANDBACK_VERIFIED"):
        raise ValueError("Unreviewed/unclean full-role gate cannot authorize parity preparation")
    old = json.loads((old_dir / "report.json").read_text())
    token_file = old_dir / "tokens.jsonl"
    if file_sha256(token_file) != old["artifact_sha256"][token_file.name]:
        raise ValueError("Frozen synthetic tokens changed")
    rows = [json.loads(line) for line in token_file.read_text().splitlines()]
    validate_compat_rows(rows, compat)
    checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
    if set(p.name for p in checkpoint.iterdir() if p.is_file()) != set(compat["checkpoint_files_sha256"]):
        raise ValueError("Only the reviewed fixed-final SFT initializer may be used")
    for name, sha in compat["checkpoint_files_sha256"].items():
        if file_sha256(checkpoint / name) != sha:
            raise ValueError("Reviewed initializer weights changed")
    # These are native server process options, not training/model changes.
    os.environ.update(config.get("server_environment", {}))
    import torch
    import vllm.envs as native_envs
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import ModelConfig
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.cli_args import FlexibleArgumentParser, make_arg_parser
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.registry import MultiModalRegistry
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead
    if torch.cuda.is_initialized():
        raise ValueError("CPU protocol inspection initialized CUDA")
    required = {"prompt_logprobs", "return_token_ids", "add_special_tokens", "prompt"}
    if not required.issubset(CompletionRequest.model_fields):
        raise ValueError("Native completion protocol does not support exact raw-token probes")
    required_engine_args = {"enable_prefix_caching", "enforce_eager", "tensor_parallel_size", "gpu_memory_utilization", "language_model_only"}
    if not required_engine_args.issubset(inspect.signature(EngineArgs).parameters):
        raise ValueError("Native serving options changed")
    cli = make_arg_parser(FlexibleArgumentParser())
    serving = cli.parse_args(["--model", str(checkpoint), *server_options(config)])
    if serving.tensor_parallel_size != 1 or serving.enable_prefix_caching or not serving.enforce_eager:
        raise ValueError("Native CLI altered the single-card/eager/no-prefix-cache gate")
    if config.get("gdn_prefill_backend"):
        if (serving.gdn_prefill_backend != "triton" or native_envs.VLLM_USE_DEEP_GEMM
                or native_envs.VLLM_DEEP_GEMM_WARMUP != "skip"
                or serving.kernel_config.enable_flashinfer_autotune):
            raise ValueError("Native kernel/backend options did not disable unavailable optional JIT paths")
        from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
        # Execute native skip routing with an explicit NO-MODEL CPU shim.
        # Unexpected get_model would fail; this is not a production worker test.
        kernel_warmup(SimpleNamespace(
            vllm_config=SimpleNamespace(kernel_config=serving.kernel_config),
            model_runner=SimpleNamespace(is_pooling_model=True, attn_groups=[])))
    native_text_only_supported = None
    if config.get("language_model_only"):
        if not serving.language_model_only:
            raise ValueError("Native CLI did not enable text-only mode")
        # Exercise the real branch used by InputProcessor BEFORE MultiModalBudget.
        # No weights/engine/CUDA; the accepted save deliberately lacks image-processor files.
        engine_args = EngineArgs.from_cli_args(serving)
        model_config = engine_args.create_model_config()
        native_text_only_supported = MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config)
        if (native_text_only_supported or not model_config.get_multimodal_config().language_model_only
                or model_config.get_multimodal_config().enable_mm_embeds):
            raise ValueError("Native text-only registry did not bypass multimodal processor budgeting")
    probes = []
    for row in rows:
        payload = completion_request(row, config)
        request = CompletionRequest.model_validate(payload)
        if (request.prompt != row["input_ids"] or request.prompt_logprobs != 0 or not request.return_token_ids
                or request.add_special_tokens or request.max_tokens != 1 or request.stream):
            raise ValueError("Native request protocol changed tokens or request budget")
        probes.append({"user_id": row["user_id"], "request": payload})
    if torch.cuda.is_initialized():
        raise ValueError("CPU schema probes initialized CUDA")
    run.mkdir()
    (run / "tokens.jsonl").write_bytes(token_file.read_bytes())
    (run / "requests.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in probes))
    report = {"status": "PPO_PARITY_CPU_PREPARATION_PASS_GPU_NUMERICS_REMAIN", "source_commit": commit,
        "config_sha256": file_sha256(path), "code_sha256": parity_code_hashes(ROOT), "slurm_job_id": job,
        "parent_role_report_sha256": config["parent_role_report_sha256"], "samples": len(rows),
        "max_total_input_tokens": max(len(r["input_ids"]) for r in rows), "physical_request_cap": 20,
        "native_schema_probes": len(probes), "native_cli_options_parsed": True,
        "native_text_only_registry_inputs_supported": native_text_only_supported,
        "native_gdn_prefill_backend": getattr(serving, "gdn_prefill_backend", None),
        "native_server_environment_checked": config.get("server_environment", {}),
        "native_kernel_warmup_skip_no_model_cpu_probe": bool(config.get("gdn_prefill_backend")),
        "native_fsdp_interfaces_imported": True,
        "native_fsdp_default_model_dtype": inspect.signature(FSDPEngineConfig).parameters["model_dtype"].default,
        "native_fsdp_default_training_precision_warning": "If model_dtype is None, unmodified native FSDP loads training parameters FP32; BF16 role smoke does not prove its optimizer memory fit. Freeze an explicit precision/offload profile before worker updates.",
        "native_source_sha256": {str(Path(inspect.getfile(c)).relative_to(Path(sys.prefix))):
            file_sha256(Path(inspect.getfile(c))) for c in (CompletionRequest, EngineArgs, ModelConfig, MultiModalRegistry,
                FSDPEngineConfig, FSDPEngineWithLMHead, FSDPEngineWithValueHead)},
        "checkpoint_stat": {n: {"size": (checkpoint / n).stat().st_size, "mtime_ns": (checkpoint / n).stat().st_mtime_ns,
                               "inode": (checkpoint / n).stat().st_ino} for n in compat["checkpoint_files_sha256"]},
        "overlay_sha256": old["overlay_sha256"], "gpu_requested": False, "optimizer_updates": 0,
        "full_weights_loaded": False, "books_outcomes_accessed": False, "training_ready": False,
        "artifact_sha256": {n: file_sha256(run / n) for n in ("tokens.jsonl", "requests.jsonl")},
        "elapsed_seconds": time.monotonic() - started}
    if config.get("gdn_prefill_backend"):
        native_root = Path(inspect.getfile(EngineArgs)).parents[1]
        for relative in ("model_executor/layers/mamba/gdn_linear_attn.py",
                         "model_executor/warmup/kernel_warmup.py", "utils/deep_gemm.py", "envs.py"):
            native_file = native_root / relative
            report["native_source_sha256"][str(native_file.relative_to(Path(sys.prefix)))] = file_sha256(native_file)
    (run / "report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "samples", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
