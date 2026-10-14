#!/usr/bin/env python3
"""CPU20 native PPO batch/loss schemas and one-card memory budget, no weights."""

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
from src.cmirank.optimizer_smoke import (allocation_memory, load_optimizer_config,
    memory_profile, native_actor_batch, optimizer_code_hashes, verified_parents)
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.corrected_rollout import load_corrected_config
from src.cmirank.ppo_parity import verify_fla_overlay
from src.cmirank.cuda_toolchain import verify_cuda_toolchain
from src.cmirank.causal_conv import verify_causal_overlay
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_optimizer_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    native, _ = load_corrected_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private / "runs" / c["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]):
        raise ValueError("Fresh exact-source allocation-only CPU preparation required")
    began = time.monotonic()
    for name, version in runtime["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned runtime changed")
    observations, previous, parent = verified_parents(private, c)
    verify_fla_overlay(private, native)
    verify_cuda_toolchain(private, ROOT, native)
    verify_causal_overlay(private, ROOT, native)
    checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
    for n, sha in compat["checkpoint_files_sha256"].items():
        if file_sha256(checkpoint / n) != sha:
            raise ValueError("Initializer changed")
    for n, sha in parent["native_source_sha256"].items():
        if file_sha256(Path(sys.prefix) / n) != sha:
            raise ValueError("Reviewed native source changed")
    for n, sha in parent["overlay_sha256"].items():
        if file_sha256(private / "overlays" / compat["overlay_name"] / n) != sha:
            raise ValueError("Reviewed FA2 overlay changed")
    profile = memory_profile()
    host = allocation_memory(job)
    if host["maximum_bytes"] - host["current_bytes"] < c["host_load_reserve_bytes"]:
        raise ValueError("Own64GiB allocation lacks the fixed20GiB host-load margin; keep keeper active")
    inventory = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid,memory.total",
        "--format=csv,noheader,nounits"], text=True).strip()
    gpu_uuid, total = [x.strip() for x in inventory.split(",")]
    if profile["persistent_train_bytes"] + c["workspace_reserve_bytes"] > int(total) * 2**20:
        raise ValueError("Actor-only FP32+AdamW estimate exceeds physical card budget")
    import torch
    from types import SimpleNamespace
    from verl.workers.config.actor import ActorConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.model import HFModelConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, prepare_micro_batches
    from verl.workers.engine.base import BaseEngine
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding
    from verl.utils import tensordict_utils as tu
    torch.set_num_threads(4)
    actor = ActorConfig(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=20,
        ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False)
    model = HFModelConfig(path=str(checkpoint), load_tokenizer=False, use_remove_padding=False,
        use_fused_kernels=False, enable_gradient_checkpointing=True)
    engine = FSDPEngineConfig(strategy="fsdp2", model_dtype="fp32", dtype="bfloat16", forward_only=False,
        use_torch_compile=False, use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
        use_remove_padding=False, param_offload=False, optimizer_offload=False, offload_policy=False)
    optim = FSDPOptimizerConfig(lr=c["actor_lr"], total_training_steps=1,
        override_optimizer_config={"foreach": False})
    tiny_optimizer = build_optimizer([torch.nn.Parameter(torch.zeros(3))], optim)
    if (type(tiny_optimizer) is not torch.optim.AdamW or tiny_optimizer.defaults["foreach"] is not False
            or model.lora_rank or engine.forward_only or engine.model_dtype != "fp32"):
        raise ValueError("Native training/storage/optimizer schema changed")
    data, weights = native_actor_batch(observations, [x["old_logprobs"] for x in previous], c, torch)
    tokens = sum(len(x["response_token_ids"]) for x in observations)
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=tokens)
    batches, _ = prepare_micro_batches(data, dp_group=None, same_micro_num_in_dp=False)
    if len(batches) != 20:
        raise ValueError("Native fixed microbatch packing changed")
    probes = []
    for obs, micro in zip(observations, batches):
        proxy = SimpleNamespace(use_ulysses_sp=False, pad_to_length=False)
        packed, _ = FSDPEngineWithLMHead.prepare_model_inputs(proxy, micro)
        if packed["input_ids"][0].tolist() != obs["input_ids"]:
            raise ValueError("Native model input packing changed token order")
        # CPU full-sequence LP proxy only; actual CE/backward/AdamW is mandatory GPU34.
        p, n = obs["prompt_tokens"], len(obs["response_token_ids"])
        current = torch.cat((torch.zeros(p-1), micro["old_log_probs"].values().detach(), torch.zeros(1))).requires_grad_()
        output = {"log_probs": torch.nested.as_nested_tensor([current], layout=torch.jagged)}
        extracted = no_padding_2_padding(output["log_probs"], micro)
        if not torch.equal(extracted[0], micro["old_log_probs"].values()):
            raise ValueError("Response/EOS next-token shift differs")
        loss, _ = ppo_loss(actor, output, micro)
        loss.backward()
        if not torch.isfinite(loss) or current.grad is None or not torch.isfinite(current.grad).all():
            raise ValueError("Native PPO loss proxy did not propagate finite LP gradients")
        if current.grad[:p-1].count_nonzero() or current.grad[-1] != 0:
            raise ValueError("Prompt or wraparound token received policy-loss gradients")
        probes.append({"user_id": obs["user_id"], "response_tokens_including_eos": n,
            "native_loss": float(loss), "native_response_alignment_pass": True, "prompt_gradient_zero": True})
    if torch.cuda.is_initialized():
        raise ValueError("CPU rehearsal initialized CUDA")
    classes = (ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, build_optimizer,
        FSDPEngineWithLMHead, BaseEngine, ppo_loss, no_padding_2_padding, prepare_micro_batches,
        FSDPParamGroup, torch.optim.AdamW)
    sources = {**parent["native_source_sha256"], **{str(Path(inspect.getfile(x)).relative_to(Path(sys.prefix))):
        file_sha256(Path(inspect.getfile(x))) for x in classes}}
    run.mkdir()
    (run / "probes.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in probes))
    report = {"status": "NATIVE_ACTOR_OPTIMIZER_CPU_PREPARATION_PASS_GPU_GATE_REMAINS",
        "source_commit": commit, "config_sha256": file_sha256(path), "code_sha256": optimizer_code_hashes(ROOT),
        "samples": 20, "gpu_requested": False, "optimizer_updates": 0, "full_weights_loaded": False,
        "training_ready": False, "native_microbatches": len(batches), "response_tokens": tokens,
        "memory_profile": profile, "allocation_memory": host, "physical_gpu_uuid": gpu_uuid,
        "physical_gpu_total_mib": int(total), "native_source_sha256": sources,
        "overlay_sha256": parent["overlay_sha256"], "checkpoint_stat": {n: {
            "size": (checkpoint/n).stat().st_size, "mtime_ns": (checkpoint/n).stat().st_mtime_ns,
            "inode": (checkpoint/n).stat().st_ino} for n in compat["checkpoint_files_sha256"]},
        "proxy_weights_min": min(weights), "proxy_weights_max": max(weights),
        "artifact_sha256": {"probes.jsonl": file_sha256(run / "probes.jsonl")},
        "elapsed_seconds": time.monotonic()-began}
    (run / "report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "samples", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
