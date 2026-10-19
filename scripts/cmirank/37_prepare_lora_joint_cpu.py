#!/usr/bin/env python3
"""Fresh source-bound CPU20 packing/budget gate before native joint LoRA GPU."""

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.lora_joint import (load_joint_config, reviewed_lora_parent, joint_code_hashes, joint_gpu_budget,
    native_model_config, native_engine_config, native_optimizer_config)
from src.cmirank.optimizer_smoke import load_optimizer_config, verified_parents, native_actor_batch, native_critic_batch
from src.cmirank.host_memory import cpu_memory_budget
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("optimizer", "checkpoint"), default="optimizer")
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    if args.phase == "checkpoint":
        from src.cmirank.lora_checkpoint import (load_checkpoint_config, checkpoint_code_hashes,
            reviewed_joint_parent, checkpoint_budget, native_checkpoint_config)
        c, path = load_checkpoint_config(ROOT)
    else:
        c, path = load_joint_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / c["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Fresh clean-source CPU joint preparation required")
    began = time.monotonic()
    for name, version in {**runtime["versions"], "peft": "0.21.2"}.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned native runtime changed")
    parent = reviewed_lora_parent(private, ROOT, c)
    joint_parent = reviewed_joint_parent(private, ROOT, c) if args.phase == "checkpoint" else None
    observations, old, source = verified_parents(private, load_optimizer_config(ROOT)[0])
    budget = cpu_memory_budget(job, c["host_load_reserve_bytes"])
    if not budget["assessment"]["within_estimated_budget"]:
        raise ValueError("Original host margin estimate unavailable; no handoff")
    checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
    for n, sha in compat["checkpoint_files_sha256"].items():
        if file_sha256(checkpoint/n) != sha:
            raise ValueError("Frozen initializer changed")
    import torch
    from verl.workers.config.actor import ActorConfig
    from verl.workers.config.critic import CriticConfig
    from verl.workers.config.model import HFModelConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead, prepare_micro_batches
    from verl.workers.utils.losses import ppo_loss, value_loss
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
    from verl.utils.fsdp_utils import get_fsdp_state_ctx
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding
    torch.set_num_threads(4)
    probes = []
    for role in ("actor", "critic"):
        schema = native_model_config(private, c, compat, parent, role)
        engine_schema = native_engine_config()
        tiny_optimizer = build_optimizer([torch.nn.Parameter(torch.zeros(3))], native_optimizer_config(c, role))
        if (schema.lora_rank != c["rank"] or schema.model_type != ("language_model" if role == "actor" else "value_model")
                or engine_schema.model_dtype != "fp32" or engine_schema.param_offload
                or type(tiny_optimizer) is not torch.optim.AdamW or tiny_optimizer.defaults["foreach"] is not False):
            raise ValueError("Actual native model/engine/optimizer LoRA schemas differ")
        if role == "actor":
            data, _ = native_actor_batch(observations, [x["old_logprobs"] for x in old], c, torch)
        else:
            data = native_critic_batch(observations,
                [[0.01*(i+j+1) for j in range(len(x["response_token_ids"]))] for i, x in enumerate(observations)], c, torch)
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=sum(len(x["response_token_ids"]) for x in observations))
        batches, _ = prepare_micro_batches(data, dp_group=None, same_micro_num_in_dp=False)
        if len(batches) != 20:
            raise ValueError("Twenty native microbatches required per role")
        cls, fn = (ActorConfig, ppo_loss) if role == "actor" else (CriticConfig, value_loss)
        loss_config = cls(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=20,
            ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False)
        for obs, micro in zip(observations, batches):
            proxy = SimpleNamespace(use_ulysses_sp=False, pad_to_length=False, module=SimpleNamespace())
            packed, _ = FSDPEngineWithLMHead.prepare_model_inputs(proxy, micro)
            if packed["input_ids"][0].tolist() != obs["input_ids"]:
                raise ValueError("Native packing changed frozen prefix/response tokens")
            field = "old_log_probs" if role == "actor" else "values"
            p = obs["prompt_tokens"]
            current = torch.cat((torch.zeros(p-1), micro[field].values().detach(), torch.zeros(1))).requires_grad_()
            if role == "actor":
                output = {"log_probs": torch.nested.as_nested_tensor([current], layout=torch.jagged)}
            else:
                output = FSDPEngineWithValueHead.prepare_model_outputs(proxy,
                    SimpleNamespace(logits=current[None, :, None]), {}, micro, None)
            key = "log_probs" if role == "actor" else "values"
            if not torch.equal(no_padding_2_padding(output[key], micro)[0], micro[field].values()):
                raise ValueError("Native next-token/EOS response alignment changed")
            loss, _ = fn(loss_config, output, micro)
            loss.backward()
            if (not torch.isfinite(loss) or current.grad is None or not torch.isfinite(current.grad).all()
                    or current.grad[:p-1].count_nonzero() or current.grad[-1] != 0):
                raise ValueError("Native loss proxy/response-only gradient invalid")
            probes.append({"role": role, "user_id": obs["user_id"], "native_loss": float(loss.detach()),
                "response_tokens_including_eos": len(obs["response_token_ids"]), "prompt_gradient_zero": True})
    if torch.cuda.is_initialized():
        raise ValueError("CPU preparation initialized CUDA")
    total = int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.total",
        "--format=csv,noheader,nounits"], text=True).strip())*2**20
    gpu_budget = joint_gpu_budget(parent["roles"], total, c["workspace_reserve_bytes"])
    if not gpu_budget["within_estimated_budget"]:
        raise ValueError("Both LoRA roles plus fixed workspace estimate exceed actual GPU1")
    checkpoint_assessment = None
    if args.phase == "checkpoint":
        schema = native_checkpoint_config(c)
        if (schema.save_lora_only or schema.async_save or not schema.strict
                or schema.save_contents != ["model", "optimizer", "extra"]
                or schema.load_contents != schema.save_contents):
            raise ValueError("Native complete checkpoint must retain the scalar head and optimizer/RNG")
        checkpoint_assessment = checkpoint_budget(parent["roles"], total,
            budget["allocation_memory"]["maximum_bytes"], budget["stats_bytes"]["anon"],
            joint_parent["peak_worker_rss_kib"]*1024, shutil.disk_usage(private/"runs").free, c)
        if not checkpoint_assessment["within_estimated_budget"]:
            raise ValueError("Complete native checkpoint staging exceeds unchanged card/host/disk budget")
    classes = (FSDPEngineWithLMHead, FSDPEngineWithValueHead, ppo_loss, value_loss, ActorConfig, CriticConfig,
        HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig, build_optimizer)
    if args.phase == "checkpoint":
        classes += (FSDPCheckpointManager, BaseCheckpointManager, get_fsdp_state_ctx, type(schema))
    native_sources = {**source["native_source_sha256"], **parent["native_source_sha256"],
        **{str(Path(inspect.getfile(x)).relative_to(Path(sys.prefix))): file_sha256(Path(inspect.getfile(x))) for x in classes}}
    for n, sha in native_sources.items():
        if file_sha256(Path(sys.prefix)/n) != sha:
            raise ValueError("Reviewed native source changed")
    run.mkdir()
    (run / "probes.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in probes))
    result = {"status": ("LORA_CHECKPOINT_CPU20_PASS_GPU_GATE_REMAINS" if args.phase == "checkpoint"
                        else "LORA_JOINT_CPU20_PASS_GPU_GATE_REMAINS"), "source_commit": commit,
        "config_sha256": file_sha256(path), "code_sha256": (checkpoint_code_hashes(ROOT)
            if args.phase == "checkpoint" else joint_code_hashes(ROOT)),
        "samples_per_role": 20, "gpu_requested": False, "full_weights_loaded": False,
        "optimizer_updates": 0, "training_ready": False, "physical_requests": 0,
        "roles": parent["roles"], "targets": parent["targets"], "gpu_budget": gpu_budget,
        "host_budget": cpu_memory_budget(job, c["host_load_reserve_bytes"]),
        "native_source_sha256": native_sources, "overlay_sha256": source["overlay_sha256"],
        "checkpoint_stat": {n: {"size": (checkpoint/n).stat().st_size, "mtime_ns": (checkpoint/n).stat().st_mtime_ns,
            "inode": (checkpoint/n).stat().st_ino} for n in compat["checkpoint_files_sha256"]},
        "artifact_sha256": {"probes.jsonl": file_sha256(run/"probes.jsonl")}, "elapsed_seconds": time.monotonic()-began}
    if checkpoint_assessment:
        result["checkpoint_budget"] = checkpoint_assessment
        result["joint_parent_peak_rss_bytes"] = joint_parent["peak_worker_rss_kib"]*1024
    (run/"report.json").write_text(artifact_json_dumps(result))
    print(json.dumps({k: result[k] for k in ("status", "samples_per_role", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
