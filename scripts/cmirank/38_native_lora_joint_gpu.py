#!/usr/bin/env python3
"""Two co-resident4B native LoRA roles, one discarded update each on GPU1.

No Books labels, rollout/reference server, full checkpoint or PPO/GAE learning.
The unchanged native engines own all forward/backward/clipping/AdamW operations.
"""

import argparse
from datetime import timedelta
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.lora import audit_scope
from src.cmirank.lora_joint import (load_joint_config, verify_joint_preparation,
    native_model_config, native_engine_config, native_optimizer_config)
from src.cmirank.optimizer_smoke import (allocation_memory, load_optimizer_config, verified_parents,
    native_actor_batch, native_critic_batch)
from src.cmirank.corrected_rollout import load_corrected_config, native_microbatch
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_parity import compare_logprobs
from src.cmirank.gpu_resources import canonical_gpu_uuid
from src.cmirank.provenance import artifact_json_dumps


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("joint",), default="joint")
    parser.add_argument("--phase", choices=("optimizer", "checkpoint"), default="optimizer")
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_joint_config(ROOT)
    if args.phase == "checkpoint":
        from src.cmirank.lora_checkpoint import (load_checkpoint_config, verify_checkpoint_preparation,
            native_checkpoint_config, run_checkpoint_roundtrip)
        c, path = load_checkpoint_config(ROOT)
    native, _ = load_corrected_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private/"runs"/c["run_id"] or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Exact clean deployed joint LoRA source required")
    prepared = (verify_checkpoint_preparation(private, ROOT, c, path, commit) if args.phase == "checkpoint"
                else verify_joint_preparation(private, ROOT, c, path, commit))
    observations, _, _ = verified_parents(private, load_optimizer_config(ROOT)[0])
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned joint LoRA task interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    import torch
    import torch.distributed as dist
    from verl.workers.config.actor import ActorConfig
    from verl.workers.config.critic import CriticConfig
    from verl.trainer.config import CheckpointConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead
    from verl.workers.utils.losses import ppo_loss, value_loss
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
                != canonical_gpu_uuid(os.environ["MEMREC_SELECTED_GPU_UUID"])):
        raise ValueError("Both native roles must bind only the authorized physical GPU1")
    torch.set_num_threads(4)
    torch.manual_seed(c["seed"])
    began = time.monotonic()
    engines, scores, results = {}, {}, {}
    vectors = []
    def memory(label):
        torch.cuda.synchronize()
        row = {"event": label, "elapsed_seconds": time.monotonic()-began,
            "allocated_mib": torch.cuda.memory_allocated()/2**20,
            "reserved_mib": torch.cuda.memory_reserved()/2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
            "allocation_memory": allocation_memory(job)}
        with (run/"memory.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True)+"\n")
        print(json.dumps(row), flush=True)
        return row
    try:
        dist.init_process_group(backend="nccl", init_method="file://"+str(run/"native-rendezvous"),
            rank=0, world_size=1, timeout=timedelta(seconds=120), device_id=torch.device("cuda:0"))
        for role in ("actor", "critic"):
            model = native_model_config(private, c, compat, prepared, role)
            config = native_engine_config()
            optim = native_optimizer_config(c, role)
            cls = FSDPEngineWithLMHead if role == "actor" else FSDPEngineWithValueHead
            engine = cls(model, config, optim,
                native_checkpoint_config(c) if args.phase == "checkpoint" else CheckpointConfig())
            engine.initialize()
            engines[role] = engine
            observed = audit_scope(engine.module, prepared["targets"][role], c, role)
            if observed != prepared["roles"][role]:
                raise ValueError("Actual native LoRA scope/storage differs from reviewed official meta receipt")
            base = engine.module.get_base_model()
            if role == "critic" and not isinstance(base, hf.Qwen3_5ForTokenClassification):
                raise ValueError("Generic PEFT must preserve the unmodified native scalar critic")
            layer = base.model.language_model.layers[0].linear_attn
            if (not hf.is_fast_path_available or layer.chunk_gated_delta_rule is not hf.chunk_gated_delta_rule
                    or not layer.chunk_gated_delta_rule.__module__.startswith("fla.")
                    or layer.causal_conv1d_fn is not hf.causal_conv1d_fn
                    or type(engine.optimizer) is not torch.optim.AdamW
                    or engine.optimizer.defaults["foreach"] is not False or engine.optimizer.state):
                raise ValueError("Native audited kernel/empty AdamW initialization differs")
            memory(f"native_lora_{role}_initialized_fp32")
        memory("both_lora_roles_resident_before_scoring")
        def score_once(engine, role):
            collected = []
            with engine.eval_mode(), torch.no_grad():
                for obs in observations:
                    ids = torch.tensor(obs["input_ids"], device="cuda:0")
                    data = native_microbatch(ids, native, torch)
                    _, result = engine.forward_step(data, loss_function=None, forward_only=True)
                    key = "log_probs" if role == "actor" else "values"
                    values = result["model_output"][key].values()[obs["prompt_tokens"]-1:len(obs["input_ids"])-1]
                    if values.numel() != len(obs["response_token_ids"]) or not torch.isfinite(values).all():
                        raise ValueError("Native checkpoint response/EOS scoring invalid")
                    collected.append(values.float().cpu().tolist())
            return collected

        for role, engine in engines.items():
            collected = []
            with engine.eval_mode(), torch.no_grad():
                for obs in observations:
                    ids = torch.tensor(obs["input_ids"], device="cuda:0")
                    data = native_microbatch(ids, native, torch)
                    pair = []
                    for _ in range(2):
                        _, result = engine.forward_step(data, loss_function=None, forward_only=True)
                        key = "log_probs" if role == "actor" else "values"
                        values = result["model_output"][key].values()[obs["prompt_tokens"]-1:len(obs["input_ids"])-1]
                        if values.numel() != len(obs["response_token_ids"]) or not torch.isfinite(values).all():
                            raise ValueError("Actual native response/EOS readout invalid")
                        pair.append(values.float().cpu().tolist())
                        del result, values
                    repeat_pass = (compare_logprobs(*pair, native)["passed"] if role == "actor" else pair[0] == pair[1])
                    if not repeat_pass:
                        raise ValueError("Actual native pre-update scoring failed original role-repeat bounds")
                    collected.append(pair[0])
                    vectors.append({"role": role, "user_id": obs["user_id"], "values": pair[0],
                        "repeated_values": pair[1], "repeat_delta": max(abs(a-b) for a, b in zip(*pair))})
                    del ids, data
            scores[role] = collected
            memory(f"native_lora_{role}_scored_twice20")
        for role, engine in engines.items():
            if role == "actor":
                data, weights = native_actor_batch(observations, scores[role], c, torch, "cuda:0")
                for obs, weight in zip(observations, weights):
                    next(x for x in vectors if x["role"] == role and x["user_id"] == obs["user_id"])["sequence_is_weight"] = weight
            else:
                data = native_critic_batch(observations, scores[role], c, torch, "cuda:0")
                weights = []
            cls, loss_fn = (ActorConfig, ppo_loss) if role == "actor" else (CriticConfig, value_loss)
            loss_config = cls(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=20,
                ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False)
            # Samples, not a full4B bitwise checkpoint comparison. Frozen
            # parameters must also have no gradients/optimizer state at all.
            before = {name: local_tensor(p).detach().reshape(-1)[:32].cpu().clone()
                for name, p in engine.module.named_parameters()}
            with engine.train_mode():
                result = engine.train_batch(data, loss_function=partial(loss_fn, config=loss_config))
                norm = float(result["metrics"]["grad_norm"])
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("Native LoRA gradient norm zero/nonfinite")
                count = moments = gradients = changed_adapters = changed_head = 0
                nonzero_head_grad = False
                for name, p in engine.module.named_parameters():
                    state = engine.optimizer.state.get(p, {})
                    after = local_tensor(p).detach().reshape(-1)[:32].cpu()
                    if not p.requires_grad:
                        if p.grad is not None or state or not torch.equal(before[name], after):
                            raise ValueError("Frozen backbone/vision/head-original received update/state")
                        continue
                    grad = local_tensor(p.grad) if p.grad is not None else None
                    if (grad is None or grad.dtype != torch.float32 or not torch.isfinite(grad).all()
                            or not state or float(state["step"]) != 1 or grad.device.type != "cuda"):
                        raise ValueError("All adapters/head need finite FP32 CUDA gradients and first AdamW state")
                    count += p.numel()
                    gradients += grad.numel()*grad.element_size()
                    head = ".score.modules_to_save.default." in name
                    nonzero_head_grad |= head and bool(grad.abs().max() > 0)
                    changed = not torch.equal(before[name], after)
                    changed_head += head and changed
                    changed_adapters += not head and changed
                    for key in ("exp_avg", "exp_avg_sq"):
                        value = local_tensor(state[key])
                        if (value.dtype != torch.float32 or value.numel() != p.numel()
                                or not torch.isfinite(value).all() or value.device.type != "cuda"):
                            raise ValueError("All adapter/head AdamW moments must remain finite FP32 on GPU1")
                        moments += value.numel()*value.element_size()
                scope = prepared["roles"][role]
                if (count != scope["trainable_parameters"] or gradients != scope["gradient_bytes"]
                        or moments != scope["adam_moment_bytes"] or changed_adapters == 0
                        or role == "critic" and (not nonzero_head_grad or changed_head == 0)):
                    raise ValueError("Incomplete native LoRA or scalar-head optimizer update")
            results[role] = {"trainable_parameters": count, "gradient_bytes": gradients,
                "adam_moment_bytes": moments, "changed_adapter_parameter_samples": changed_adapters,
                "changed_head_parameter_samples": changed_head, "head_nonzero_gradient": nonzero_head_grad,
                "native_gradient_norm_before_clip": norm, "is_weight_min": min(weights) if weights else None,
                "is_weight_max": max(weights) if weights else None, "optimizer_updates": 1,
                "frozen_parameters_have_no_grad_or_optimizer_state": True,
                "frozen_parameter_samples_unchanged": True, "full_frozen_checkpoint_equality_proven": False}
            memory(f"native_lora_{role}_updated_both_roles_still_resident")
            if args.phase == "checkpoint":
                results[role]["checkpoint"] = run_checkpoint_roundtrip(engine, data,
                    partial(loss_fn, config=loss_config), c, role, run, memory,
                    lambda active_engine: score_once(active_engine, role))
                results[role]["optimizer_updates"] = c["updates_per_role"]
        final = memory("both_lora_roles_and_adam_states_resident_complete")
        report = {"status": c["worker_status"], "role": "joint", "source_commit": commit,
            "parameterization": "lora", "samples": 20, "optimizer_updates": c["optimizer_updates"],
            "updates_per_role": c["updates_per_role"],
            "roles": results, "model_dtype": "fp32", "forward_dtype": "bfloat16", "rank": c["rank"],
            "native_engine_strategy": "fsdp2", "optimizer": "torch.optim.AdamW",
            "actor_critic_joint_fit_proven": True, "rollout_loaded": False, "reference_loaded": False,
            "full_ppo_fit_proven": False, "full_checkpoint_roundtrip_proven": False, "multiturn_gae_proven": False,
            "physical_requests": 0, "books_outcomes_accessed": False, "ranking_metrics_computed": False,
            "update_discarded": True, "training_ready": False, "elapsed_seconds": time.monotonic()-began,
            "peak_vram_allocated_mib": final["peak_allocated_mib"], "peak_vram_reserved_mib": final["peak_reserved_mib"],
            "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if args.phase == "checkpoint":
            report.update(native_checkpoint_roundtrip_proven=True,
                restored_logical_updates_per_role=c["restored_logical_updates_per_role"],
                checkpoint_scope="full_native_role_model_optimizer_scheduler_rng_no_hf_merge", research_checkpoint=False)
        engines.clear()
        engine = model = base = layer = data = result = before = p = state = after = grad = value = None
        gc.collect()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        (run/"native-vectors.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in vectors))
        (run/"native-worker-report.json").write_text(artifact_json_dumps(report))
    finally:
        engines.clear()
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
