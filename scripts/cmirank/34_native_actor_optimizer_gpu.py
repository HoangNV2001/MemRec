#!/usr/bin/env python3
"""One discarded native FSDP2/AdamW update on20 frozen synthetic responses.

Not Books training, PPO+GAE training, co-residency or checkpoint qualification.
No optimizer/model/loss implementation is replaced or monkeypatched here.
"""

import argparse
from datetime import timedelta
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.optimizer_smoke import (allocation_memory, load_optimizer_profile, memory_profile,
    native_actor_batch, native_critic_batch, TEXT_PARAMETERS, VISION_PARAMETERS, verified_parents, verify_optimizer_preparation)
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
    parser.add_argument("--role", choices=("actor", "critic"), default="actor")
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_optimizer_profile(ROOT, args.role)
    native, _ = load_corrected_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / c["run_id"] or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Exact clean deployed native actor source required")
    prepared = verify_optimizer_preparation(private, ROOT, c, path, commit)
    observations, _, _ = verified_parents(private, c)
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned actor interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    import torch
    import torch.distributed as dist
    from verl.workers.config.model import HFModelConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig
    from verl.workers.config.actor import ActorConfig
    from verl.workers.config.critic import CriticConfig
    from verl.trainer.config import CheckpointConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead
    from verl.workers.utils.losses import ppo_loss, value_loss
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
                != canonical_gpu_uuid(os.environ["MEMREC_SELECTED_GPU_UUID"])):
        raise ValueError("Native actor must bind only the authorized physical GPU1")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    began = time.monotonic()
    engine = None
    def memory(label):
        torch.cuda.synchronize()
        record = {"event": label, "elapsed_seconds": time.monotonic()-began,
            "allocated_mib": torch.cuda.memory_allocated()/2**20,
            "reserved_mib": torch.cuda.memory_reserved()/2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
            "allocation_memory": allocation_memory(job)}
        with (run / "memory.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True)+"\n")
        print(json.dumps(record), flush=True)
        return record
    try:
        dist.init_process_group(backend="nccl", init_method="file://" + str(run / "native-rendezvous"),
            rank=0, world_size=1, timeout=timedelta(seconds=120), device_id=torch.device("cuda:0"))
        compat, _ = load_compat_config(ROOT)
        checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
        model = HFModelConfig(path=str(checkpoint), load_tokenizer=False, use_remove_padding=False,
            use_fused_kernels=False, enable_gradient_checkpointing=True,
            model_type="language_model" if args.role == "actor" else "value_model")
        config = FSDPEngineConfig(strategy="fsdp2", model_dtype="fp32", dtype="bfloat16", forward_only=False,
            use_torch_compile=False, use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
            use_remove_padding=False, param_offload=False, optimizer_offload=False, offload_policy=False)
        optim = FSDPOptimizerConfig(lr=c["optimizer_lr"], total_training_steps=1,
            override_optimizer_config={"foreach": False})
        engine_cls = FSDPEngineWithLMHead if args.role == "actor" else FSDPEngineWithValueHead
        engine = engine_cls(model, config, optim, CheckpointConfig())
        engine.initialize()
        # Ordinary torch freeze before first forward/FSDP2 lazy initialization.
        # Native ActorConfig.freeze_vision_tower has no consumer in this pinned
        # backend; never pretend merely setting that flag freezes this branch.
        scope = {"text": 0, "vision": 0}
        for name, p in engine.module.named_parameters():
            visual = ".visual." in name
            p.requires_grad_(not visual)
            scope["vision" if visual else "text"] += p.numel()
            if p.dtype != torch.float32:
                raise ValueError("Native training parameters are not FP32 master storage")
        if (scope["text"], scope["vision"]) != (c["trainable_parameters"], VISION_PARAMETERS):
            raise ValueError("Full-text/frozen-vision parameter boundary changed")
        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
        if args.role == "critic":
            from src.cmirank.native_critic import parameter_shapes
            if (not isinstance(engine.module, hf.Qwen3_5ForTokenClassification)
                    or parameter_shapes(engine.module) != prepared["native_critic_contract"]["parameter_shapes"]):
                raise ValueError("Native critic must preserve the qualified backbone and scalar score head; no fallback")
        layer = engine.module.model.language_model.layers[0].linear_attn
        if (not hf.is_fast_path_available or layer.chunk_gated_delta_rule is not hf.chunk_gated_delta_rule
                or not layer.chunk_gated_delta_rule.__module__.startswith("fla.")
                or layer.causal_conv1d_fn is not hf.causal_conv1d_fn
                or type(engine.optimizer) is not torch.optim.AdamW
                or engine.optimizer.defaults["foreach"] is not False or engine.optimizer.state):
            raise ValueError("Native audited kernel/empty AdamW initialization differs")
        memory(f"native_{args.role}_initialized_fp32")
        scores, repeats = [], []
        with engine.eval_mode(), torch.no_grad():
            for obs in observations:
                ids = torch.tensor(obs["input_ids"], device="cuda:0")
                data = native_microbatch(ids, native, torch)
                passes = []
                for _ in range(2):
                    _, result = engine.forward_step(data, loss_function=None, forward_only=True)
                    key = "log_probs" if args.role == "actor" else "values"
                    lp = result["model_output"][key].values()[obs["prompt_tokens"]-1:len(obs["input_ids"])-1]
                    if lp.numel() != len(obs["response_token_ids"]) or not torch.isfinite(lp).all():
                        raise ValueError("Native FP32 actor response/EOS likelihood alignment failed")
                    passes.append(lp.float().cpu().tolist())
                    del result, lp
                repeat = (compare_logprobs(*passes, native) if args.role == "actor" else {
                    "passed": passes[0] == passes[1],
                    "max_absolute_value_delta": max(abs(a-b) for a, b in zip(*passes))})
                if not repeat["passed"]:
                    raise ValueError("Actual training-engine pre-update repeat failed original bounds")
                scores.append(passes[0])
                repeats.append({"user_id": obs["user_id"],
                    "old_logprobs" if args.role == "actor" else "old_values": passes[0], "native_repeat": repeat})
                del data, ids
        memory("native_old_policy_scored_twice20")
        if args.role == "actor":
            data, weights = native_actor_batch(observations, scores, c, torch, "cuda:0")
        else:
            data = native_critic_batch(observations, scores, c, torch, "cuda:0")
            weights = []
        loss_config_cls = ActorConfig if args.role == "actor" else CriticConfig
        actor = loss_config_cls(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=20,
            ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False)
        # Small parameter samples document an actual update, not full checkpoint
        # equality. No tensor snapshot/optimizer offload onto the64GiB host.
        before = {name: local_tensor(p).detach().reshape(-1)[:32].cpu().clone()
            for name, p in engine.module.named_parameters()}
        with engine.train_mode():
            loss_fn = ppo_loss if args.role == "actor" else value_loss
            result = engine.train_batch(data, loss_function=partial(loss_fn, config=actor))
            norm = float(result["metrics"]["grad_norm"])
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Native gradient norm was zero/nonfinite; no valid AdamW update")
            memory("native_one_update_complete_states_and_grads_resident")
            state_params = moment_bytes = grad_bytes = changed = changed_backbone = head_parameters = 0
            head_nonzero_gradient = False
            for name, p in engine.module.named_parameters():
                state = engine.optimizer.state.get(p, {})
                grad = p.grad
                after = local_tensor(p).detach().reshape(-1)[:32].cpu()
                if ".visual." in name:
                    if p.requires_grad or grad is not None or state or not torch.equal(before[name], after):
                        raise ValueError("Frozen visual branch received gradients/state/update")
                    continue
                if grad is None or not state or float(state["step"]) != 1:
                    raise ValueError("A full-text parameter lacks first-update native optimizer state")
                if grad.dtype != torch.float32 or not torch.isfinite(local_tensor(grad)).all():
                    raise ValueError("Native master gradients must be FP32 and finite")
                grad_bytes += local_tensor(grad).numel() * local_tensor(grad).element_size()
                if name.startswith("score."):
                    head_parameters += p.numel()
                    head_nonzero_gradient |= bool(local_tensor(grad).abs().max() > 0)
                state_params += p.numel()
                for key in ("exp_avg", "exp_avg_sq"):
                    value = local_tensor(state[key])
                    if value.dtype != torch.float32 or value.numel() != p.numel() or not torch.isfinite(value).all():
                        raise ValueError("Native full-text AdamW moments must be complete, FP32 and finite")
                    moment_bytes += value.numel() * value.element_size()
                changed += not torch.equal(before[name], after)
                changed_backbone += not name.startswith("score.") and not torch.equal(before[name], after)
            if (state_params != c["trainable_parameters"] or moment_bytes != 8*c["trainable_parameters"]
                    or grad_bytes != 4*c["trainable_parameters"] or changed_backbone == 0):
                raise ValueError("Full text optimizer/gradient scope or actual parameter update unproven")
            if args.role == "critic" and (head_parameters != c["scalar_head_parameters"] or not head_nonzero_gradient):
                raise ValueError("Native scalar score head lacks complete optimizer state/nonzero gradients")
        final_memory = memory("native_state_audit_complete")
        report = {"status": c["worker_status"], "role": args.role,
            "source_commit": commit, "samples": 20, "native_engine_strategy": "fsdp2", "model_dtype": "fp32",
            "forward_dtype": "bfloat16", "optimizer_updates": 1, "optimizer": "torch.optim.AdamW",
            "optimizer_state_parameters": state_params, "adam_moment_bytes": moment_bytes,
            "gradient_bytes_during_update": grad_bytes, "native_gradient_norm_before_clip": norm,
            "changed_parameter_samples": changed, "changed_backbone_parameter_samples": changed_backbone,
            "scalar_head_parameters": head_parameters, "scalar_head_nonzero_gradient": head_nonzero_gradient,
            "memory_profile": memory_profile(args.role),
            "peak_vram_allocated_mib": final_memory["peak_allocated_mib"],
            "peak_vram_reserved_mib": final_memory["peak_reserved_mib"],
            "is_weight_min": min(weights) if weights else None, "is_weight_max": max(weights) if weights else None,
            "same_engine_repeat_samples": 20, "physical_requests": 0, "update_discarded": True,
            "training_ready": False, "books_outcomes_accessed": False, "ranking_metrics_computed": False,
            "co_residency_proven": False, "critic_optimizer_proven": args.role == "critic",
            "full_checkpoint_roundtrip_proven": False, "multiturn_gae_proven": False,
            "old_strict_cross_engine_gate_status": "FAIL_UNCHANGED", "elapsed_seconds": time.monotonic()-began}
        # Model exits before CPU report serialization/review; no trained checkpoint.
        engine = data = result = before = None
        p = state = grad = value = after = layer = None
        gc.collect()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        (run / ("native-old-logprobs.jsonl" if args.role == "actor" else "native-old-values.jsonl")).write_text(
            "".join(json.dumps(x, sort_keys=True)+"\n" for x in repeats))
        (run / "native-worker-report.json").write_text(artifact_json_dumps(report))
    finally:
        engine = None
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
