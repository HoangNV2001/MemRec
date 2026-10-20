#!/usr/bin/env python3
"""Native MPSS/GAE PPO on20 complete synthetic trajectories; all updates discarded.

The vLLM sampler must already have exited/released. This worker alone loads
actor+critic on the designated GPU1; reference is native disable_adapter().
"""

import argparse
from contextlib import nullcontext
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
from src.cmirank.multiturn_ppo import (load_multiturn_config, verify_multiturn_preparation,
    validate_episodes, episode_gae, native_training_batch)
from src.cmirank.lora import audit_scope
from src.cmirank.lora_joint import native_model_config, native_engine_config, native_optimizer_config
from src.cmirank.lora_checkpoint import native_checkpoint_config, train_state_fingerprint
from src.cmirank.lora_rollout import checkpoint_directory
from src.cmirank.corrected_rollout import native_microbatch
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.gpu_resources import canonical_gpu_uuid
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    require_allocation()
    c, path = load_multiturn_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private/"runs"/c["run_id"] or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Exact clean-source native multi-turn worker required")
    rows, prepared = verify_multiturn_preparation(private, ROOT, c, path, commit)
    manifest = json.loads((run/"manifest.json").read_text())
    if (file_sha256(run/"generated.jsonl") != manifest["generated_sha256"]
            or file_sha256(run/"episodes.jsonl") != manifest["episodes_sha256"]):
        raise ValueError("Physical sampled trajectory artifacts changed")
    observations = list(map(json.loads, (run/"generated.jsonl").read_text().splitlines()))
    episodes = list(map(json.loads, (run/"episodes.jsonl").read_text().splitlines()))
    validate_episodes(observations, episodes, rows)
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned native PPO worker interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    import torch
    import torch.distributed as dist
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead
    from verl.workers.config.actor import ActorConfig
    from verl.workers.config.critic import CriticConfig
    from verl.workers.utils.losses import ppo_loss, value_loss
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
                != canonical_gpu_uuid(os.environ["MEMREC_SELECTED_GPU_UUID"])):
        raise ValueError("Native multi-turn PPO may bind only physical GPU1")
    torch.set_num_threads(4)
    torch.manual_seed(c["seed"])
    began = time.monotonic()
    engines, before, scores, results = {}, {}, {}, {}
    def append(name, row):
        with (run/f"{name}.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
    def readouts(engine, role, reference=False):
        values = []
        with engine.eval_mode(), torch.no_grad(), (engine.disable_adapter() if reference else nullcontext()):
            for obs in observations:
                data = native_microbatch(torch.tensor(obs["input_ids"], device="cuda:0"), c, torch)
                _, output = engine.forward_step(data, loss_function=None, forward_only=True)
                key = "log_probs" if role == "actor" else "values"
                v = output["model_output"][key].values()[obs["prompt_tokens"]-1:len(obs["input_ids"])-1]
                if len(v) != len(obs["response_token_ids"]) or not torch.isfinite(v).all():
                    raise ValueError("Finite response/EOS-aligned native scores required")
                values.append(v.float().cpu().tolist())
        return values
    try:
        dist.init_process_group("nccl", init_method="file://"+str(run/"native-rendezvous"),
            rank=0, world_size=1, timeout=timedelta(seconds=120), device_id=torch.device("cuda:0"))
        compat, _ = load_compat_config(ROOT)
        for role, cls in (("actor", FSDPEngineWithLMHead), ("critic", FSDPEngineWithValueHead)):
            engine = cls(native_model_config(private, c, compat, prepared, role), native_engine_config(),
                native_optimizer_config(c, role), native_checkpoint_config(c))
            engine.initialize()
            engines[role] = engine
            engine.load_checkpoint(str(checkpoint_directory(private, c)/role), del_local_after_load=False)
            if audit_scope(engine.module, prepared["targets"][role], c, role) != prepared["roles"][role]:
                raise ValueError("Audited LoRA actor/critic scope/storage changed")
            base = engine.module.get_base_model()
            layer = base.model.language_model.layers[0].linear_attn
            if not hf.is_fast_path_available or not layer.chunk_gated_delta_rule.__module__.startswith("fla."):
                raise ValueError("Native fast-path changed")
            before[role] = train_state_fingerprint(engine)
            audit = json.loads((private/"runs"/c["checkpoint_parent_run_id"]/f"checkpoint-{role}-audit.json").read_text())
            if before[role] != audit["train_state_before"]:
                raise ValueError("Cold restore differs from known discarded T0 checkpoint")
            scores[role] = readouts(engine, role)
            print(json.dumps({"event": "native_scored", "role": role, "actions": 180,
                "elapsed_seconds": time.monotonic()-began}), flush=True)
        reference = readouts(engines["actor"], "actor", reference=True)
        if train_state_fingerprint(engines["actor"]) != before["actor"]:
            raise ValueError("Reference context changed actor/Adam state")
        scored = [{"user_id": obs["user_id"], "turn": obs["turn"], "old_logprobs": a,
            "reference_logprobs": ref, "critic_values": value}
            for obs, a, ref, value in zip(observations, scores["actor"], reference, scores["critic"])]
        for score in scored:
            append("native-logprobs", score)
        advantages, returns, audits = episode_gae(observations, episodes, scores["critic"], c, torch, "cuda:0")
        for audit in audits:
            append("gae", audit)
        # Fixed180 action rows =>5 optimizer updates/role. The9-turn return
        # continuity has already been computed before splitting mini-batches.
        for role in ("critic", "actor"):
            engine = engines[role]
            cls, loss_fn = (ActorConfig, ppo_loss) if role == "actor" else (CriticConfig, value_loss)
            extra = {"use_kl_loss": True, "kl_loss_coef": c["kl_coef"], "kl_loss_type": c["kl_loss_type"]} if role == "actor" else {}
            config = cls(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=36,
                ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False, **extra)
            for step, start in enumerate(range(0, 180, 36), 1):
                sl = slice(start, start+36)
                data = native_training_batch(observations[sl], scored[sl], advantages[sl], returns[sl],
                    c, role, torch, "cuda:0")
                if any(g["lr"] != c[f"{role}_lr"] for g in engine.optimizer.param_groups):
                    raise ValueError("Approved fixed LoRA learning rate changed")
                with engine.train_mode():
                    output = engine.train_batch(data, loss_function=partial(loss_fn, config=config))
                engine.lr_scheduler_step()
                norm = float(output["metrics"]["grad_norm"])
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("Native PPO gradient zero/nonfinite")
                append("updates", {"role": role, "optimizer_update": step, "actions": 36,
                    "grad_norm": norm, "lr": c[f"{role}_lr"], "discarded": True,
                    "elapsed_seconds": time.monotonic()-began})
                print(json.dumps({"event": "native_ppo_update", "role": role, "step": step,
                    "grad_norm": norm, "elapsed_seconds": time.monotonic()-began}), flush=True)
                del data, output
            after = train_state_fingerprint(engine)
            if after == before[role]:
                raise ValueError("Native PPO did not change any adapter/head/Adam state")
            head_changed = role != "critic"
            for name, p in engine.module.named_parameters():
                state = engine.optimizer.state.get(p, {})
                if not p.requires_grad:
                    if p.grad is not None or state:
                        raise ValueError("Frozen backbone/vision received PPO gradient/state")
                    continue
                grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                if grad is None or grad.dtype != torch.float32 or not torch.isfinite(grad).all() or float(state["step"]) != 6:
                    raise ValueError("Every adapter/head needs finite FP32 gradient and five actual new Adam steps")
                for key in ("exp_avg", "exp_avg_sq"):
                    moment = state[key].to_local() if hasattr(state[key], "to_local") else state[key]
                    if moment.dtype != torch.float32 or not torch.isfinite(moment).all():
                        raise ValueError("Native Adam moments must remain finite FP32")
                if ".score.modules_to_save.default." in name:
                    head_changed |= before[role][name]["parameter"] != after[name]["parameter"]
            if not head_changed:
                raise ValueError("Native critic scalar head was not updated")
            results[role] = {"optimizer_updates": 5, "trainable_tensors": len(after),
                "state_changed": True, "scalar_head_changed": head_changed if role == "critic" else None}
            (run/f"{role}-state-audit.json").write_text(artifact_json_dumps({"before": before[role], "after": after}))
        report = {"status": c["worker_status"], "source_commit": commit, "episodes": 20, "actions": 180,
            "optimizer_updates": 10, "roles": results, "native_episode_continuous_gae": True,
            "gamma": 1.0, "gae_lambda": 1.0, "mpss_exact_return": True, "discard_update": True,
            "training_ready": False, "books_outcomes_accessed": False, "ranking_metrics_computed": False,
            "research_checkpoint_produced": False, "sampler_and_train_sequential": True,
            "reference_scope": "native_disable_adapter_same_frozen_sft_base",
            "elapsed_seconds": time.monotonic()-began,
            "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_vram_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
        engines.clear()
        engine = base = layer = grad = p = state = advantages = returns = None
        gc.collect()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        (run/"native-worker-report.json").write_text(artifact_json_dumps(report))
    finally:
        engines.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
