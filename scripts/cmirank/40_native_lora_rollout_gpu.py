#!/usr/bin/env python3
"""Cold native actor/critic resume, adapter-disabled reference,20 fresh rollouts.

Only a task-owned process sees GPU1. No gradient, optimizer update or GAE;
the native engines restore their ordinary complete model/Adam/extra checkpoint.
"""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.lora import audit_scope
from src.cmirank.lora_joint import native_model_config, native_engine_config, native_optimizer_config
from src.cmirank.lora_checkpoint import native_checkpoint_config, train_state_fingerprint, rng_probe
from src.cmirank.lora_rollout import load_lora_rollout_config, verify_lora_rollout_preparation, checkpoint_directory
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
    c, path = load_lora_rollout_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private/"runs"/c["run_id"] or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Exact clean-source native cold-resume worker required")
    rows, prepared = verify_lora_rollout_preparation(private, ROOT, c, path, commit)
    manifest = json.loads((run/"manifest.json").read_text())
    observations = [json.loads(x) for x in (run/"generated.jsonl").read_text().splitlines()]
    if (file_sha256(run/"generated.jsonl") != manifest["generated_sha256"]
            or [r["user_id"] for r in rows] != [r["user_id"] for r in observations]):
        raise ValueError("All20 actual adapter-sampled responses must be bound to this run")
    def interrupted(signum, frame):
        raise InterruptedError(f"Native cold-resume worker interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    import torch
    import torch.distributed as dist
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
                != canonical_gpu_uuid(os.environ["MEMREC_SELECTED_GPU_UUID"])):
        raise ValueError("Native cold resume may bind only physical GPU1")
    torch.set_num_threads(4)
    torch.manual_seed(c["seed"])
    began = time.monotonic()
    engines, scores, controls = {}, {}, []
    def append(name, row):
        with (run/f"{name}.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
    def readouts(engine, entries, role, reference=False):
        values = []
        with engine.eval_mode(), torch.no_grad(), (engine.disable_adapter() if reference else nullcontext()):
            for row in entries:
                data = native_microbatch(torch.tensor(row["input_ids"], device="cuda:0"), c, torch)
                _, output = engine.forward_step(data, loss_function=None, forward_only=True)
                key = "log_probs" if role == "actor" else "values"
                response = output["model_output"][key].values()[row["prompt_tokens"]-1:len(row["input_ids"])-1]
                if response.numel() != len(row["response_token_ids"]) or not torch.isfinite(response).all():
                    raise ValueError("Native cold-resume response/EOS readout invalid")
                values.append(response.float().cpu().tolist())
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
                raise ValueError("Cold native LoRA scope/storage changed")
            base = engine.module.get_base_model()
            layer = base.model.language_model.layers[0].linear_attn
            if (not hf.is_fast_path_available or not layer.chunk_gated_delta_rule.__module__.startswith("fla.")
                    or layer.causal_conv1d_fn is not hf.causal_conv1d_fn):
                raise ValueError("Cold resume lost the audited native kernel path")
            audit = json.loads((private/"runs"/c["checkpoint_parent_run_id"]/f"checkpoint-{role}-audit.json").read_text())
            if (train_state_fingerprint(engine) != audit["train_state_before"]
                    or engine.lr_scheduler.state_dict() != audit["scheduler_before"] or rng_probe() != audit["expected_rng_draws"]):
                raise ValueError("Fresh native process failed complete role/Adam/scheduler/RNG restore")
            frozen = readouts(engine, rows, role)
            for row, values, expected in zip(rows, frozen, audit["response_values_before"]):
                append("cold-restore-controls", {"role": role, "user_id": row["user_id"],
                    "values": values, "expected_checkpoint_values": expected})
            if frozen != audit["response_values_before"]:
                raise ValueError("Cold native checkpoint changed one of20 original response readouts")
            pair = [readouts(engine, observations, role) for _ in range(2)]
            if pair[0] != pair[1]:
                raise ValueError("Same native training engine failed exact repeated fresh-rollout scoring")
            scores[role] = pair
            if role == "actor":
                reference_frozen = readouts(engine, rows, role, reference=True)
                original = private/"runs"/"cmirank-lora-joint-optimizer-smoke-v1-20261015-hnv"/"native-vectors.jsonl"
                if file_sha256(original) != c["reference_control_vectors_sha256"]:
                    raise ValueError("Frozen reference vectors changed since CPU review")
                initial = {x["user_id"]: x["values"] for x in map(json.loads, original.read_text().splitlines()) if x["role"] == "actor"}
                for row, ref in zip(rows, reference_frozen):
                    append("reference-controls", {"user_id": row["user_id"], "reference_values": ref,
                        "expected_base_values": initial[row["user_id"]]})
                if reference_frozen != [initial[r["user_id"]] for r in rows]:
                    raise ValueError("Adapter-disabled reference differs from frozen initial base on20 controls")
                reference = readouts(engine, observations, role, reference=True)
                if train_state_fingerprint(engine) != audit["train_state_before"]:
                    raise ValueError("Reference context changed actor adapter/Adam state")
            torch.cuda.synchronize()
            append("memory", {"event": f"native_{role}_cold_restored_scored",
                "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20})
        for i, obs in enumerate(observations):
            append("native-logprobs", {"user_id": obs["user_id"],
                "old_logprobs": scores["actor"][0][i], "recomputed_logprobs": scores["actor"][1][i],
                "reference_logprobs": reference[i], "reference_scope": "native_disable_adapter_same_frozen_base_control20_exact",
                "critic_values": scores["critic"][0][i], "recomputed_critic_values": scores["critic"][1][i],
                "response_tokens_including_eos": len(obs["response_token_ids"])})
        report = {"status": c["worker_status"], "source_commit": commit, "samples": 20,
            "cold_process_restart_proven": True, "actor_and_critic_complete_train_state_equal": True,
            "cold_rng_probe_equal": True,
            "adapter_disabled_reference_control_samples": 20, "reference_control_max_delta": 0.0,
            "optimizer_constructed": True, "optimizer_updates": 0, "training_ready": False,
            "ranking_metrics_computed": False, "books_outcomes_accessed": False,
            "full_ppo_fit_proven": False, "multiturn_gae_proven": False,
            "model_dtype": "fp32", "forward_dtype": "bfloat16", "elapsed_seconds": time.monotonic()-began,
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_vram_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
        engines.clear()
        engine = base = layer = None
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
