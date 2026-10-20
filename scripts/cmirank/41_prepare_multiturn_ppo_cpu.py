#!/usr/bin/env python3
"""CPU20 complete-episode GAE/loss/packing rehearsal before any GPU handoff."""

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
from src.cmirank.multiturn_ppo import (load_multiturn_config, multiturn_code_hashes,
    reviewed_rollout_parent, prompt_row, validate_episodes, episode_gae, native_training_batch)
from src.cmirank.format_sft import synthetic_request
from src.cmirank.lora_rollout import rollout_host_budget, checkpoint_directory
from src.cmirank.lora_checkpoint import checkpoint_budget
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_multiturn_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private/"runs"/c["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(sys.prefix).resolve() != private/"envs"/runtime["environment_name"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Fresh exact-source CUDA-hidden CPU rehearsal required")
    started = time.monotonic()
    for name, version in {**runtime["versions"], "peft": "0.21.2"}.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned native runtime changed")
    parent = reviewed_rollout_parent(private, ROOT, c)
    host = rollout_host_budget(job, c)
    if not host["assessment"]["within_estimated_budget"]:
        raise ValueError("Original host margin missing; no GPU handoff")
    previous = parent["checkpoint_budget"]
    measured_rss = (previous["host_required_bytes"]-parent["host_budget"]["stats_bytes"]["anon"]
        -previous["largest_role_staging_bytes"]-c["checkpoint_host_workspace_bytes"])
    total = int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.total",
        "--format=csv,noheader,nounits"], text=True).strip())*2**20
    budget = checkpoint_budget(parent["roles"], total, host["allocation_memory"]["maximum_bytes"],
        host["stats_bytes"]["anon"], measured_rss, shutil.disk_usage(private/"runs").free, c)
    if not budget["within_estimated_budget"]:
        raise ValueError("Measured native restore budget exceeds unchanged card/host/disk limits")
    checkpoint = private/"runs"/compat["sft_run_id"]/"checkpoint"
    for name, sha in compat["checkpoint_files_sha256"].items():
        if file_sha256(checkpoint/name) != sha:
            raise ValueError("Frozen SFT initializer changed")
    for name, sha in c["native_checkpoint_sha256"].items():
        if file_sha256(checkpoint_directory(private, c)/name) != sha:
            raise ValueError("Immutable discarded T0 resume checkpoint changed")
    import torch
    from transformers import AutoTokenizer
    from verl.workers.config.actor import ActorConfig
    from verl.workers.config.critic import CriticConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, prepare_micro_batches
    from verl.workers.utils.losses import ppo_loss, value_loss
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    from verl.utils import tensordict_utils as tu
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from src.cmirank.corrected_rollout import generated_request
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    requests, observations, episodes, scores = [], [], [], []
    for i in range(20):
        request = synthetic_request(c["data_seed"], "holdout", i)
        requests.append(request.to_dict())
        active, exclusions = [f"C{j:02d}" for j in range(10)], []
        for turn in range(9):
            prompt = prompt_row(request, active, tokenizer, c, turn)
            CompletionRequest(**{**generated_request(prompt, c), "model": c["served_adapter_name"]})
            text = f"<answer>{active[0]}</answer>" + "\n"*(i % 3)
            suffix = tokenizer.encode(text+tokenizer.eos_token, add_special_tokens=False)
            n = len(suffix)
            obs = {**prompt, "input_ids": prompt["input_ids"]+suffix, "response_token_ids": suffix,
                "raw_text": text, "finish_reason": "stop", "eos_observed": True, "format_valid": True,
                "behavior_logprobs": [-2.0]*n}
            observations.append(obs)
            scores.append({"old_logprobs": [-2.0]*n, "reference_logprobs": [-2.01]*n,
                "critic_values": [0.1*(j+turn) for j in range(n)]})
            exclusions.append(active.pop(0))
        episodes.append({"user_id": request.user_id, "rank_request_sha256": request.sha256(),
            "exclusions": exclusions, "survivor": active[0], "valid": True})
    validate_episodes(observations, episodes, requests)
    adv, ret, audits = episode_gae(observations, episodes, [s["critic_values"] for s in scores], c, torch)
    probes = []
    # All20 episodes ×9 actions, both native losses, including KL and exact
    # next-token/EOS alignment. No native GPU output kernel is faked on CPU.
    for role in ("actor", "critic"):
        for start in range(0, 180, c["global_minibatch"]):
            sl = slice(start, start+c["global_minibatch"])
            data = native_training_batch(observations[sl], scores[sl], adv[sl], ret[sl], c, role, torch)
            tu.assign_non_tensor(data, dp_size=1,
                batch_num_tokens=sum(len(o["response_token_ids"]) for o in observations[sl]))
            micros, _ = prepare_micro_batches(data, dp_group=None, same_micro_num_in_dp=False)
            if len(micros) != 36:
                raise ValueError("Exactly36 native microbatches per optimizer step required")
            cls, fn = (ActorConfig, ppo_loss) if role == "actor" else (CriticConfig, value_loss)
            extra = {"use_kl_loss": True, "kl_loss_coef": c["kl_coef"], "kl_loss_type": c["kl_loss_type"]} if role == "actor" else {}
            config = cls(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=36,
                ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False, **extra)
            for obs, micro in zip(observations[sl], micros):
                proxy = SimpleNamespace(use_ulysses_sp=False, pad_to_length=False)
                inputs, _ = FSDPEngineWithLMHead.prepare_model_inputs(proxy, micro)
                if inputs["input_ids"][0].tolist() != obs["input_ids"]:
                    raise ValueError("Native complete prompt/response packing changed")
                field = "old_log_probs" if role == "actor" else "values"
                p = obs["prompt_tokens"]
                current = torch.cat((torch.zeros(p-1), micro[field].values().detach(), torch.zeros(1))).requires_grad_()
                output = {"log_probs" if role == "actor" else "values":
                    torch.nested.as_nested_tensor([current], layout=torch.jagged)}
                loss, _ = fn(config, output, micro)
                loss.backward()
                if (not torch.isfinite(loss) or not torch.isfinite(current.grad).all()
                        or current.grad[:p-1].count_nonzero() or current.grad[-1] != 0):
                    raise ValueError("Native CPU loss/response-only gradients invalid")
                probes.append({"role": role, "user_id": obs["user_id"], "turn": obs["turn"],
                    "loss": float(loss.detach()), "prompt_gradient_zero": True})
    if torch.cuda.is_initialized():
        raise ValueError("CPU rehearsal initialized CUDA")
    run.mkdir()
    for name, rows in (("requests", requests), ("probes", probes), ("gae-probes", audits)):
        (run/f"{name}.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in rows))
    sources = dict(parent["native_source_sha256"])
    for obj in (compute_gae_advantage_return, ppo_loss, value_loss, prepare_micro_batches):
        file = Path(inspect.getfile(obj))
        sources[str(file.relative_to(Path(sys.prefix)))] = file_sha256(file)
    report = {**{k: parent[k] for k in ("roles", "targets", "overlay_sha256", "checkpoint_stat",
        "native_checkpoint_stat", "checkpoint_budget", "eos_token_id")},
        "status": "MULTITURN_PPO_CPU20_PASS_GPU_GATE_REMAINS", "source_commit": commit,
        "config_sha256": file_sha256(path), "code_sha256": multiturn_code_hashes(ROOT),
        "episodes": 20, "turns": 180, "native_loss_probes": 360, "gpu_requested": False,
        "optimizer_updates": 0, "training_ready": False, "books_outcomes_accessed": False,
        "host_budget": host, "checkpoint_budget": budget, "native_source_sha256": sources,
        "artifact_sha256": {p.name: file_sha256(p) for p in run.glob("*.jsonl")},
        "elapsed_seconds": time.monotonic()-started}
    (run/"report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "episodes", "native_loss_probes", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
