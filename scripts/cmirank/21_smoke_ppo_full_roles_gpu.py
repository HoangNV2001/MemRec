#!/usr/bin/env python3
"""Twenty native-kernel/full4B role probes on GPU0; zero PPO/optimizer updates.

CPU inputs/hash preparation is a separate finished task. Only then hand off the
exact current keeper; bind before torch imports. Native VeRL critic loader, no
custom critic/optimizer. Immediate model release and outer keeper handback.
"""

import argparse
import gc
import importlib.metadata
import json
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
from src.cmirank.gpu_resources import canonical_gpu_uuid, select_reserved_gpu, verified_numeric_cuda_binding
from src.cmirank.ppo_compat import compat_code_hashes, load_compat_config, verify_preparation
from src.cmirank.ppo_runtime import load_runtime_config, response_reward_layout
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.reserved_handoff import gpu_snapshot, handoff
from src.cmirank.rewards import mpss_rewards, ndcg_at_k


def save(path, value):
    path.write_text(artifact_json_dumps(value))


def freeze_vision(model, torch, *, reference=False):
    vision, text = 0, 0
    for name, parameter in model.named_parameters():
        if name.startswith("v_head."):
            continue
        if ".visual." in name:
            parameter.requires_grad_(False)
            vision += parameter.numel()
        else:
            parameter.requires_grad_(not reference)
            text += parameter.numel()
            if parameter.dtype != torch.bfloat16:
                raise ValueError("Full text backbone is not BF16")
    if (text, vision) != (4205751296, 333514240):
        raise ValueError("Official full-text/frozen-vision parameter boundary changed")
    return {"text_parameters": text, "frozen_vision_parameters": vision}


def gradient_check(model, torch):
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise ValueError("Missing/nonfinite/zero full-role gradients")
    if any(p.grad is not None for n, p in model.named_parameters() if ".visual." in n):
        raise ValueError("Unused vision received gradients")
    model.zero_grad(set_to_none=True)
    return float(norm.item())


def response_logprobs(model, row, torch):
    ids = torch.tensor([row["input_ids"]], device="cuda:0")
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    n = row["prompt_tokens"]
    # Predict each response token from its preceding position, including EOS.
    logits = output.logits[:, n - 1:-1].float()
    probabilities = logits.log_softmax(-1).gather(-1, ids[:, n:].unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(probabilities).all() or probabilities.shape[-1] != row["completion_tokens"]:
        raise ValueError("Malformed/nonfinite response-only log probabilities")
    return probabilities


def kernel_probes(torch, config, head_dim, emit):
    from flash_attn import flash_attn_func
    for i in range(config["samples"]):
        size = 16 + i
        inputs = [torch.randn(1, size, 2, head_dim, device="cuda:0", dtype=torch.bfloat16,
                              requires_grad=True) for _ in range(3)]
        output = flash_attn_func(*inputs, dropout_p=0.0, causal=True)
        reference_inputs = [x.detach().float().requires_grad_(True) for x in inputs]
        q, k, v = [x.transpose(1, 2) for x in reference_inputs]
        scores = q @ k.transpose(-1, -2) / head_dim ** 0.5
        mask = torch.ones(size, size, dtype=torch.bool, device="cuda:0").triu(1)
        reference = (scores.masked_fill(mask, float("-inf")).softmax(-1) @ v).transpose(1, 2)
        forward = (output.float() - reference).abs().max().item()
        output.float().square().mean().backward()
        reference.square().mean().backward()
        backward = max((x.grad.float() - r.grad).abs().max().item() for x, r in zip(inputs, reference_inputs))
        relative_backward = max(((x.grad.float() - r.grad).norm() / r.grad.norm().clamp_min(1e-12)).item()
                                for x, r in zip(inputs, reference_inputs))
        if (not torch.isfinite(output).all() or any(not torch.isfinite(x.grad).all() for x in inputs)
                or forward > config["kernel_forward_atol"] or backward > config["kernel_backward_atol"]
                or relative_backward > config["kernel_backward_relative_l2_tol"]):
            raise ValueError("FA2 CUDA forward/backward differs from explicit causal attention reference")
        emit("kernel", {"sample": i, "head_dim": head_dim, "sequence_length": size,
                        "forward_max_absolute_delta": forward, "backward_max_absolute_delta": backward,
                        "backward_max_relative_l2_delta": relative_backward})
        del output, reference, inputs, reference_inputs, q, k, v, scores, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner-confirmed-generator-step", required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config, path = load_compat_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    overlay = private / "overlays" / config["overlay_name"]
    if (run != private / "runs" / config["run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 4
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or os.environ.get("PYTHONPATH") != str(overlay)):
        raise ValueError("Fresh exact-source bounded private compatibility task required")
    for name, version in runtime["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned PPO runtime changed")
    rows, prepared = verify_preparation(private / "runs" / config["prepare_run_id"], commit=commit,
        config_sha=file_sha256(path), code_sha=compat_code_hashes(ROOT), config=config)
    # Hash only the compact overlay before handoff. Large checkpoint hashes
    # were completed by the separate CPU task; verify unchanged stat receipt.
    for name, sha in prepared["overlay_sha256"].items():
        if file_sha256(overlay / name) != sha:
            raise ValueError("Prepared native kernel overlay changed")
    checkpoint = private / "runs" / config["sft_run_id"] / "checkpoint"
    for name, expected in prepared["checkpoint_stat"].items():
        stat = (checkpoint / name).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino} != expected:
            raise ValueError("Checkpoint changed since full CPU SHA verification")
    run.mkdir()
    began = time.monotonic()
    actor = reference = critic = torch = card = None
    exit_code = 1
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned role smoke interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def emit(name, row):
        with (run / f"{name}.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if (row.get("sample", 0) + 1) % 5 == 0:
            print(json.dumps({"stage": name, "completed_samples": row["sample"] + 1,
                              "elapsed_seconds": time.monotonic() - began}), flush=True)
    manifest = {"source_commit": commit, "config": config, "config_sha256": file_sha256(path),
        "preparation_report_sha256": file_sha256(private / "runs" / config["prepare_run_id"] / "report.json"),
        "slurm_job_id": job, "optimizer_updates": 0, "generation_calls": 0,
        "training_ready": False, "books_outcomes_accessed": False}
    save(run / "manifest.json", manifest)
    try:
        manifest["gpu0_handoff"] = handoff(run, args.owner_confirmed_generator_step, gpu_index=0, protect_other_workloads=True)
        cards, apps, raw, raw_apps = gpu_snapshot()
        card = select_reserved_gpu(cards, set(apps), gpu_index=0)
        if card.uuid != manifest["gpu0_handoff"]["gpu0_uuid"]:
            raise ValueError("Handoff and selected card differ")
        (run / "gpus-before.csv").write_text(raw)
        (run / "apps-before.csv").write_text(raw_apps)
        cards, apps, raw, raw_apps = gpu_snapshot()
        current = select_reserved_gpu(cards, set(apps), gpu_index=0)
        if current.uuid != card.uuid:
            raise ValueError("Card changed immediately before CUDA import")
        card = current
        pci = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"], text=True)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = verified_numeric_cuda_binding(cards, card, pci)
        import torch
        from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
        from verl.utils.model import load_valuehead_model
        from verl.trainer.ppo.core_algos import compute_gae_advantage_return
        if (torch.cuda.device_count() != 1
                or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid)) != canonical_gpu_uuid(card.uuid)):
            raise ValueError("Logical CUDA0 is not the approved physical GPU0")
        torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"], 0)
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        manifest.update(gpu_uuid=card.uuid, physical_gpu_index=card.index,
                        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], torch_cuda_version=torch.version.cuda)
        save(run / "manifest.json", manifest)
        kernel_probes(torch, config, prepared["head_dim"], emit)  # First20 before full4B loading.
        def load(directory):
            model, info = Qwen3_5ForConditionalGeneration.from_pretrained(directory, local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0},
                attn_implementation=config["attn_implementation"], output_loading_info=True)
            if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError("Full model loading keys differ")
            model.eval()
            return model
        actor = load(checkpoint)
        scope = freeze_vision(actor, torch)
        reference = load(checkpoint)
        freeze_vision(reference, torch, reference=True)
        # Keep both actual full models resident on the same physical card for
        # real identical-weight parity, not a CPU reference or tiny substitute.
        for i, row in enumerate(rows):
            probabilities = response_logprobs(actor, row, torch)
            with torch.no_grad():
                expected = response_logprobs(reference, row, torch)
            delta = (probabilities.detach() - expected).abs().max().item()
            if delta > config["logprob_atol"]:
                raise ValueError("Identical-weight full actor/reference response log probabilities differ")
            (-probabilities.mean()).backward()
            norm = gradient_check(actor, torch)
            if any(p.grad is not None for p in reference.parameters()):
                raise ValueError("Frozen reference received gradients")
            emit("actor-reference", {"sample": i, "user_id": row["user_id"],
                 "response_logprob_max_absolute_delta": delta, "full_text_gradient_norm": norm})
            del probabilities, expected
        # Full standard HF checkpoint roundtrip, no trained ranking checkpoint.
        reference = None
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            before = [response_logprobs(actor, row, torch).cpu() for row in rows]
        saved = run / "discarded-compatibility-checkpoint"
        actor.save_pretrained(saved, safe_serialization=True, max_shard_size="2GB")
        actor = reference = None
        gc.collect()
        torch.cuda.empty_cache()
        actor = load(saved)
        freeze_vision(actor, torch, reference=True)
        with torch.no_grad():
            after = [response_logprobs(actor, row, torch).cpu() for row in rows]
        roundtrip = max((a - b).abs().max().item() for a, b in zip(before, after))
        if roundtrip > config["checkpoint_atol"]:
            raise ValueError("Full checkpoint changed response probabilities")
        actor = None
        del before, after
        gc.collect()
        torch.cuda.empty_cache()
        # Invoke VeRL's actual production loader unchanged, not our tiny probe
        # wrapper or a new classifier. No monkey-patched attention/optimizer.
        hf_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        hf_config.num_labels = 1
        hf_config.classifier_dropout = 0.0
        hf_config.hidden_dropout = "0"
        hf_config.summary_dropout_prob = 0.0
        critic = load_valuehead_model(str(checkpoint), torch.bfloat16, hf_config, False).to("cuda:0").eval()
        scope_critic = freeze_vision(critic, torch)
        if not hasattr(critic, "v_head"):
            raise ValueError("Unexpected native critic architecture; no classifier substitution")
        labels = [f"C{i:02d}" for i in range(10)]
        for i, row in enumerate(rows):
            ids = torch.tensor([row["input_ids"]], device="cuda:0")
            _, _, values = critic(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            if values.shape != ids.shape or not torch.isfinite(values).all():
                raise ValueError("Native full critic value shape/nonfinite output")
            values[:, row["prompt_tokens"]:].square().mean().backward()
            norm = gradient_check(critic, torch)
            rewards, rank = mpss_rewards(labels, labels[:-1], labels[-1], labels[i % 10])
            placed, _ = response_reward_layout([row["completion_tokens"]] * 9, list(rewards))
            reward_tensor = torch.tensor([placed], device="cuda:0", dtype=torch.float32)
            _, returns = compute_gae_advantage_return(reward_tensor, torch.zeros_like(reward_tensor),
                torch.ones_like(reward_tensor), gamma=1.0, lam=1.0)
            oracle = reward_tensor.flip(-1).cumsum(-1).flip(-1)
            error = (returns - oracle).abs().max().item()
            if error > 1e-6 or abs(sum(placed) - ndcg_at_k(rank)) > 1e-12:
                raise ValueError("CUDA gamma1/response-only MPSS/GAE mismatch")
            emit("native-critic", {"sample": i, "user_id": row["user_id"],
                 "full_role_gradient_norm": norm, "gae_max_absolute_delta": error})
            del ids, values, reward_tensor, returns, oracle
        # Native value-head state save/reload uses the same full critic;
        # full PPO optimizer/FSDP checkpointing remains a later gate.
        head = {k: v.detach().cpu().clone() for k, v in critic.v_head.state_dict().items()}
        torch.save(head, run / "discarded-value-head.pt")
        critic.v_head.load_state_dict(torch.load(run / "discarded-value-head.pt", weights_only=True), strict=True)
        head_equal = all(torch.equal(v.cpu(), head[k]) for k, v in critic.v_head.state_dict().items())
        if not head_equal:
            raise ValueError("Value-head serialization changed state")
        result = {"status": "FULL_4B_ROLE_COMPAT_PASS_CLEANUP_REVIEW_AND_PPO_WORKER_GATES_REMAIN",
            "source_commit": commit, "config_sha256": file_sha256(path), "preparation_report_sha256": manifest["preparation_report_sha256"],
            "kernel_samples": 20, "actor_reference_samples": 20, "native_critic_samples": 20,
            "actor_scope": scope, "critic_backbone_scope": scope_critic,
            "native_critic_loader": "verl.utils.model.load_valuehead_model_unmodified",
            "checkpoint_max_response_logprob_delta": roundtrip, "value_head_state_roundtrip": head_equal,
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "process_host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "co_residency_scope": "actor_and_reference_together_critic_after_unload_no_optimizer_or_rollout_engine",
            "optimizer_updates": 0, "generation_calls": 0, "training_ready": False,
            "ppo_worker_or_rollout_proven": False, "books_outcomes_accessed": False,
            "ranking_metrics_computed": False, "elapsed_seconds": time.monotonic() - began,
            "remaining_gates": ["vLLM_rollout_trainer_logprob_parity", "FSDP_PPO_worker_update_and_full_optimizer_memory",
                                "PPO_checkpoint_roundtrip", "canonical_primary_memory", "ranking_evaluation"]}
        # Release models before report/hash/review housekeeping.
        critic = None
        del head
        gc.collect()
        torch.cuda.empty_cache()
        save(run / "report.json", result)
        exit_code = 0
        print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "peak_vram_allocated_mib")}), flush=True)
    except BaseException as error:
        save(run / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                    "elapsed_seconds": time.monotonic() - began})
        raise
    finally:
        actor = reference = critic = None
        if torch is not None:
            gc.collect()
            torch.cuda.empty_cache()
        save(run / "child-cleanup.json", {"process_exit_code": exit_code, "gpu_uuid": card.uuid if card else None,
                                         "training_ready": False, "optimizer_updates": 0})


if __name__ == "__main__":
    main()
