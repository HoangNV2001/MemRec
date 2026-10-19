"""Native LoRA resume qualification; no imports that initialize CUDA.

Use the default complete native checkpoint. The pinned VeRL LoRA-only filter
matches lora_/adapter_ names, not the critic's modules_to_save scalar head.
This is a serialization choice, not permission to train the frozen backbone.
"""

import hashlib
import json
from pathlib import Path
import subprocess

from .lora_joint import load_joint_config, joint_code_hashes, reviewed_lora_parent
from .provenance import file_sha256, artifact_json_dumps


def load_checkpoint_config(root):
    c, _ = load_joint_config(root)
    path = root / "configs/cmirank/ppo_lora_checkpoint_v1.json"
    r = json.loads(path.read_text())
    fixed = {"schema_version": 1,
        "prepare_run_id": "cmirank-lora-checkpoint-data-v1-20261019-hnv",
        "run_id": "cmirank-lora-checkpoint-smoke-v1-20261019-hnv",
        "joint_parent_source_commit": "9e3e9f32ca0bdfa5dd197e9ea0e5cdd92f7c3d5c",
        "joint_parent_report_sha256": "896445dcaffd50c539b33e900d684964807025571a2f5e35e703463a10b4aa39",
        "joint_parent_prepare_report_sha256": "6d51b8bff84e939141c78314fd0d2f3a0b5a22f171e1ee3f56bb2072a6112ae9",
        "parameterization": "lora_checkpoint", "optimizer_updates": 6, "updates_per_role": 3,
        "restored_logical_updates_per_role": 2, "checkpoint_save_lora_only": False,
        "checkpoint_contents": ["model", "optimizer", "extra"],
        "checkpoint_host_workspace_bytes": 8*2**30, "checkpoint_disk_reserve_bytes": 8*2**30,
        "worker_status": "NATIVE_LORA_CHECKPOINT_PASS_ROLLOUT_MULTITURN_GATES_REMAIN"}
    if r != fixed or any(type(r[k]) is not type(v) for k, v in fixed.items()):
        raise ValueError("Only the preregistered native20 checkpoint/resume gate is allowed")
    c.update(r)
    return c, path


def checkpoint_code_hashes(root):
    names = ("configs/cmirank/ppo_lora_checkpoint_v1.json", "src/cmirank/lora_checkpoint.py")
    return {**joint_code_hashes(root), **{n: file_sha256(root/n) for n in names}}


def reviewed_joint_parent(private, root, c):
    # Review each artifact against its producing commit, never demand that an
    # already-completed run was produced by the new checkpoint gate's source.
    original, _ = load_joint_config(root)
    run = private/"runs"/original["run_id"]
    prep = private/"runs"/original["prepare_run_id"]
    if (file_sha256(run/"report.json") != c["joint_parent_report_sha256"]
            or file_sha256(prep/"report.json") != c["joint_parent_prepare_report_sha256"]):
        raise ValueError("Reviewed joint native GPU20 parent changed")
    r = json.loads((run/"report.json").read_text())
    p = json.loads((prep/"report.json").read_text())
    m = json.loads((run/"manifest.json").read_text())
    if (r["status"] != original["worker_status"] or r["source_commit"] != c["joint_parent_source_commit"]
            or p["source_commit"] != r["source_commit"] or r["samples"] != 20
            or r["optimizer_updates"] != 2 or not r["actor_critic_joint_fit_proven"]
            or r["full_ppo_fit_proven"] or r["training_ready"] or r["books_outcomes_accessed"]
            or file_sha256(run/"native-worker-report.json") != r["worker_report_sha256"]
            or m["preparation_report_sha256"] != c["joint_parent_prepare_report_sha256"]):
        raise ValueError("Joint parent must be a completed isolated native two-role update gate")
    for name, sha in p["code_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe producing-source path")
        content = subprocess.check_output(["git", "show", r["source_commit"]+":"+name], cwd=root)
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError("Joint CPU receipt does not match its producing source")
    for name, sha in p["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(prep/name) != sha:
            raise ValueError("Joint CPU data changed")
    for name in ("cleanup", "child-cleanup", "native-release"):
        a = json.loads((run/f"{name}.json").read_text())
        if name == "cleanup" and (a["process_exit_code"] or not a["gpu_released"] or not a["allocation_still_running"]):
            raise ValueError("Joint parent cleanup failed")
        if name == "child-cleanup" and (a["process_exit_code"] or not a["owned_hf_worker_exited"]):
            raise ValueError("Joint parent worker did not exit")
        if name == "native-release" and (not a["child_exited"] or not a["gpu_released"]):
            raise ValueError("Joint parent native release failed")
    handback = json.loads((run/"keeper-handback.json").read_text())
    if (handback["status"] != "KEEPER_GPU1_HANDBACK_VERIFIED" or handback["gpu0_signalled"]
            or handback["reserved_job_cancelled"]):
        raise ValueError("Joint parent must have restored keeper1 without touching protected resources")
    reviewed_lora_parent(private, root, c)
    return r


def checkpoint_budget(roles, gpu_bytes, host_cap_bytes, host_anon_bytes, parent_rss_bytes,
                      disk_free_bytes, c):
    if set(roles) != {"actor", "critic"}:
        raise ValueError("Both native LoRA roles required")
    values = (gpu_bytes, host_cap_bytes, host_anon_bytes, parent_rss_bytes, disk_free_bytes)
    if any(type(x) is not int or x < 0 for x in values) or min(gpu_bytes, host_cap_bytes, parent_rss_bytes) <= 0:
        raise ValueError("Actual positive allocation/card and measured parent RSS required")
    # Native FSDP2 state context is null; do not assume CPU staging. Conservatively
    # allow an entire model+Adam role copy on BOTH GPU and host during restore.
    largest = max(r["parameter_bytes"]+r["adam_moment_bytes"] for r in roles.values())
    gpu_need = sum(r["persistent_train_bytes"] for r in roles.values())+largest+c["workspace_reserve_bytes"]
    host_need = host_anon_bytes+parent_rss_bytes+largest+c["checkpoint_host_workspace_bytes"]
    disk_need = sum(r["parameter_bytes"]+r["adam_moment_bytes"] for r in roles.values())+c["checkpoint_disk_reserve_bytes"]
    return {"method": "both_roles_plus_one_native_checkpoint_restore_copy_not_fit_proof",
        "largest_role_staging_bytes": largest, "gpu_required_bytes": gpu_need, "gpu_bytes": gpu_bytes,
        "host_required_bytes": host_need, "host_cap_bytes": host_cap_bytes,
        "disk_required_bytes": disk_need, "disk_free_bytes": disk_free_bytes,
        "within_estimated_budget": gpu_need <= gpu_bytes and host_need <= host_cap_bytes and disk_need <= disk_free_bytes,
        "full_ppo_fit_proven": False, "checkpoint_fit_proven": False}


def native_checkpoint_config(c):
    from verl.trainer.config import CheckpointConfig
    return CheckpointConfig(save_contents=list(c["checkpoint_contents"]),
        load_contents=list(c["checkpoint_contents"]), save_lora_only=c["checkpoint_save_lora_only"],
        async_save=False, strict=True)


def verify_checkpoint_preparation(private, root, c, path, commit):
    run = private/"runs"/c["prepare_run_id"]
    p = json.loads((run/"report.json").read_text())
    if (p["status"] != "LORA_CHECKPOINT_CPU20_PASS_GPU_GATE_REMAINS" or p["source_commit"] != commit
            or p["config_sha256"] != file_sha256(path) or p["code_sha256"] != checkpoint_code_hashes(root)
            or p["samples_per_role"] != 20 or p["gpu_requested"] or p["full_weights_loaded"]
            or p["optimizer_updates"] or p["training_ready"] or not p["checkpoint_budget"]["within_estimated_budget"]):
        raise ValueError("Fresh exact-source CPU20 native checkpoint preparation required")
    for name, sha in p["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(run/name) != sha:
            raise ValueError("Checkpoint CPU20 artifacts changed")
    # Large immutable parent artifacts were reviewed in CPU preparation. Native
    # GPU launch rechecks their bound reports, not repeated full-weight hashing.
    original, _ = load_joint_config(root)
    if (file_sha256(private/"runs"/original["run_id"]/"report.json") != c["joint_parent_report_sha256"]
            or file_sha256(private/"runs"/c["cpu_parent_run_id"]/"report.json") != c["cpu_parent_report_sha256"]):
        raise ValueError("Reviewed LoRA dependency reports changed since preparation")
    return p


def tensor_digest(tensor):
    """Exact complete tensor bytes, one CPU tensor at a time; never a probe hash."""
    x = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    x = x.detach().contiguous().cpu()
    descriptor = json.dumps([str(x.dtype), list(x.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(descriptor+x.numpy().tobytes()).hexdigest()


def train_state_fingerprint(engine):
    states = {}
    for name, p in engine.module.named_parameters():
        if not p.requires_grad:
            continue
        state = engine.optimizer.state[p]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Complete standard AdamW state required for every adapter/head")
        states[name] = {"parameter": tensor_digest(p),
            **{key: tensor_digest(value) for key, value in state.items()}}
    if not states:
        raise ValueError("No native adapter/head state to restore")
    return states


def rng_probe():
    import random
    import numpy as np
    import torch
    return {"python": [random.random() for _ in range(8)],
        "numpy": np.random.random(8).tolist(), "torch_cpu": torch.rand(8).tolist(),
        "torch_cuda": torch.rand(8, device="cuda:0").cpu().tolist()}


def run_checkpoint_roundtrip(engine, data, loss_fn, c, role, run, memory, score):
    """Save after step1, change via native step2, restore, continue native step2.

    Three physical updates/role, two logical updates after resume, all discarded.
    Exact restored state and20 response readouts are required; we do not claim
    uninterrupted-vs-resumed next-gradient bitwise parity for nondeterministic kernels.
    """
    from copy import deepcopy
    import gc
    import math
    import torch
    engine.lr_scheduler_step()
    original = train_state_fingerprint(engine)
    scheduler = deepcopy(engine.lr_scheduler.state_dict())
    expected_scores = score(engine)
    directory = run/"native-checkpoints"/role
    if directory.exists():
        raise ValueError("Never overwrite or rotate a previous checkpoint")
    engine.save_checkpoint(str(directory), global_step=1, max_ckpt_to_keep=None)
    memory(f"native_{role}_checkpoint_saved")
    # Capture expected draws immediately after save; native resume must recover
    # all RNG families without a hand-written state reset in the test harness.
    expected_rng = rng_probe()
    with engine.train_mode():
        changed = engine.train_batch(data, loss_function=loss_fn)
    engine.lr_scheduler_step()
    if (not math.isfinite(float(changed["metrics"]["grad_norm"]))
            or train_state_fingerprint(engine) == original or engine.lr_scheduler.state_dict() == scheduler):
        raise ValueError("Native second update did not perturb the checkpoint's training state")
    engine.load_checkpoint(str(directory), del_local_after_load=False)
    memory(f"native_{role}_checkpoint_restored")
    restored = train_state_fingerprint(engine)
    restored_rng = rng_probe()
    if restored != original or engine.lr_scheduler.state_dict() != scheduler or restored_rng != expected_rng:
        raise ValueError("Native resume failed exact adapter/head/Adam/scheduler/RNG restoration")
    restored_scores = score(engine)
    if restored_scores != expected_scores or len(restored_scores) != 20:
        raise ValueError("Native checkpoint changed any of20 response/EOS role readouts")
    audit_path = run/f"checkpoint-{role}-audit.json"
    audit_path.write_text(artifact_json_dumps({"role": role, "train_state_before": original,
        "train_state_restored": restored, "scheduler_before": scheduler,
        "scheduler_restored": engine.lr_scheduler.state_dict(), "expected_rng_draws": expected_rng,
        "restored_rng_draws": restored_rng,
        "response_values_before": expected_scores, "response_values_restored": restored_scores}))
    with engine.train_mode():
        continued = engine.train_batch(data, loss_function=loss_fn)
        norm = float(continued["metrics"]["grad_norm"])
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Restored native optimizer continuation failed")
        for p in engine.module.parameters():
            if not p.requires_grad:
                if p.grad is not None or engine.optimizer.state.get(p):
                    raise ValueError("Restored frozen parameter received gradient/Adam state")
                continue
            state = engine.optimizer.state[p]
            grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            if (float(state["step"]) != 2 or grad is None or grad.dtype != torch.float32
                    or not torch.isfinite(grad).all()):
                raise ValueError("Restored LoRA optimizer must continue at native step2 with finiteFP32 gradients")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key].to_local() if hasattr(state[key], "to_local") else state[key]
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("Restored AdamW moments must remain finiteFP32")
    engine.lr_scheduler_step()
    if train_state_fingerprint(engine) == original:
        raise ValueError("Native resumed continuation did not change adapters/head")
    memory(f"native_{role}_checkpoint_continued_logical_step2")
    result = {"samples": 20, "physical_updates_per_role": 3, "logical_updates_after_resume": 2,
        "complete_trainable_and_adam_state_equal": True, "scheduler_state_equal": True,
        "rng_probe_families": ["python", "numpy", "torch_cpu", "torch_cuda"],
        "rng_probe_equal": True, "response_readout_max_absolute_delta": 0.0,
        "native_continuation_step": 2, "native_continuation_grad_norm": norm,
        "cold_process_restart_proven": False,
        "uninterrupted_next_update_bitwise_parity_proven": False,
        "frozen_backbone_full_bitwise_equality_proven": False,
        "serialized_bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
        "audit_sha256": file_sha256(audit_path),
        "trainable_state_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()}
    del original, restored, changed, continued
    gc.collect()
    return result
