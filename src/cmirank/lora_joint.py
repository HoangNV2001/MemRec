"""Single-card native LoRA actor+critic qualification, no eager CUDA imports."""

import json
from pathlib import Path

from .lora import load_lora_config, lora_code_hashes
from .provenance import file_sha256


def load_joint_config(root):
    c = load_lora_config(root)[0].copy()
    path = root / "configs/cmirank/ppo_lora_joint_v1.json"
    r = json.loads(path.read_text())
    fixed = {"schema_version": 1, "prepare_run_id": "cmirank-lora-joint-optimizer-data-v1-20261015-hnv",
        "run_id": "cmirank-lora-joint-optimizer-smoke-v1-20261015-hnv",
        "cpu_parent_run_id": "cmirank-lora-native-cpu-v4-20261015-hnv",
        "cpu_parent_source_commit": "8ff2dc33a86dab1ab2d154368a1c087728289d16",
        "cpu_parent_report_sha256": "565b51744ca72be75139bd662ed02900d62520932d893cce9c4c463518913663",
        "parameterization": "lora_joint", "role": "joint", "optimizer_updates": 2, "updates_per_role": 1,
        "gpu_host_admission": "cpu_hidden_cache_estimate_not_fit_proof",
        "worker_status": "NATIVE_LORA_JOINT_UPDATE_PASS_ROLLOUT_CHECKPOINT_MULTITURN_GATES_REMAIN"}
    if r != fixed or any(type(r[k]) is not type(v) for k, v in fixed.items()):
        raise ValueError("Only two discarded native LoRA role updates on the reviewed20 cases")
    c.update(r)
    return c, path


def reviewed_lora_parent(private, root, c):
    run = private / "runs" / c["cpu_parent_run_id"]
    if file_sha256(run / "report.json") != c["cpu_parent_report_sha256"]:
        raise ValueError("Reviewed CPU LoRA parent changed")
    p = json.loads((run / "report.json").read_text())
    if (p["status"] != "LORA_NATIVE_CPU20_PASS_FULL_GPU_AND_MULTITURN_GATES_REMAIN"
            or p["source_commit"] != c["cpu_parent_source_commit"] or p["samples_per_role"] != 20
            or p["code_sha256"] != lora_code_hashes(root) or p["gpu_requested"]
            or p["training_ready"] or p["full_weights_loaded"] or p["books_outcomes_accessed"]):
        raise ValueError("CPU native adapter scope/reload dependency must remain exact")
    for name, sha in p["artifact_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or file_sha256(run/relative) != sha:
            raise ValueError("Reviewed LoRA CPU artifact changed")
    if (p["roles"]["actor"]["trainable_parameters"] != 64929792
            or p["roles"]["critic"]["trainable_parameters"] != 64932353):
        raise ValueError("Official adapter/scalar-head scope differs from reviewed meta receipt")
    return p


def joint_code_hashes(root):
    names = ("configs/cmirank/ppo_lora_joint_v1.json", "src/cmirank/lora_joint.py",
        "scripts/cmirank/37_prepare_lora_joint_cpu.py", "scripts/cmirank/38_native_lora_joint_gpu.py",
        "scripts/cmirank/33_smoke_actor_optimizer_gpu.py", "scripts/cmirank/run_real_policy_smoke.sh",
        "scripts/cmirank/run_real_policy_with_keeper.sh", "src/cmirank/reserved_handoff.py",
        "src/cmirank/gpu_resources.py", "src/cmirank/ppo_parity.py")
    return {**lora_code_hashes(root), **{n: file_sha256(root/n) for n in names}}


def joint_gpu_budget(roles, total_bytes, workspace_bytes):
    if set(roles) != {"actor", "critic"} or total_bytes <= 0 or workspace_bytes <= 0:
        raise ValueError("Both roles and actual card/workspace budget required")
    # Both sets of gradients remain live for complete audits. Reference,
    # rollout, KV-cache and checkpoint staging are intentionally not loaded.
    states = sum(r["persistent_train_bytes"] for r in roles.values())
    return {"both_roles_params_grad_mom_bytes": states, "workspace_reserve_bytes": workspace_bytes,
        "physical_gpu_bytes": total_bytes, "within_estimated_budget": states+workspace_bytes <= total_bytes,
        "rollout_loaded": False, "reference_loaded": False, "full_ppo_fit_proven": False}


def verify_joint_preparation(private, root, c, path, commit):
    directory = private / "runs" / c["prepare_run_id"]
    p = json.loads((directory / "report.json").read_text())
    if (p["status"] != "LORA_JOINT_CPU20_PASS_GPU_GATE_REMAINS" or p["source_commit"] != commit
            or p["config_sha256"] != file_sha256(path) or p["code_sha256"] != joint_code_hashes(root)
            or p["samples_per_role"] != 20 or p["gpu_requested"] or p["full_weights_loaded"]
            or p["optimizer_updates"] or p["training_ready"]):
        raise ValueError("Exact-source joint CPU20 preparation required before handoff")
    for name, sha in p["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(directory/name) != sha:
            raise ValueError("Joint CPU20 artifacts changed")
    reviewed_lora_parent(private, root, c)
    return p


def native_model_config(private, c, compat, prepared, role):
    from verl.workers.config.model import HFModelConfig
    if role not in ("actor", "critic"):
        raise ValueError("Only native actor/critic LoRA model schemas")
    extra = {"lora_rank": c["rank"], "lora_alpha": c["alpha"],
        "target_modules": sorted(prepared["targets"][role])}
    if role == "critic":
        extra["lora_adapter_path"] = str(private/"runs"/c["cpu_parent_run_id"]/"critic-initial-adapter")
    return HFModelConfig(path=str(private/"runs"/compat["sft_run_id"]/"checkpoint"),
        load_tokenizer=False, use_remove_padding=False, use_fused_kernels=False,
        enable_gradient_checkpointing=True,
        model_type="language_model" if role == "actor" else "value_model", **extra)


def native_engine_config():
    from verl.workers.config.engine import FSDPEngineConfig
    return FSDPEngineConfig(strategy="fsdp2", model_dtype="fp32", dtype="bfloat16", forward_only=False,
        use_torch_compile=False, use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
        use_remove_padding=False, param_offload=False, optimizer_offload=False, offload_policy=False)


def native_optimizer_config(c, role):
    from verl.workers.config.optimizer import FSDPOptimizerConfig
    return FSDPOptimizerConfig(lr=c[f"{role}_lr"], total_training_steps=1,
        override_optimizer_config={"foreach": False})
