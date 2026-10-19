"""Sequential one-card LoRA sampler and native reference/resume qualification.

Standard PyTorch CPU mmap + PEFT export from a hash-locked discarded native
checkpoint, no merged base checkpoint, parameter-name rewrite or model patch.
"""

import hashlib
import json
from pathlib import Path
import subprocess

from .corrected_rollout import load_corrected_config, corrected_server_options
from .lora_checkpoint import load_checkpoint_config, checkpoint_code_hashes, tensor_digest
from .provenance import file_sha256


def load_lora_rollout_config(root):
    native, _ = load_corrected_config(root)
    base, _ = load_checkpoint_config(root)
    c = {**native, **base}
    path = root/"configs/cmirank/ppo_lora_rollout_v1.json"
    r = json.loads(path.read_text())
    fixed = {"schema_version": 1,
        "prepare_run_id": "cmirank-lora-rollout-data-v1-20261019-hnv",
        "run_id": "cmirank-lora-rollout-smoke-v1-20261019-hnv",
        "review_run_id": "cmirank-lora-rollout-review-v1-20261019-hnv",
        "checkpoint_parent_source_commit": "9e7f8ee3c5b6bdd294ca7c2372b54d51390bba35",
        "checkpoint_parent_report_sha256": "95daed7a66c1d748ba8e0e960526c98a3232fd2fb0fc998841ae27e31183822d",
        "checkpoint_parent_prepare_report_sha256": "4b67fe8fd2f0204af3bdc8537d2559f5301df55a363a29aaecb4f27ea4f4d1f0",
        "reference_control_vectors_sha256": "dea51b33b7f219133accb0a053b8bca28c50531ea2f55d9777f7927e670473b1",
        "native_checkpoint_sha256": {
            "actor/model_world_size_1_rank_0.pt": "1da301b27baf3851fe47b190fe6aa8974638a8b55ba3b07fc3741b032566afcd",
            "actor/optim_world_size_1_rank_0.pt": "aff26e1c3a844c2311a3418753b89aabb7397d8c21af55aebd959622974425a2",
            "actor/extra_state_world_size_1_rank_0.pt": "31de6d2f67a8b0369660992e41d40d0ef062ffc8a3bd3bb6dba3639db42623bc",
            "critic/model_world_size_1_rank_0.pt": "6fcb30736af4f5b542315cb02a3afbb2b182fccbcea3cd7ecd260594328048f2",
            "critic/optim_world_size_1_rank_0.pt": "e47f487ebfdfa2c9cf61593b5c3df2a2e3cd6ca6128be93bd4888afdc139f8b7",
            "critic/extra_state_world_size_1_rank_0.pt": "c35ec1d4cee228172d6f74dea48b73b59d3980261c369a2f526a9279fa0dc5ad"},
        "parameterization": "lora_rollout", "optimizer_updates": 0, "updates_per_role": 0,
        "physical_request_cap": 60, "served_adapter_name": "cmirank-lora-update1-hnv", "port": 18126,
        "worker_status": "NATIVE_LORA_COLD_RESTORE_REFERENCE_SCORE_PASS_CPU_REVIEW_REMAINS"}
    if r != fixed or any(type(r[k]) is not type(v) for k, v in fixed.items()):
        raise ValueError("Only the fixed20 sequential LoRA sampler/reference gate is allowed")
    c.update(r)
    c["checkpoint_parent_run_id"] = base["run_id"]
    c["checkpoint_parent_prepare_run_id"] = base["prepare_run_id"]
    return c, path


def rollout_code_hashes(root):
    names = ("configs/cmirank/ppo_lora_rollout_v1.json", "src/cmirank/lora_rollout.py",
        "scripts/cmirank/39_prepare_lora_rollout_cpu.py", "scripts/cmirank/40_native_lora_rollout_gpu.py",
        "scripts/cmirank/29_smoke_corrected_rollout_gpu.py", "scripts/cmirank/31_review_corrected_rollout_cpu.py",
        "src/cmirank/corrected_rollout.py")
    return {**checkpoint_code_hashes(root), **{n: file_sha256(root/n) for n in names}}


def checkpoint_directory(private, c):
    return private/"runs"/c["checkpoint_parent_run_id"]/"native-checkpoints"


def reviewed_checkpoint_parent(private, root, c):
    run = private/"runs"/c["checkpoint_parent_run_id"]
    prep = private/"runs"/c["checkpoint_parent_prepare_run_id"]
    if (file_sha256(run/"report.json") != c["checkpoint_parent_report_sha256"]
            or file_sha256(prep/"report.json") != c["checkpoint_parent_prepare_report_sha256"]):
        raise ValueError("Reviewed native checkpoint dependency changed")
    r = json.loads((run/"report.json").read_text())
    p = json.loads((prep/"report.json").read_text())
    if (r["source_commit"] != c["checkpoint_parent_source_commit"] or p["source_commit"] != r["source_commit"]
            or r["status"] != "NATIVE_LORA_CHECKPOINT_PASS_ROLLOUT_MULTITURN_GATES_REMAIN"
            or r["samples"] != 20 or r["optimizer_updates"] != 6 or r["training_ready"]
            or not r["native_checkpoint_roundtrip_proven"] or r["books_outcomes_accessed"]
            or file_sha256(run/"native-worker-report.json") != r["worker_report_sha256"]):
        raise ValueError("Completed isolated checkpoint gate required")
    for name, sha in p["code_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe producing-source path")
        if hashlib.sha256(subprocess.check_output(["git", "show", r["source_commit"]+":"+name], cwd=root)).hexdigest() != sha:
            raise ValueError("Checkpoint parent differs from its producing source")
    for name, sha in p["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(prep/name) != sha:
            raise ValueError("Checkpoint parent CPU artifacts changed")
    for role in ("actor", "critic"):
        a = run/f"checkpoint-{role}-audit.json"
        if file_sha256(a) != r["roles"][role]["checkpoint"]["audit_sha256"]:
            raise ValueError("Complete role-state/readout audit changed")
    cleanup = json.loads((run/"cleanup.json").read_text())
    handback = json.loads((run/"keeper-handback.json").read_text())
    if (cleanup["process_exit_code"] or not cleanup["gpu_released"] or not cleanup["allocation_still_running"]
            or handback["status"] != "KEEPER_GPU1_HANDBACK_VERIFIED"
            or handback["gpu0_signalled"] or handback["reserved_job_cancelled"]):
        raise ValueError("Reviewed checkpoint cleanup and keeper1 restoration required")
    for name, sha in c["native_checkpoint_sha256"].items():
        if file_sha256(checkpoint_directory(private, c)/name) != sha:
            raise ValueError("Native saved model/Adam/RNG bytes changed")
    controls = private/"runs"/"cmirank-lora-joint-optimizer-smoke-v1-20261015-hnv"/"native-vectors.jsonl"
    if file_sha256(controls) != c["reference_control_vectors_sha256"]:
        raise ValueError("Frozen initial native reference control vectors changed")
    return r, p


def lora_server_options(c, adapter):
    return [*corrected_server_options(c), "--enable-lora", "--max-lora-rank", str(c["rank"]),
        "--max-loras", "1", "--max-cpu-loras", "1", "--lora-modules", f"{c['served_adapter_name']}={adapter}"]


def export_adapter_cpu(checkpoint, model, scope, audit, directory):
    """Ordinary trusted torch.load mmap; CPU-only. Fail if CUDA is needed."""
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    trainable = {}
    for name, shape in scope["trainable_shapes"].items():
        p = state[name]
        value = p.to_local() if hasattr(p, "to_local") else p
        if (value.device.type != "cpu" or value.dtype != torch.float32 or list(value.shape) != shape
                or tensor_digest(value) != audit["train_state_before"][name]["parameter"]):
            raise ValueError("CPU checkpoint adapter tensor differs from complete native update audit")
        trainable[name] = value.detach()
    if len(trainable) != 496 or any(".lora_" not in n for n in trainable):
        raise ValueError("Export all496 actor adapter tensors, never base/vision/head")
    model.save_pretrained(directory, state_dict=trainable, safe_serialization=True, save_embedding_layers=False)
    from safetensors.torch import load_file
    exported = load_file(str(directory/"adapter_model.safetensors"))
    from peft import get_peft_model_state_dict
    expected = get_peft_model_state_dict(model, state_dict=trainable, save_embedding_layers=False)
    if (set(exported) != set(expected) or sum(p.numel() for p in exported.values()) != scope["trainable_parameters"]
            or any(tensor_digest(exported[n]) != tensor_digest(p) for n, p in expected.items())):
        raise ValueError("Standard PEFT export changed adapter bytes or coverage")
    if torch.cuda.is_initialized():
        raise ValueError("CPU export initialized CUDA")
    return {"exported_tensors": len(exported), "exported_parameters": scope["trainable_parameters"],
        "all_tensor_bytes_equal": True, "base_weights_exported": False, "cuda_initialized": False}


def verify_lora_rollout_preparation(private, root, c, path, commit):
    run = private/"runs"/c["prepare_run_id"]
    p = json.loads((run/"report.json").read_text())
    if (p["status"] != "LORA_ROLLOUT_CPU20_EXPORT_PASS_GPU_GATE_REMAINS" or p["source_commit"] != commit
            or p["config_sha256"] != file_sha256(path) or p["code_sha256"] != rollout_code_hashes(root)
            or p["samples_per_role"] != 20 or p["gpu_requested"] or p["training_ready"] or p["optimizer_updates"]
            or not p["adapter_export"]["all_tensor_bytes_equal"] or p["adapter_export"]["cuda_initialized"]):
        raise ValueError("Fresh exact-source CPU20 export/sampler qualification required")
    for name, sha in p["artifact_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or file_sha256(run/relative) != sha:
            raise ValueError("Prepared adapter/token/request artifact changed")
    for name, expected in p["native_checkpoint_stat"].items():
        s = (checkpoint_directory(private, c)/name).stat()
        if {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino} != expected:
            raise ValueError("Native checkpoint changed after CPU hashing")
    rows = [json.loads(x) for x in (run/"tokens.jsonl").read_text().splitlines()]
    if len(rows) != 20 or len({r["user_id"] for r in rows}) != 20 or any(r["user_id"] >= 0 for r in rows):
        raise ValueError("All20 synthetic fixtures required, no Books examples")
    return rows, p
