"""Researcher-approved one-GPU LoRA qualification; no eager torch/CUDA import.

The language backbone stays frozen. The native scalar critic head remains
trainable through PEFT's modules_to_save, never a head-only critic fallback.
"""

import json
from math import prod
from pathlib import Path

from .provenance import file_sha256


def load_lora_config(root, version=3):
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("Unknown LoRA qualification profile")
    path = root / "configs/cmirank/ppo_lora_v1.json"
    c = json.loads(path.read_text())
    fixed = {"schema_version": 1, "prepare_run_id": "cmirank-lora-native-cpu-v1-20261015-hnv",
        "actor_run_id": "cmirank-lora-actor-smoke-v1-20261015-hnv",
        "critic_run_id": "cmirank-lora-critic-smoke-v1-20261015-hnv",
        "parameterization": "lora", "rank": 32, "alpha": 32, "bias": "none", "dropout": 0.0,
        "target_scope": "all_linear_modules_under_model.language_model", "critic_task_type": None,
        "critic_modules_to_save": ["score"], "actor_lr": 1e-5, "critic_lr": 2e-5,
        "model_dtype": "fp32", "forward_dtype": "bfloat16", "samples": 20, "optimizer_updates": 1,
        "micro_batch_size": 1, "optimizer": "torch.optim.AdamW", "foreach": False, "seed": 42,
        "gpu_index": 1, "tensor_parallel_size": 1, "cpu_timeout_seconds": 600,
        "timeout_minutes": 25, "worker_timeout_seconds": 1200,
        "host_load_reserve_bytes": 20*2**30, "workspace_reserve_bytes": 8*2**30,
        "physical_requests": 0, "update_discarded": True, "training_ready": False,
        "books_outcomes_accessed": False, "ranking_metrics_computed": False, "quantization": False}
    if c != fixed or any(type(c[k]) is not type(v) for k, v in fixed.items()):
        raise ValueError("Only the preregistered one-GPU20 LoRA qualification profile is allowed")
    for revision in range(2, version+1):
        path = root / f"configs/cmirank/ppo_lora_v{revision}.json"
        retry = json.loads(path.read_text())
        fixed_retry = {"schema_version": revision, "inherits": f"configs/cmirank/ppo_lora_v{revision-1}.json",
            "prepare_run_id": f"cmirank-lora-native-cpu-v{revision}-20261015-hnv",
            "actor_run_id": f"cmirank-lora-actor-smoke-v{revision}-20261015-hnv",
            "critic_run_id": f"cmirank-lora-critic-smoke-v{revision}-20261015-hnv"}
        if (set(retry) != set(fixed_retry) | {"reason"}
                or any(type(retry[k]) is not type(v) or retry[k] != v for k, v in fixed_retry.items())
                or not isinstance(retry["reason"], str) or not retry["reason"]):
            raise ValueError("CPU fidelity retry cannot change scientific/resource fields")
        c.update(retry)
    return c, path


def lora_code_hashes(root):
    names = ("configs/cmirank/ppo_lora_v1.json", "configs/cmirank/ppo_lora_v2.json",
        "configs/cmirank/ppo_lora_v3.json", "src/cmirank/lora.py",
        "scripts/cmirank/36_prepare_lora_cpu.py", "src/cmirank/optimizer_smoke.py",
        "src/cmirank/native_critic.py", "src/cmirank/ppo_compat.py")
    return {n: file_sha256(root / n) for n in names}


def adapter_parameter_count(shapes, rank):
    if type(rank) is not int or rank <= 0 or not shapes:
        raise ValueError("Positive rank and nonempty language linear targets required")
    total = 0
    for name, shape in shapes.items():
        if (not name.startswith("model.language_model.") or len(shape) != 2
                or any(type(n) is not int or n <= 0 for n in shape)):
            raise ValueError("LoRA must cover language linear modules only, never vision/head")
        total += rank * sum(shape)
    return total


def linear_targets(model):
    import torch
    result = {n: list(m.weight.shape) for n, m in model.named_modules()
              if n.startswith("model.language_model.") and isinstance(m, torch.nn.Linear)}
    adapter_parameter_count(result, 32)
    return result


def audit_scope(model, shapes, c, role):
    if role not in ("actor", "critic"):
        raise ValueError("Only actor/critic LoRA scope")
    groups = {"adapter": 0, "head": 0, "frozen": 0}
    trainable_shapes = {}
    covered = {"A": set(), "B": set()}
    parameter_bytes = 0
    for name, p in model.named_parameters():
        count = p.numel()
        parameter_bytes += count*p.element_size()
        if p.dtype.is_floating_point is False or p.element_size() != 4:
            raise ValueError(f"Frozen base and trainable storage must preserve FP32: {name} has {p.dtype}")
        if not p.requires_grad:
            groups["frozen"] += count
            continue
        if ".visual." in name:
            raise ValueError("LoRA gate preserves FP32 trainable state and frozen vision")
        if ".lora_A.default." in name or ".lora_B.default." in name:
            if ".model.language_model." not in name:
                raise ValueError("Adapter outside the language backbone")
            side = "A" if ".lora_A.default." in name else "B"
            prefix = name.split(f".lora_{side}.default.", 1)[0]
            if not prefix.startswith("base_model.model."):
                raise ValueError("Unexpected PEFT parameter namespace")
            target = prefix[len("base_model.model."):]
            expected = ([c["rank"], shapes[target][1]] if side == "A"
                        else [shapes[target][0], c["rank"]]) if target in shapes else None
            if list(p.shape) != expected:
                raise ValueError("Adapter target/shape differs from the official linear scope")
            covered[side].add(target)
            groups["adapter"] += count
        elif role == "critic" and ".score.modules_to_save.default." in name:
            groups["head"] += count
        else:
            raise ValueError(f"Unexpected trainable backbone/head parameter: {name}")
        trainable_shapes[name] = list(p.shape)
    expected_head = model.config.text_config.hidden_size + 1 if role == "critic" else 0
    if (any(names != set(shapes) for names in covered.values())
            or groups["adapter"] != adapter_parameter_count(shapes, c["rank"]) or groups["head"] != expected_head):
        raise ValueError("Incomplete adapter coverage or missing scalar critic head")
    trainable = groups["adapter"] + groups["head"]
    if trainable >= groups["frozen"] or sum(prod(s) for s in trainable_shapes.values()) != trainable:
        raise ValueError("LoRA must not silently enable full backbone training")
    return {"groups": groups, "trainable_parameters": trainable, "trainable_shapes": trainable_shapes,
        "parameter_bytes": parameter_bytes, "gradient_bytes": 4*trainable,
        "adam_moment_bytes": 8*trainable, "persistent_train_bytes": parameter_bytes + 12*trainable,
        "role": role, "joint_fit_proven": False}


def make_critic_adapter(model, shapes, c, directory, *, materialize_meta=False):
    """Standard generic PEFT adapter preserves token-classification forward.

    Native VeRL can load a pretrained PEFT adapter. task_type=None deliberately
    avoids its fresh-adapter CAUSAL_LM wrapper for a value-only model; no model
    or engine is patched. Only adapter/head tensors are saved, never base weights.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.other import ModulesToSaveWrapper
    config = LoraConfig(r=c["rank"], lora_alpha=c["alpha"], lora_dropout=c["dropout"],
        target_modules=sorted(shapes), bias=c["bias"], task_type=None, modules_to_save=["score"])
    wrapped = get_peft_model(model, config)
    if materialize_meta:
        # Meta holds the official full backbone shapes only. Initialize just
        # adapters via PEFT and the head via native HF initialization, on CPU.
        for module in wrapped.modules():
            if isinstance(module, LoraLayer):
                module.lora_A["default"].to_empty(device="cpu")
                module.lora_B["default"].to_empty(device="cpu")
                module.reset_lora_parameters("default", init_lora_weights=True)
            elif isinstance(module, ModulesToSaveWrapper):
                head = module.modules_to_save["default"]
                head.to_empty(device="cpu")
                model._init_weights(head)
    scope = audit_scope(wrapped, shapes, c, "critic")
    if any(p.is_meta for p in wrapped.parameters() if p.requires_grad):
        raise ValueError("Unmaterialized adapter/head cannot be exported")
    wrapped.save_pretrained(str(directory), safe_serialization=True, save_embedding_layers=False)
    saved = json.loads((Path(directory) / "adapter_config.json").read_text())
    if saved["task_type"] is not None or saved["modules_to_save"] != ["score"]:
        raise ValueError("Critic adapter must retain generic PEFT semantics and the value head")
    return wrapped, scope
