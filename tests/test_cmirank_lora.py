import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cmirank.lora import load_lora_config, adapter_parameter_count, audit_scope
from src.cmirank.optimizer_smoke import load_optimizer_config

ROOT = Path(__file__).resolve().parents[1]


def test_lora_is_new_approved_single_card_protocol_not_relabelled_full_pass():
    c, _ = load_lora_config(ROOT)
    assert c["gpu_index"] == 1 and c["tensor_parallel_size"] == 1
    assert c["rank"] == c["alpha"] == 32 and not c["quantization"]
    assert c["critic_modules_to_save"] == ["score"] and c["critic_task_type"] is None
    assert c["model_dtype"] == "fp32" and c["forward_dtype"] == "bfloat16"
    assert not c["training_ready"] and not c["books_outcomes_accessed"]
    assert load_optimizer_config(ROOT)[0]["actor_lr"] == 1e-6


@pytest.mark.parametrize("key,value", [("rank", 64), ("rank", 32.0), ("gpu_index", 0),
    ("tensor_parallel_size", 2), ("critic_modules_to_save", []), ("critic_task_type", "CAUSAL_LM"),
    ("training_ready", True), ("model_dtype", "bfloat16"), ("quantization", True), ("actor_lr", 1e-6)])
def test_no_hidden_lora_capacity_or_resource_override(tmp_path, key, value):
    c = deepcopy(load_lora_config(ROOT)[0])
    c[key] = value
    folder = tmp_path / "configs/cmirank"
    folder.mkdir(parents=True)
    (folder / "ppo_lora_v1.json").write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_lora_config(tmp_path)


def test_parameter_math_excludes_base_and_unused_vision():
    shapes = {"model.language_model.layers.0.self_attn.q_proj": [256, 128],
              "model.language_model.layers.0.mlp.down_proj": [128, 512]}
    assert adapter_parameter_count(shapes, 32) == 32*(256+128+128+512)
    for bad in ({}, {"model.visual.projector": [32, 32]}, {"score": [1, 256]},
                {"model.language_model.bad": [0, 256]}):
        with pytest.raises(ValueError):
            adapter_parameter_count(bad, 32)


def test_cpu_module_import_has_no_torch_cuda_and_has_no_handoff():
    for filename in ("src/cmirank/lora.py", "scripts/cmirank/36_prepare_lora_cpu.py"):
        tree = ast.parse((ROOT / filename).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert all(not x.startswith(("torch", "transformers", "peft")) for x in names)
    cpu = (ROOT / "scripts/cmirank/36_prepare_lora_cpu.py").read_text()
    assert "handoff(" not in cpu and "scancel" not in cpu and "killpg" not in cpu
    assert 'if torch.cuda.is_initialized()' in cpu
    assert 'FSDPEngine._build_lora_module(proxy, module)' in cpu
    assert 'with torch.device("meta")' in cpu
    assert '"full_sequence_replayed": False' in cpu


def test_critic_is_standard_peft_not_custom_model_or_head_only():
    source = (ROOT / "src/cmirank/lora.py").read_text()
    assert 'task_type=None, modules_to_save=["score"]' in source
    assert 'module.reset_lora_parameters("default", init_lora_weights=True)' in source
    assert 'model._init_weights(head)' in source
    assert 'p.requires_grad_(True)' not in source
    assert 'save_embedding_layers=False' in source


def scope_fixture():
    class Parameter:
        def __init__(self, shape, trainable):
            self.shape = shape
            self.requires_grad = trainable
            self.dtype = SimpleNamespace(is_floating_point=True)

        def element_size(self):
            return 4

        def numel(self):
            from math import prod
            return prod(self.shape)
    target = "model.language_model.layers.0.self_attn.q_proj"
    params = [(f"base_model.model.{target}.lora_A.default.weight", Parameter([32, 2], True)),
        (f"base_model.model.{target}.lora_B.default.weight", Parameter([2, 32], True)),
        ("base_model.model.score.modules_to_save.default.weight", Parameter([1, 64], True)),
        ("base_model.model.score.modules_to_save.default.bias", Parameter([1], True)),
        ("base_model.model.model.language_model.embed_tokens.weight", Parameter([5000], False))]
    model = SimpleNamespace(named_parameters=lambda: params,
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=64)))
    return model, params, {target: [2, 2]}


def test_complete_lora_critic_counts_adapter_and_head_not_backbone():
    model, _, shapes = scope_fixture()
    report = audit_scope(model, shapes, load_lora_config(ROOT)[0], "critic")
    assert report["groups"] == {"adapter": 128, "head": 65, "frozen": 5000}
    assert report["gradient_bytes"] == 4*193
    assert report["adam_moment_bytes"] == 8*193
    assert not report["joint_fit_proven"]


@pytest.mark.parametrize("mutation", ["head_only", "backbone_unfrozen", "head_frozen", "wrong_target", "wrong_shape"])
def test_scope_rejects_head_only_and_unintended_trainables(mutation):
    model, params, shapes = scope_fixture()
    if mutation == "head_only":
        params[0][1].requires_grad = params[1][1].requires_grad = False
    elif mutation == "backbone_unfrozen":
        params[-1][1].requires_grad = True
    elif mutation == "head_frozen":
        params[2][1].requires_grad = False
    elif mutation == "wrong_target":
        params[0] = (params[0][0].replace("q_proj", "v_proj"), params[0][1])
    else:
        params[0][1].shape = [2, 32]
    with pytest.raises(ValueError):
        audit_scope(model, shapes, load_lora_config(ROOT)[0], "critic")
