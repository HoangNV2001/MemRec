import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src.cmirank.lora_rollout import load_lora_rollout_config, lora_server_options, export_adapter_cpu
from src.cmirank.corrected_rollout import load_corrected_config
from src.cmirank.lora_checkpoint import load_checkpoint_config, tensor_digest

ROOT = Path(__file__).resolve().parents[1]


def test_lora_sampling_keeps_prior_full_support_correction_and_approved_train_scope():
    c, _ = load_lora_rollout_config(ROOT)
    sampling, _ = load_corrected_config(ROOT)
    lora, _ = load_checkpoint_config(ROOT)
    for k in ("temperature", "top_p", "top_k", "logprobs_mode", "rollout_is", "rollout_is_threshold",
              "rollout_is_batch_normalize", "rollout_rs", "bypass_mode", "gpu_memory_utilization", "max_new_tokens"):
        assert c[k] == sampling[k]
    for k in ("rank", "alpha", "model_dtype", "forward_dtype", "actor_lr", "critic_lr", "gpu_index"):
        assert c[k] == lora[k]
    assert c["physical_request_cap"] == 60 and c["optimizer_updates"] == 0
    assert not c["training_ready"] and not c["books_outcomes_accessed"]
    assert len(c["native_checkpoint_sha256"]) == 6


@pytest.mark.parametrize("key,value", [("physical_request_cap", 61), ("rank", 64), ("gpu_index", 0),
    ("temperature", 0), ("optimizer_updates", 1), ("checkpoint_parent_report_sha256", "0"*64),
    ("served_adapter_name", "base"), ("schema_version", True)])
def test_lora_rollout_contract_is_not_a_tuning_or_resource_override(tmp_path, key, value):
    folder = tmp_path/"configs/cmirank"
    folder.mkdir(parents=True)
    for source in (ROOT/"configs/cmirank").glob("*.json"):
        (folder/source.name).write_bytes(source.read_bytes())
    path = folder/"ppo_lora_rollout_v1.json"
    c = json.loads(path.read_text())
    c[key] = value
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_lora_rollout_config(tmp_path)


def test_standard_lora_server_does_not_enable_runtime_dynamic_loading_or_merge():
    c, _ = load_lora_rollout_config(ROOT)
    options = lora_server_options(c, Path("/private/actor-adapter"))
    assert options[-1] == c["served_adapter_name"]+"=/private/actor-adapter"
    assert options[options.index("--tensor-parallel-size")+1] == "1"
    assert options[options.index("--max-lora-rank")+1] == "32"
    assert options[options.index("--max-loras")+1] == "1"
    assert "--enforce-eager" in options and "--no-enable-prefix-caching" in options
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in str(options)


@pytest.mark.parametrize("corrupt", [False, True])
def test_cpu_export_checks_every_native_adapter_byte_without_loading_base(tmp_path, monkeypatch, corrupt):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    state = {f"base.model.language_model.projection{i}.lora_A.default.weight": torch.arange(4.).reshape(2, 2)+i
        for i in range(496)}
    scope = {"trainable_shapes": {n: list(p.shape) for n, p in state.items()}, "trainable_parameters": 496*4}
    audit = {"train_state_before": {n: {"parameter": tensor_digest(p)} for n, p in state.items()}}
    path = tmp_path/"native.pt"
    if corrupt:
        next(reversed(state.values()))[-1, -1] += 1
    torch.save(state, path)
    output = tmp_path/"adapter"
    def peft_state(model, state_dict, save_embedding_layers):
        assert save_embedding_layers is False
        return {k.replace(".default", ""): v for k, v in state_dict.items()}
    def save(directory, state_dict, safe_serialization, save_embedding_layers):
        assert safe_serialization and not save_embedding_layers
        directory.mkdir()
        safe.save_file(peft_state(None, state_dict, False), str(directory/"adapter_model.safetensors"))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(get_peft_model_state_dict=peft_state))
    model = SimpleNamespace(save_pretrained=save)
    if corrupt:
        with pytest.raises(ValueError, match="complete native update audit"):
            export_adapter_cpu(path, model, scope, audit, output)
    else:
        r = export_adapter_cpu(path, model, scope, audit, output)
        assert r["all_tensor_bytes_equal"] and r["exported_tensors"] == 496
        assert not r["base_weights_exported"] and not r["cuda_initialized"]


def test_cpu_and_coordinator_do_not_import_cuda_eagerly_or_patch_native_modules():
    for name in ("src/cmirank/lora_rollout.py", "scripts/cmirank/39_prepare_lora_rollout_cpu.py"):
        source = (ROOT/name).read_text()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith(("torch", "peft", "vllm", "transformers")) for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not node.module.startswith(("torch", "peft", "vllm", "transformers"))
        assert "scancel" not in source and "monkeypatch" not in source
    source = (ROOT/"scripts/cmirank/40_native_lora_rollout_gpu.py").read_text()
    assert 'engine.load_checkpoint(' in source and 'engine.disable_adapter()' in source
    assert 'engine.forward_step(' in source and 'world_size=1' in source
    assert 'train_batch(' not in source and 'optimizer.step(' not in source
    coordinator = (ROOT/"scripts/cmirank/29_smoke_corrected_rollout_gpu.py").read_text()
    assert coordinator.index('stop_owned(server, str(checkpoint))') < coordinator.index('worker = subprocess.Popen')
    assert 'gpu_index=1, protect_other_workloads=True' in coordinator
    for name in ("run_real_policy_smoke.sh", "run_real_policy_with_keeper.sh"):
        assert 'lora_rollout' in (ROOT/"scripts/cmirank"/name).read_text()
