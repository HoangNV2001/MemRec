from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

from src.cmirank.optimizer_smoke import (load_optimizer_config, memory_profile,
    TEXT_PARAMETERS, VISION_PARAMETERS, validate_observations)
from src.cmirank.ppo_parity import load_parity_config

ROOT = Path(__file__).resolve().parents[1]


def test_actor_only_native_profile_and_legacy_failure_contract_unchanged():
    c, _ = load_optimizer_config(ROOT)
    assert c["optimizer_updates"] == 1 and c["discard_update"] and not c["training_ready"]
    assert c["model_dtype"] == "fp32" and c["forward_dtype"] == "bfloat16"
    p = memory_profile()
    assert p["persistent_train_bytes"] == 16*TEXT_PARAMETERS + 4*VISION_PARAMETERS
    assert p["roles_resident"] == ["actor"] and not p["actor_critic_rollout_fit_proven"]
    assert p["persistent_train_bytes"] + c["workspace_reserve_bytes"] < 80*2**30
    old, _ = load_parity_config(ROOT)
    assert old["schema_version"] == 8 and old["max_token_absolute_logprob_delta"] == 0.05


@pytest.mark.parametrize("key,value", [("samples", 19), ("gpu_index", 0), ("optimizer_updates", 2),
    ("actor_lr", 2e-6), ("model_dtype", "bfloat16"), ("foreach", True), ("freeze_vision", False),
    ("training_ready", True), ("physical_requests", 20), ("optimizer_offload", True),
    ("co_residency_proven", True), ("full_checkpoint_roundtrip_proven", True), ("surprise", 1)])
def test_rejects_hidden_expansion_or_training_profile_tuning(tmp_path, key, value):
    c, _ = load_optimizer_config(ROOT, 1)
    c[key] = value
    p = tmp_path / "configs/cmirank"
    p.mkdir(parents=True)
    (p / "ppo_actor_optimizer_v1.json").write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_optimizer_config(tmp_path, 1)


def observations():
    return [{"user_id": -100-i, "input_ids": [1, 2, 3, 248046], "prompt_tokens": 2,
        "response_token_ids": [3, 248046], "behavior_logprobs": [-.1, -.2],
        "finish_reason": "stop", "eos_observed": True, "format_valid": True} for i in range(20)]


def likelihoods():
    return [{"user_id": -100-i, "old_logprobs": [-.1, -.2], "recomputed_logprobs": [-.1, -.2]} for i in range(20)]


def test_all_twenty_including_exact_response_eos():
    validate_observations(observations(), likelihoods(), load_optimizer_config(ROOT)[0])


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "order", "books", "eos", "prefix", "lp", "nonfinite", "repair"])
def test_rejects_observation_filtering_leakage_or_mask_mismatch(mutation):
    obs, native = deepcopy(observations()), likelihoods()
    if mutation == "drop": obs.pop()
    if mutation == "duplicate": obs[1] = obs[0]
    if mutation == "order": native.reverse()
    if mutation == "books": obs[0]["user_id"] = native[0]["user_id"] = 1
    if mutation == "eos": obs[0]["response_token_ids"][-1] = obs[0]["input_ids"][-1] = 3
    if mutation == "prefix": obs[0]["prompt_tokens"] = 3
    if mutation == "lp": obs[0]["behavior_logprobs"].pop()
    if mutation == "nonfinite": native[0]["old_logprobs"][0] = float("nan")
    if mutation == "repair": obs[0]["format_valid"] = False
    with pytest.raises(ValueError):
        validate_observations(obs, native, load_optimizer_config(ROOT)[0])


def test_coordinator_import_and_owned_resource_order():
    mod = importlib.import_module("scripts.cmirank.33_smoke_actor_optimizer_gpu")
    assert callable(mod.main) and callable(mod.stop_owned)
    source = (ROOT / "scripts/cmirank/33_smoke_actor_optimizer_gpu.py").read_text()
    assert "import torch" not in source
    assert source.index('verify_optimizer_preparation(private') < source.index('handoff(run') < source.index('worker = subprocess.Popen')
    worker = (ROOT / "scripts/cmirank/34_native_actor_optimizer_gpu.py").read_text()
    cpu = (ROOT / "scripts/cmirank/32_prepare_actor_optimizer_cpu.py").read_text()
    assert "train_micro_batch_size_per_gpu" not in worker + cpu
    assert "micro_batch_size_per_gpu=1" in worker and "micro_batch_size_per_gpu=1" in cpu
    assert "engine.train_batch(data, loss_function=partial(ppo_loss" in worker
    assert "engine.optimizer.step(" not in worker and "torch.optim.AdamW(" not in worker
    assert worker.index('p.requires_grad_(not visual)') < worker.index('with engine.eval_mode()')
    for name in ("run_real_policy_smoke.sh", "run_real_policy_with_keeper.sh"):
        assert "actor_optimizer" in (ROOT / "scripts/cmirank" / name).read_text()


def test_v2_only_changes_release_integration_namespaces():
    original, _ = load_optimizer_config(ROOT, 1)
    retry, _ = load_optimizer_config(ROOT, 2)
    permitted = {"schema_version", "inherits", "prepare_run_id", "run_id", "reason"}
    assert all(original[k] == retry[k] for k in original if k not in permitted)
    assert original["run_id"] != retry["run_id"] and retry["schema_version"] == 2


def test_release_kind_is_supported_and_owned_exit_still_required():
    from src.cmirank.gpu_resources import GPUCard
    from src.cmirank.ppo_parity import wait_owned_hf_release
    source = (ROOT / "scripts/cmirank/33_smoke_actor_optimizer_gpu.py").read_text()
    assert 'child_exited=worker.poll() == 0, child_kind="native_fsdp2"' in source
    config, _ = load_parity_config(ROOT, 8)
    snapshot = lambda: ([GPUCard(1, "GPU-tested", "H100", 0, 1, 81559)], {}, "cards", "apps")
    result = wait_owned_hf_release(snapshot, config, "GPU-tested", child_exited=True,
        child_kind="native_fsdp2", emit=lambda row: None)
    assert result["gpu_released"]
    with pytest.raises(ValueError):
        wait_owned_hf_release(snapshot, config, "GPU-tested", child_exited=False,
            child_kind="native_fsdp2", emit=lambda row: None)


@pytest.mark.parametrize("key,value", [("optimizer_updates", 2), ("gpu_index", 0),
    ("foreach", True), ("model_dtype", "bfloat16")])
def test_v2_rejects_profile_override(tmp_path, key, value):
    folder = tmp_path / "configs/cmirank"
    folder.mkdir(parents=True)
    (folder / "ppo_actor_optimizer_v1.json").write_bytes((ROOT / "configs/cmirank/ppo_actor_optimizer_v1.json").read_bytes())
    c = json.loads((ROOT / "configs/cmirank/ppo_actor_optimizer_v2.json").read_text())
    c[key] = value
    (folder / "ppo_actor_optimizer_v2.json").write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_optimizer_config(tmp_path)
