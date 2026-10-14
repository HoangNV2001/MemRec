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
    c, _ = load_optimizer_config(ROOT)
    c[key] = value
    p = tmp_path / "configs/cmirank"
    p.mkdir(parents=True)
    (p / "ppo_actor_optimizer_v1.json").write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_optimizer_config(tmp_path)


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
    assert "engine.train_batch(data, loss_function=partial(ppo_loss" in worker
    assert "engine.optimizer.step(" not in worker and "torch.optim.AdamW(" not in worker
    assert worker.index('p.requires_grad_(not visual)') < worker.index('with engine.eval_mode()')
    for name in ("run_real_policy_smoke.sh", "run_real_policy_with_keeper.sh"):
        assert "actor_optimizer" in (ROOT / "scripts/cmirank" / name).read_text()
