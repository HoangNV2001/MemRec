from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from src.cmirank.multiturn_ppo import (load_multiturn_config, sample_trajectories,
    validate_episodes, episode_gae, native_training_batch, episode_groups, minibatch_slices,
    audit_update_gradients, audit_gradient_cleanup)
from src.cmirank.format_sft import synthetic_request
from src.cmirank.rewards import malformed_mpss_rewards

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    eos_token_id = 248046

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return [10, 11] + list(text.encode())


def fixtures(malformed=False):
    c, _ = load_multiturn_config(ROOT)
    requests = [synthetic_request(c["data_seed"], "holdout", i).to_dict() for i in range(20)]
    journal, calls = [], []
    def call(payload, user, turn):
        calls.append(payload)
        # Deterministic fixture removes labels in ascending order; not a
        # synthetic ranking oracle or the real GPU sampling implementation.
        text = "bad format" if malformed else f"<answer>C{turn:02d}</answer>"
        n = 3 + turn % 3
        ids = [100+i for i in range(n-1)] + [248046]
        return {"choices": [{"text": text, "token_ids": ids, "finish_reason": "stop",
            "logprobs": {"token_logprobs": [-2.0]*n}}]}
    obs, episodes = sample_trajectories(requests, Tokenizer(), c, call, lambda name, row: journal.append((name, row)))
    return c, requests, obs, episodes, calls


def test_all20_nine_turns_have_strict_continuity_and_target_free_prompts():
    c, requests, obs, episodes, calls = fixtures()
    validate_episodes(obs, episodes, requests)
    assert len(calls) == len(obs) == 180
    assert len(episodes) == 20 and all(e["valid"] for e in episodes)
    assert len({tuple(p["prompt"]) for p in calls[:9]}) == 9
    for row in requests:
        assert "target" not in row and row["user_id"] < 0
    for payload in calls:
        assert payload["model"] == c["served_adapter_name"] and payload["temperature"] == 0.9
        assert "response_format" not in payload and payload["top_k"] == -1


def test_malformed_is_retained_for_all_users_without_retry_or_training_subset():
    _, requests, obs, episodes, calls = fixtures(malformed=True)
    assert len(calls) == len(obs) == 20 and len(episodes) == 20
    assert all(not e["valid"] for e in episodes)
    with pytest.raises(ValueError, match="All20"):
        validate_episodes(obs, episodes, requests)


@pytest.mark.parametrize("field,value", [("turn", 1), ("active_labels", ["C01"]),
    ("user_id", 123), ("finish_reason", "length"), ("rank_request_sha256", "0"*64)])
def test_no_turn_reorder_id_or_mask_repair(field, value):
    _, requests, obs, episodes, _ = fixtures()
    obs[0][field] = value
    with pytest.raises(ValueError):
        validate_episodes(obs, episodes, requests)


@pytest.mark.parametrize("key,value", [("gpu_index", 0), ("gamma", 0.99), ("episodes", 21),
    ("optimizer_updates", 0), ("schema_version", True), ("kl_coef", 0.0)])
def test_fixed_multiturn_contract_disallows_hidden_tuning(tmp_path, key, value):
    folder = tmp_path/"configs/cmirank"
    folder.mkdir(parents=True)
    for src in (ROOT/"configs/cmirank").glob("*.json"):
        (folder/src.name).write_bytes(src.read_bytes())
    path = folder/"ppo_lora_multiturn_v1.json"
    c = json.loads(path.read_text()); c[key] = value
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_multiturn_config(tmp_path)


def test_episode_adapter_really_crosses_nine_response_boundaries(monkeypatch):
    torch = pytest.importorskip("torch")
    c, _, obs, episodes, _ = fixtures()
    calls = []
    # ABI double for local pure-adapter tests only. The remote CPU gate calls
    # the installed, hash-recorded unmodified native VeRL GAE implementation.
    module = ModuleType("verl.trainer.ppo.core_algos")
    def gae(reward, value, mask, gamma, lam):
        calls.append((reward.clone(), value.clone(), mask.clone(), gamma, lam))
        returns = torch.flip(torch.cumsum(torch.flip(reward, [-1]), -1), [-1])
        return (returns-value)*mask, returns*mask
    module.compute_gae_advantage_return = gae
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.core_algos", module)
    values = [[0.1*(i+j) for j in range(len(r["response_token_ids"]))] for i, r in enumerate(obs)]
    adv, ret, audits = episode_gae(obs, episodes, values, c, torch)
    assert len(calls) == 1 and calls[0][0].shape[0] == 20
    assert len(adv) == len(ret) == 180
    assert {a["target_rank"] for a in audits} == set(range(1, 11))
    for i, audit in enumerate(audits):
        assert abs(audit["episode_return"]-audit["expected_ndcg5"]) < 1e-12
        assert abs(float(ret[i*9][0])-audit["expected_ndcg5"]) < 1e-6
    # An incorrect per-turn terminal adapter would miss final correction.
    assert float(ret[0][0]) == pytest.approx(0.0)
    assert audits[0]["mpss_rewards"][-1] <= 0


def test_shared_wrappers_keep_original_guarded_handoff_and_restore():
    for name in ("run_real_policy_smoke.sh", "run_real_policy_with_keeper.sh"):
        source = (ROOT/"scripts/cmirank"/name).read_text()
        assert '"$MEMREC_TASK_KIND" == lora_multiturn' in source
        assert "scancel" not in source and "trap" in source
    source = (ROOT/"scripts/cmirank/29_smoke_corrected_rollout_gpu.py").read_text()
    assert source.index('stop_owned(server, str(checkpoint))') < source.index('validate_episodes(observations, episodes, rows,')
    assert 'worker_report["optimizer_updates"] != expected_updates' in source


def test_padding_retry_preserves_method_and_resource_contract():
    old, _ = load_multiturn_config(ROOT, version=1)
    new, _ = load_multiturn_config(ROOT, version=2)
    changes = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    assert changes == {"schema_version", "prepare_run_id", "run_id", "reason", "inherits"}
    assert old["data_seed"] == new["data_seed"]


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_jagged_response_padding_is_dynamic_per_microbatch(monkeypatch, role):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    c, _, obs, _, _ = fixtures()
    # Minimal TensorDict metadata double only; real Torch jagged tensors.
    # CPU41 independently tests the installed VeRL loss/padding implementation.
    class TensorDict(dict):
        def __init__(self, fields, batch_size):
            super().__init__(fields)
            self.batch_size = batch_size
            self.metadata = {}
    monkeypatch.setitem(sys.modules, "tensordict", SimpleNamespace(TensorDict=TensorDict))
    tu = SimpleNamespace(assign_non_tensor=lambda data, **kwargs: data.metadata.update(kwargs))
    monkeypatch.setitem(sys.modules, "verl.utils", SimpleNamespace(tensordict_utils=tu))
    def correction(delta, mask, **kwargs):
        return torch.exp(delta.sum(-1)).clamp(max=2)[:, None].expand_as(delta), {}
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.rollout_corr_helper",
        SimpleNamespace(compute_rollout_correction_weights=correction))
    rows = obs[:3]
    ns = [len(r["response_token_ids"]) for r in rows]
    assert len(set(ns)) > 1
    scores = [{"old_logprobs": [-2.0]*n, "reference_logprobs": [-2.1]*n,
        "critic_values": [0.1]*n} for n in ns]
    advantages = [torch.ones(n) for n in ns]
    returns = [torch.ones(n)*0.5 for n in ns]
    batch = native_training_batch(rows, scores, advantages, returns, c, role, torch)
    assert batch.metadata["max_response_len"] == -1
    for i, n in enumerate(ns):
        assert batch["responses"][i].numel() == batch["response_mask"][i].numel() == n
        for field in (("old_log_probs", "ref_log_prob", "advantages", "rollout_is_weights")
                if role == "actor" else ("values", "returns")):
            assert batch[field][i].numel() == n


@pytest.mark.parametrize("prefix_length", range(9))
@pytest.mark.parametrize("target_index", range(10))
def test_every_malformed_prefix_has_exact_minus_one_without_target_credit(prefix_length, target_index):
    labels = [f"C{i:02d}" for i in range(10)]
    prefix = labels[:prefix_length]
    rewards = malformed_mpss_rewards(labels, prefix, labels[target_index])
    assert len(rewards) == prefix_length+1
    assert sum(rewards) == pytest.approx(-1, abs=1e-12)
    assert rewards[-1] <= -1


def mixed_fixtures():
    c, requests, obs, episodes, _ = fixtures()
    observed, terminal = [], []
    for i, episode in enumerate(episodes):
        if i % 2:
            observed.extend(deepcopy(obs[i*9:(i+1)*9])); terminal.append(deepcopy(episode))
        else:
            bad = i % 9
            group = deepcopy(obs[i*9:i*9+bad+1])
            group[-1].update(raw_text="<answer>C99</answer>", format_valid=False)
            observed.extend(group)
            terminal.append({**episode, "exclusions": episode["exclusions"][:bad],
                "valid": False, "survivor": None})
    return c, requests, observed, terminal


def test_malformed_branch_retains_all20_no_retry_fill_filter_or_quality_rescue():
    c, requests, obs, episodes = mixed_fixtures()
    validate_episodes(obs, episodes, requests, allow_malformed=True)
    assert len(episodes) == 20 and sum(e["valid"] for e in episodes) == 10
    groups = episode_groups(obs, episodes)
    assert sum(map(len, groups)) == len(obs) < 180
    assert all(g[-1]["format_valid"] == e["valid"] for g, e in zip(groups, episodes))
    with pytest.raises(ValueError):
        validate_episodes(obs, episodes, requests)
    slices = minibatch_slices(len(obs), c)
    assert [i for sl in slices for i in range(sl.start, sl.stop)] == list(range(len(obs)))
    assert all(sl.stop-sl.start <= 36 for sl in slices)


@pytest.mark.parametrize("mutation", ["lost_invalid_response", "invented_survivor", "resume_after_invalid", "hide_invalid_flag"])
def test_malformed_branch_still_fails_closed_on_corrupted_trajectories(mutation):
    _, requests, obs, episodes = mixed_fixtures()
    if mutation == "lost_invalid_response":
        obs.pop(0)
    elif mutation == "invented_survivor":
        episodes[0]["survivor"] = "C00"
    elif mutation == "resume_after_invalid":
        obs.insert(1, deepcopy(obs[0]))
    else:
        obs[0]["format_valid"] = True
    with pytest.raises(ValueError):
        validate_episodes(obs, episodes, requests, allow_malformed=True)


def test_mixed_native_gae_adapter_keeps_invalid_response_and_minus_one_terminal(monkeypatch):
    torch = pytest.importorskip("torch")
    c, requests, obs, episodes = mixed_fixtures()
    validate_episodes(obs, episodes, requests, allow_malformed=True)
    module = ModuleType("verl.trainer.ppo.core_algos")
    def gae(reward, value, mask, gamma, lam):
        returns = torch.flip(torch.cumsum(torch.flip(reward, [-1]), -1), [-1])
        return (returns-value)*mask, returns*mask
    module.compute_gae_advantage_return = gae
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.core_algos", module)
    adv, ret, audits = episode_gae(obs, episodes, [[0.2]*len(o["response_token_ids"]) for o in obs], c, torch)
    assert len(adv) == len(ret) == len(obs)
    offset = 0
    for group, audit in zip(episode_groups(obs, episodes), audits):
        expected = audit["expected_ndcg5"] if audit["valid"] else -1.0
        assert audit["episode_return"] == pytest.approx(expected)
        assert float(ret[offset][0]) == pytest.approx(expected, abs=1e-6)
        if not audit["valid"]:
            assert audit["expected_ndcg5"] == 0 and audit["target_rank"] is None
        offset += len(group)


@pytest.mark.parametrize("key,value", [("invalid_format_penalty", -0.1), ("train_invalid_trajectories", False),
    ("invalid_terminal_metric", 1.0), ("max_optimizer_updates", 12)])
def test_invalid_branch_does_not_tune_penalty_or_hide_failure(tmp_path, key, value):
    folder = tmp_path/"configs/cmirank"; folder.mkdir(parents=True)
    for src in (ROOT/"configs/cmirank").glob("*.json"):
        (folder/src.name).write_bytes(src.read_bytes())
    path = folder/"ppo_lora_multiturn_v3.json"
    c = json.loads(path.read_text()); c[key] = value; path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_multiturn_config(tmp_path)


def test_lifecycle_retry_preserves_all_method_and_resource_values():
    old, _ = load_multiturn_config(ROOT, version=3)
    new, path = load_multiturn_config(ROOT)
    changes = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    assert changes == {"schema_version", "prepare_run_id", "run_id", "reason", "inherits"}
    assert path.name == "ppo_lora_multiturn_v4.json"


def tiny_updated_engine():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    model = torch.nn.Module()
    model.register_parameter("adapter", torch.nn.Parameter(torch.ones(3, 3)))
    model.register_parameter("frozen", torch.nn.Parameter(torch.ones(3), requires_grad=False))
    optimizer = torch.optim.AdamW([model.adapter], lr=1e-5, foreach=False)
    model.adapter.square().sum().backward(); optimizer.step()
    return torch, SimpleNamespace(module=model, optimizer=optimizer, mode="train"), {
        "trainable_parameters": 9, "gradient_bytes": 36, "adam_moment_bytes": 72}


def test_gradient_audit_before_cleanup_then_release_after_exit():
    torch, engine, scope = tiny_updated_engine()
    receipt = audit_update_gradients(engine, scope, 1, torch)
    assert receipt["gradient_bytes"] == 36 and receipt["adam_moment_bytes"] == 72
    assert receipt["audit_inside_native_train_context"]
    with pytest.raises(ValueError, match="context must have exited"):
        audit_gradient_cleanup(engine)
    engine.optimizer.zero_grad(); engine.mode = None
    audit_gradient_cleanup(engine)
    with pytest.raises(ValueError, match="inside the native train context"):
        audit_update_gradients(engine, scope, 1, torch)


@pytest.mark.parametrize("corrupt", ["missing_grad", "nan_grad", "wrong_step", "nan_moment", "bf16_moment", "frozen_state"])
def test_lifecycle_fix_does_not_weaken_gradient_state_guards(corrupt):
    torch, engine, scope = tiny_updated_engine()
    p = engine.module.adapter
    if corrupt == "missing_grad": p.grad = None
    elif corrupt == "nan_grad": p.grad[0, 0] = float("nan")
    elif corrupt == "wrong_step": engine.optimizer.state[p]["step"] += 1
    elif corrupt == "nan_moment": engine.optimizer.state[p]["exp_avg"][0, 0] = float("nan")
    elif corrupt == "bf16_moment": engine.optimizer.state[p]["exp_avg"] = engine.optimizer.state[p]["exp_avg"].bfloat16()
    else: engine.optimizer.state[engine.module.frozen] = {"step": 1}
    with pytest.raises(ValueError):
        audit_update_gradients(engine, scope, 1, torch)


def test_gpu_gradient_audit_is_inside_native_context_without_disabling_cleanup():
    import ast
    source = (ROOT/"scripts/cmirank/42_native_multiturn_ppo_gpu.py").read_text()
    contexts = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.With)
        and any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "train_mode" for item in n.items)]
    assert len(contexts) == 1
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "audit_update_gradients"
        for n in ast.walk(contexts[0]))
    assert "zero_grad_on_exit=False" not in source
    assert "audit_gradient_cleanup(engine)" in source
