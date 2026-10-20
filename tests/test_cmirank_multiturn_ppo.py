from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from src.cmirank.multiturn_ppo import (load_multiturn_config, sample_trajectories,
    validate_episodes, episode_gae, native_training_batch)
from src.cmirank.format_sft import synthetic_request

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
    assert source.index('stop_owned(server, str(checkpoint))') < source.index('validate_episodes(observations, episodes, rows)')
    assert 'worker_report["optimizer_updates"] != 10' in source


def test_padding_retry_preserves_method_and_resource_contract():
    old, _ = load_multiturn_config(ROOT, version=1)
    new, _ = load_multiturn_config(ROOT)
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
