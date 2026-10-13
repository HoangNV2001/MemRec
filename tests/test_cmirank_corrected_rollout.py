from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

from src.cmirank.corrected_rollout import (corrected_server_options, generated_observation,
    generated_request, load_corrected_config)
from src.cmirank.ppo_parity import load_parity_config

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_corrected_config(ROOT)[0]


def row():
    return {"user_id": -10001, "input_ids": [2, 3, 4, 5, 6], "prompt_tokens": 3}


def response(ids=None, probs=None, finish="stop"):
    return {"choices": [{"text": "<answer>C00</answer>", "token_ids": ids or [7, 248046],
        "logprobs": {"token_logprobs": probs or [-0.1, -0.2]}, "finish_reason": finish}]}


def test_separate_amendment_keeps_failed_primary_default():
    c = config()
    assert c["temperature"] == 0.9 and c["physical_request_cap"] == 20
    assert not c["bypass_mode"] and c["rollout_rs"] is None and not c["training_ready"]
    primary, _ = load_parity_config(ROOT)
    assert primary["schema_version"] == 8 and primary["temperature"] == 0
    assert primary["max_token_absolute_logprob_delta"] == 0.05


@pytest.mark.parametrize("key,value", [("gpu_index", 0), ("temperature", 1.0), ("top_p", 0.9),
    ("logprobs_mode", "raw_logprobs"), ("rollout_is_threshold", 3.0), ("rollout_rs", "sequence"),
    ("physical_request_cap", 21), ("optimizer_updates", 1), ("training_ready", True),
    ("forward_only", False), ("model_dtype", "float32"), ("use_fused_kernels", True)])
def test_no_tuned_or_expanded_config(tmp_path, key, value):
    folder = tmp_path / "configs/cmirank"
    folder.mkdir(parents=True)
    c = json.loads((ROOT / "configs/cmirank/ppo_corrected_rollout_v1.json").read_text())
    c[key] = value
    (folder / "ppo_corrected_rollout_v1.json").write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_corrected_config(tmp_path)


def test_prompt_only_full_support_sampled_logprobs():
    c = config()
    payload = generated_request(row(), c)
    assert payload["prompt"] == [2, 3, 4] and payload["temperature"] == 0.9
    assert "prompt_logprobs" not in payload and payload["logprobs"] == 0
    assert payload["top_p"] == 1 and payload["top_k"] == -1
    assert not payload["ignore_eos"] and not payload["add_special_tokens"]
    assert corrected_server_options(c)[-2:] == ["--logprobs-mode", "processed_logprobs"]


def test_keeps_sampled_eos_without_repair():
    obs = generated_observation(response(), row(), config(), 248046)
    assert obs["input_ids"] == [2, 3, 4, 7, 248046]
    assert obs["eos_observed"] and len(obs["behavior_logprobs"]) == 2


@pytest.mark.parametrize("change", ["missing_ids", "missing_lp", "nonfinite", "dropped_eos",
    "internal_eos", "positive_lp", "aborted", "wrong_length"])
def test_bad_physical_accounting_fails_closed(change):
    r = deepcopy(response())
    ch = r["choices"][0]
    if change == "missing_ids": ch.pop("token_ids")
    if change == "missing_lp": ch["logprobs"]["token_logprobs"].pop()
    if change == "nonfinite": ch["logprobs"]["token_logprobs"][0] = float("nan")
    if change == "dropped_eos": ch["token_ids"][-1] = 9
    if change == "internal_eos": ch["token_ids"] = [248046, 248046]
    if change == "positive_lp": ch["logprobs"]["token_logprobs"][0] = 0.1
    if change == "aborted": ch["finish_reason"] = "abort"
    if change == "wrong_length": ch["finish_reason"] = "length"
    with pytest.raises(ValueError):
        generated_observation(r, row(), config(), 248046)


def test_truncation_is_retained_not_given_synthetic_eos():
    obs = generated_observation(response([7]*64, [-0.2]*64, "length"), row(), config(), 248046)
    assert not obs["eos_observed"] and len(obs["response_token_ids"]) == 64


def test_coordinator_has_no_torch_and_releases_before_scoring():
    source = (ROOT / "scripts/cmirank/29_smoke_corrected_rollout_gpu.py").read_text()
    assert "import torch" not in source
    assert source.index('stop_owned(server, str(checkpoint))') < source.index('worker = subprocess.Popen')
    for name in ("run_real_policy_with_keeper.sh", "run_real_policy_smoke.sh"):
        text = (ROOT / "scripts/cmirank" / name).read_text()
        assert "corrected_rollout" in text and "scancel" not in text


def test_import_coordinator_does_not_load_training_frameworks():
    module = importlib.import_module("scripts.cmirank.29_smoke_corrected_rollout_gpu")
    assert callable(module.main) and callable(module.stop_owned)


def test_absolute_helper_loading_ignores_installed_scripts_package(monkeypatch):
    import sys
    from types import ModuleType
    from importlib.util import module_from_spec, spec_from_file_location
    monkeypatch.setitem(sys.modules, "scripts", ModuleType("scripts"))
    spec = spec_from_file_location("cpu_test_owned_coordinator", ROOT / "scripts/cmirank/29_smoke_corrected_rollout_gpu.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.stop_owned)
    assert Path(module._spec.origin) == ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py"


def test_v2_is_source_integration_only():
    a, _ = load_corrected_config(ROOT, 1)
    b, _ = load_corrected_config(ROOT, 2)
    allowed = {"schema_version", "inherits", "prepare_run_id", "run_id", "review_run_id", "reason"}
    assert all(a[k] == b[k] for k in a if k not in allowed)
    assert b["schema_version"] == 2 and b["run_id"] != a["run_id"]
