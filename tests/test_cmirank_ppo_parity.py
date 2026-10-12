import copy
import json
from pathlib import Path

import pytest

from src.cmirank.ppo_parity import (compare_logprobs, completion_request, load_parity_config,
                                    response_prompt_logprobs, server_options, verify_parity_preparation)
from src.cmirank.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def row():
    return {"user_id": -10001, "snapshot_id": "synthetic-format-holdout-v1",
            "input_ids": [11, 12, 13, 14, 248046], "prompt_tokens": 2}


def response(data):
    return {"choices": [{"prompt_token_ids": data["input_ids"], "finish_reason": "length",
        "prompt_logprobs": [None] + [{str(i): {"logprob": -0.5}} for i in data["input_ids"][1:]]}],
        "usage": {"prompt_tokens": len(data["input_ids"]), "completion_tokens": 1}}


def test_exact_raw_ids_including_eos_and_single_unscored_generation():
    config, _ = load_parity_config(ROOT)
    request = completion_request(row(), config)
    assert request["prompt"] == row()["input_ids"] and not request["add_special_tokens"]
    assert request["max_tokens"] == 1 and request["return_token_ids"] and request["prompt_logprobs"] == 0
    assert response_prompt_logprobs(response(row()), row()) == [-0.5, -0.5, -0.5]
    assert config["optimizer_updates"] == 0 and not config["ppo_worker_proven"]


@pytest.mark.parametrize("mutation", ["inserted_bos", "dropped_eos", "missing_logprob", "nan", "wrong_usage", "multiple_choices"])
def test_no_token_repair_or_prompt_truncation_can_hide_parity_error(mutation):
    data = response(row())
    if mutation == "inserted_bos": data["choices"][0]["prompt_token_ids"] = [0] + row()["input_ids"]
    if mutation == "dropped_eos": data["choices"][0]["prompt_token_ids"] = row()["input_ids"][:-1]
    if mutation == "missing_logprob": data["choices"][0]["prompt_logprobs"][-1] = None
    if mutation == "nan": data["choices"][0]["prompt_logprobs"][-1]["248046"]["logprob"] = float("nan")
    if mutation == "wrong_usage": data["usage"]["completion_tokens"] = 2
    if mutation == "multiple_choices": data["choices"].append(copy.deepcopy(data["choices"][0]))
    with pytest.raises(ValueError): response_prompt_logprobs(data, row())


@pytest.mark.parametrize("mutation", ["books", "seen_train", "empty_response", "too_long"])
def test_numerics_never_uses_real_outcomes_or_truncated_inputs(mutation):
    config, _ = load_parity_config(ROOT)
    data = row()
    if mutation == "books": data["user_id"] = 1
    if mutation == "seen_train": data["snapshot_id"] = "synthetic-format-train-v1"
    if mutation == "empty_response": data["prompt_tokens"] = 5
    if mutation == "too_long": data["input_ids"] *= 1000
    with pytest.raises(ValueError): completion_request(data, config)


def test_likelihood_drift_is_checked_per_token_and_whole_action_before_outcomes():
    config, _ = load_parity_config(ROOT)
    assert compare_logprobs([-1.0]*6, [-1.001]*6, config)["passed"]
    assert not compare_logprobs([-1.0]*6, [-1.02]*6, config)["passed"]  # token error small, action drift >10%.
    with pytest.raises(ValueError): compare_logprobs([-1.0], [], config)


def test_extension_token_must_fit_without_implicit_context_truncation():
    config, _ = load_parity_config(ROOT)
    data = {**row(), "input_ids": [11] * config["max_model_len"]}
    with pytest.raises(ValueError): completion_request(data, config)


def test_exact_source_artifact_receipt_required_for_parity_gpu(tmp_path):
    config, _ = load_parity_config(ROOT)
    rows = [{**row(), "user_id": -10001-i} for i in range(20)]
    tokens = tmp_path / "tokens.jsonl"
    tokens.write_text("".join(json.dumps(r)+"\n" for r in rows))
    report = {"status": "PPO_PARITY_CPU_PREPARATION_PASS_GPU_NUMERICS_REMAIN", "source_commit": "source",
        "config_sha256": "config", "code_sha256": {"code": "sha"}, "gpu_requested": False,
        "native_text_only_registry_inputs_supported": False,
        "native_gdn_prefill_backend": "triton", "native_server_environment_checked": config["server_environment"],
        "native_kernel_warmup_skip_no_model_cpu_probe": True,
        "optimizer_updates": 0, "samples": 20, "artifact_sha256": {"tokens.jsonl": file_sha256(tokens)}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    kwargs = dict(commit="source", config_sha="config", code_sha={"code": "sha"})
    assert len(verify_parity_preparation(tmp_path, config, **kwargs)[0]) == 20
    with pytest.raises(ValueError): verify_parity_preparation(tmp_path, config, **{**kwargs, "commit": "wrong"})


def test_cpu_coordinator_never_holds_hf_weights_while_starting_service():
    source = (ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py").read_text()
    assert 'import torch' not in source
    assert source.index('worker.wait(timeout=600)') < source.index('server = subprocess.Popen')
    assert source.index('"hf-release.json"') < source.index('server = subprocess.Popen')
    assert 'gpu_index=gpu_index, protect_other_workloads=True' in source
    assert source.index('stop_owned(server, str(checkpoint))\n        passed') < source.index('save(run / "report.json"')
    assert '--no-enable-prefix-caching' in server_options(load_parity_config(ROOT)[0])
    assert '--enforce-eager' in server_options(load_parity_config(ROOT)[0])
    assert 'AdamW(' not in source and 'optimizer.step' not in source


def test_text_only_retry_inherits_all_tokens_weights_and_numerical_thresholds():
    original, _ = load_parity_config(ROOT, 1)
    retry, _ = load_parity_config(ROOT, 2)
    changes = {k for k in original if original[k] != retry[k]}
    assert changes == {"schema_version", "prepare_run_id", "run_id"}
    assert "--language-model-only" in server_options(retry)
    assert "--language-model-only" not in server_options(original)


def test_text_only_retry_rejects_unreviewed_threshold_change(tmp_path):
    directory = tmp_path / "configs/cmirank"
    directory.mkdir(parents=True)
    for version in (1, 2):
        (directory / f"ppo_parity_v{version}.json").write_bytes((ROOT / f"configs/cmirank/ppo_parity_v{version}.json").read_bytes())
    path = directory / "ppo_parity_v2.json"
    retry = json.loads(path.read_text())
    retry["max_token_absolute_logprob_delta"] = 10
    path.write_text(json.dumps(retry))
    with pytest.raises(ValueError): load_parity_config(tmp_path, 2)


def test_cpu_preparation_checks_real_native_multimodal_registry_branch_not_only_cli():
    source = (ROOT / "scripts/cmirank/22_prepare_ppo_parity_cpu.py").read_text()
    assert "MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config)" in source
    assert "model_config = engine_args.create_model_config()" in source
    assert "from_pretrained(" not in source


def test_gpu1_backend_retry_preserves_the_probability_and_weight_contract():
    previous, _ = load_parity_config(ROOT, 2)
    current, _ = load_parity_config(ROOT, 3)
    assert {k for k in previous if previous[k] != current[k]} == {
        "schema_version", "inherits", "prepare_run_id", "run_id", "gpu_index", "resource_policy", "reason"}
    assert current["gpu_index"] == 1 and current["gdn_prefill_backend"] == "triton"
    assert server_options(current)[-4:-2] == ["--gdn-prefill-backend", "triton"]
    assert current["server_kernel_config"] == {"enable_flashinfer_autotune": False}
    assert current["server_environment"] == {"VLLM_USE_DEEP_GEMM": "0", "VLLM_DEEP_GEMM_WARMUP": "skip"}
