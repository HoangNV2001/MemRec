import copy
import json
from pathlib import Path

import pytest

from src.cmirank.ppo_parity import (compare_logprobs, completion_request, load_parity_config,
                                    response_prompt_logprobs, server_options, verify_parity_preparation,
                                    wait_owned_hf_release, summarize_repeatability)
from src.cmirank.gpu_resources import GPUCard
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
    assert config["physical_request_cap"] == 20  # New candidate must remeasure all twenty cases.


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
    config, _ = load_parity_config(ROOT, 3)
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
    service_stop = source.index('stop_owned(server, str(checkpoint))', source.index('for boot in range(config.get("server_boots", 1))'))
    assert service_stop < source.index('passed = all(') < source.rindex('save(run / "report.json"')
    assert service_stop < source.index('summarize_repeatability(observations, rows, config)')
    assert source.index('"hf-release.json"') < source.index('save(run / "report.json"')
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


def test_precision_diagnosis_never_loosens_bounds_or_selects_primary_backend():
    original, _ = load_parity_config(ROOT, 3)
    diagnostic, _ = load_parity_config(ROOT, 4)
    for field in ("max_token_absolute_logprob_delta", "max_mean_absolute_logprob_delta", "max_action_absolute_logratio"):
        assert original[field] == diagnostic[field]
    assert diagnostic["physical_request_cap"] == 0 and diagnostic["samples"] == 20
    assert len(diagnostic["diagnostic_variants"]) == 5
    source = (ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py").read_text()
    assert "PRECISION_DIAGNOSIS_COMPLETE_ORIGINAL_PARITY_GATE_REMAINS_FAILED" in source
    assert '"primary_profile_selected": False' in source
    assert "baseline_delta != 0" in source


def test_diagnostic_parent_recomputes_exact_twenty_frozen_failed_probes():
    # Fixture data are local/ignored; synthetic-only reanalysis is optional in CI.
    from src.cmirank.ppo_parity import diagnostic_parent
    directory = ROOT / "results/cmirank-ppo-vllm-hf-parity-v3-20261012-hnv"
    if not directory.is_dir(): pytest.skip("Local diagnostic predecessor not present")
    config, _ = load_parity_config(ROOT, 4)
    rows = [json.loads(s) for s in (ROOT / "results/cmirank-ppo-parity-data-v3-20261012-hnv/tokens.jsonl").read_text().splitlines()]
    pairs = diagnostic_parent(directory, config, rows)
    assert len(pairs) == 20
    assert sum(compare_logprobs(p["hf"], p["vllm"], config)["passed"] for p in pairs) == 17


def test_native_fla_candidate_changes_only_hf_kernel_dependency_not_weight_or_bounds():
    old, _ = load_parity_config(ROOT, 3)
    new, _ = load_parity_config(ROOT, 5)
    assert {k for k in old if old[k] != new[k]} == {"schema_version", "inherits", "prepare_run_id", "run_id", "reason"}
    assert new["native_gdn_gradient_samples"] == 20
    assert not new.get("hf_precision_diagnostic") and new["physical_request_cap"] == 20
    source = (ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py").read_text()
    assert source.index("verify_fla_overlay(private, config)") < source.index("= handoff(run,")
    assert 'env=hf_environment' in source and 'env=server_environment' in source
    worker = (ROOT / "scripts/cmirank/24_hf_parity_worker_gpu.py").read_text()
    assert worker.index("native_gdn_gradient_probes(config, run, torch)") < worker.index("model, info =")


def test_private_compiler_retry_preserves_fla_probes_and_every_scientific_field():
    old, _ = load_parity_config(ROOT, 5)
    new, _ = load_parity_config(ROOT, 6)
    assert {k for k in old if old[k] != new[k]} == {"schema_version", "inherits", "prepare_run_id", "run_id", "reason"}
    assert new["hf_worker_environment"] == {"FLA_TILELANG": "1"}
    source = (ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py").read_text()
    assert source.index("verify_cuda_toolchain(private, ROOT, config)") < source.index("= handoff(run,")
    assert 'hf_environment.update(CUDA_HOME=' in source
    assert 'hf_environment.update(config["hf_worker_environment"])' in source
    worker = (ROOT / "scripts/cmirank/24_hf_parity_worker_gpu.py").read_text()
    assert '"common:chunk_bwd_dqkwg:tilelang" not in registry._logged' in worker


def test_full_fastpath_completes_native_dependency_without_tuning_the_parity_recipe():
    old, _ = load_parity_config(ROOT, 6)
    new, _ = load_parity_config(ROOT, 7)
    assert {k for k in old if old[k] != new[k]} == {"schema_version", "inherits", "prepare_run_id", "run_id", "reason"}
    assert new["native_causal_gradient_samples"] == 20
    assert new["toolchain_prepare_source_commit"] != "current_consumer_source"
    source = (ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py").read_text()
    assert source.index("verify_causal_overlay(private, ROOT, config)") < source.index("= handoff(run,")
    worker = (ROOT / "scripts/cmirank/24_hf_parity_worker_gpu.py").read_text()
    assert worker.index("native_causal_gradient_probes(config, run, torch)") < worker.index("model, info =")
    assert 'not hf.is_fast_path_available' in worker


def test_causal_build_is_private_no_deps_and_bounded_owned_cpu_process_group():
    source = (ROOT / "scripts/cmirank/27_build_causal_conv_cpu.py").read_text()
    assert '"--no-index", "--no-build-isolation", "--no-deps"' in source
    assert '"--target", str(overlay)' in source and 'CAUSAL_CONV1D_FORCE_BUILD' in source
    assert 'start_new_session=True' in source and 'os.killpg(child.pid, signal.SIGTERM)' in source
    assert 'hf.causal_conv1d_fn is None' in source  # Respect CPU availability guard.
    assert 'source_hashes' in source and 'Build patched immutable official source' in source


def test_release_retry_preserves_every_scientific_field_and_idle_threshold():
    old, _ = load_parity_config(ROOT, 7)
    new, _ = load_parity_config(ROOT, 8)
    assert {k for k in old if old[k] != new[k]} == {"schema_version", "inherits", "prepare_run_id", "run_id", "reason"}
    assert new["hf_release_wait_seconds"] == 15
    source = (ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py").read_text()
    assert source.index("wait_owned_hf_release(gpu_snapshot") < source.index("server = subprocess.Popen")


def test_owned_release_waits_for_trailing_utilization_without_relaxing_guard():
    c, _ = load_parity_config(ROOT)
    t = [0]
    records = []
    selected = GPUCard(1,"GPU-approved","H100",80,1,81559)
    protected = GPUCard(0,"GPU-protected","H100",100,10000,81559)
    def snapshot():
        card = selected if not t[0] else GPUCard(1,"GPU-approved","H100",0,1,81559)
        return [protected,card], {"GPU-protected":{42}}, "raw-cards", "raw-apps"
    def sleep(dt): t[0] += dt
    report = wait_owned_hf_release(snapshot,c,"GPU-approved",child_exited=True,
                                  emit=records.append,clock=lambda:t[0],sleep=sleep)
    assert report["attempts"] == 2 and report["elapsed_seconds"] == 1
    assert len(records) == 2 and report["gpu_released"] and report["gpu_uuid"] == "GPU-approved"


@pytest.mark.parametrize("util,used,pids", [(20,1,set()),(0,512,set()),(0,1,{99})])
def test_owned_release_never_loads_when_original_boundary_or_pid_guard_stays_busy(util,used,pids):
    c, _ = load_parity_config(ROOT)
    t = [0]
    records = []
    cards = [GPUCard(0,"GPU-other","H100",0,1,81559),GPUCard(1,"GPU-approved","H100",util,used,81559)]
    def sleep(dt): t[0] += dt
    with pytest.raises(ValueError,match="original idle guards"):
        wait_owned_hf_release(lambda:(cards,{"GPU-approved":pids} if pids else {},"cards","apps"),
            c,"GPU-approved",child_exited=True,emit=records.append,clock=lambda:t[0],sleep=sleep)
    assert t[0] == 15 and len(records) == 16


def test_owned_release_does_not_wait_on_unexited_child_or_changed_physical_card():
    c, _ = load_parity_config(ROOT)
    def snapshot():
        return [GPUCard(1,"GPU-changed","H100",0,1,81559)],{},"cards","apps"
    with pytest.raises(ValueError,match="confirmed owned-child exit"):
        wait_owned_hf_release(snapshot,c,"GPU-approved",child_exited=False,emit=lambda row:None)
    with pytest.raises(ValueError,match="Physical GPU changed"):
        wait_owned_hf_release(snapshot,c,"GPU-approved",child_exited=True,emit=lambda row:None)


def repeat_fixture():
    rows = [{**row(),"user_id":-10001-i} for i in range(20)]
    vectors = [{"boot":b,"replay":r,"user_id":u["user_id"],"logprobs":[-1.0,-2.0,-3.0]}
               for b in range(2) for r in range(2) for u in rows]
    return rows,vectors


def test_repeat_diagnosis_never_changes_primary_default_or_selects_best_replay(monkeypatch):
    monkeypatch.delenv("MEMREC_PARITY_CONTRACT_VERSION",raising=False)
    primary,_ = load_parity_config(ROOT)
    diagnostic,_ = load_parity_config(ROOT,9)
    assert primary["physical_request_cap"] == 20 and not primary.get("repeatability_diagnostic")
    assert diagnostic["physical_request_cap"] == 80 and diagnostic["samples"] == 20
    for name in ("max_token_absolute_logprob_delta","max_mean_absolute_logprob_delta","max_action_absolute_logratio"):
        assert diagnostic[name] == primary[name]
    monkeypatch.setenv("MEMREC_PARITY_CONTRACT_VERSION","9")
    assert load_parity_config(ROOT)[0]["repeatability_diagnostic"]
    source = (ROOT/"scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py").read_text()
    assert 'REPEATABILITY_DIAGNOSIS_COMPLETE_PRIMARY_PARITY_GATE_REMAINS_FAILED' in source
    assert '"primary_profile_selected":False' in source


def test_fixed_repeat_pairs_distinguish_within_server_from_cross_start_drift():
    c,_ = load_parity_config(ROOT,9)
    rows,vectors = repeat_fixture()
    for vector in vectors:
        if vector["boot"]==1:
            vector["logprobs"][1] += .1
    summary,pairs = summarize_repeatability(vectors,rows,c)
    assert len(pairs)==80
    assert summary["within_boot"]["exact_equal_pairs"]==40
    assert summary["across_boot"]["original_bound_pass_pairs"]==0
    assert summary["across_boot"]["max_token_absolute_logprob_delta"] > .05


@pytest.mark.parametrize("mutation",["omit","duplicate","wrong_round","dropped_eos","nan"])
def test_repeat_diagnosis_rejects_incomplete_repaired_or_nonfinite_vectors(mutation):
    c,_ = load_parity_config(ROOT,9)
    rows,vectors = repeat_fixture()
    if mutation=="omit": vectors.pop()
    if mutation=="duplicate": vectors[-1]=copy.deepcopy(vectors[0])
    if mutation=="wrong_round": vectors[-1]["replay"]=2
    if mutation=="dropped_eos": vectors[-1]["logprobs"].pop()
    if mutation=="nan": vectors[-1]["logprobs"][0]=float("nan")
    with pytest.raises(ValueError): summarize_repeatability(vectors,rows,c)
