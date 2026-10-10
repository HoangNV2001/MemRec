import json
from pathlib import Path

import pytest

from src.cmirank.format_sft import encode_completion
from src.cmirank.ppo_compat import load_compat_config, validate_compat_rows, verify_preparation
from src.cmirank.provenance import file_sha256
from src.cmirank.native_critic import native_critic_contract, shape_scope
from tests.test_cmirank_format_sft import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def rows():
    return [{"user_id": -(i + 10001), "snapshot_id": "synthetic-format-holdout-v1",
             **encode_completion(Tokenizer(), f"Prompt{i}", f"<answer>C{i % 10:02d}</answer>",
                                 max_input=4096, max_output=128)} for i in range(20)]


def test_compat_gate_has_no_update_generation_or_quality_promotion():
    config, _ = load_compat_config(ROOT)
    assert config["samples"] == 20 and config["gpu_index"] == 0 and config["optimizer_updates"] == 0
    assert config["generation_calls"] == 0 and not config["training_ready"]
    validate_compat_rows(rows(), config)
    assert len(config["checkpoint_files_sha256"]) == 11
    assert len([name for name in config["checkpoint_files_sha256"] if name.endswith(".safetensors")]) == 5
    assert all(len(sha) == 64 for sha in config["checkpoint_files_sha256"].values())


@pytest.mark.parametrize("mutation", ["books", "training", "prompt_loss", "duplicate", "short"])
def test_only_fixed_twenty_outcome_free_holdout_rows_are_admissible(mutation):
    data = rows();config, _ = load_compat_config(ROOT)
    if mutation == "books": data[0]["user_id"] = 733
    if mutation == "training": data[0]["snapshot_id"] = "synthetic-format-train-v1"
    if mutation == "prompt_loss": data[0]["labels"][0] = data[0]["input_ids"][0]
    if mutation == "duplicate": data[0]["user_id"] = data[1]["user_id"]
    if mutation == "short": data.pop()
    with pytest.raises(ValueError): validate_compat_rows(data, config)


def test_source_hash_and_mask_receipt_required_before_gpu_handoff(tmp_path):
    config, _ = load_compat_config(ROOT, version=1)
    token_file = tmp_path / "tokens.jsonl"
    token_file.write_text("".join(json.dumps(r) + "\n" for r in rows()))
    report = {"status": "FULL_ROLE_COMPAT_CPU_PREPARATION_PASS_GPU_GATES_REMAIN",
        "source_commit": "source", "config_sha256": "config", "code_sha256": {"code": "sha"},
        "gpu_requested": False, "model_weights_loaded": False, "training_ready": False,
        "artifact_sha256": {"tokens.jsonl": file_sha256(token_file)}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    arguments = dict(commit="source", config_sha="config", code_sha={"code": "sha"}, config=config)
    assert len(verify_preparation(tmp_path, **arguments)[0]) == 20
    with pytest.raises(ValueError): verify_preparation(tmp_path, **{**arguments, "commit": "different"})
    token_file.write_text(token_file.read_text() + "\n")
    with pytest.raises(ValueError): verify_preparation(tmp_path, **arguments)


def test_task_binding_protection_and_native_loader_are_not_silent_new_optimizer():
    text = (ROOT / "scripts/cmirank/21_smoke_ppo_full_roles_gpu.py").read_text()
    assert text.index('handoff(run,') < text.index('        import torch')
    assert text.index('verified_numeric_cuda_binding') < text.index('        import torch')
    assert 'gpu_index=0, protect_other_workloads=True' in text
    assert 'load_valuehead_model(str(checkpoint)' in text
    assert "AdamW(" not in text and "optimizer.step" not in text and "model.generate" not in text
    assert '"optimizer_updates": 0' in text and '"ppo_worker_or_rollout_proven": False' in text
    controller = (ROOT / "scripts/cmirank/run_real_policy_with_keeper.sh").read_text()
    worker = (ROOT / "scripts/cmirank/run_real_policy_smoke.sh").read_text()
    assert '"$MEMREC_TASK_KIND" == ppo_compat' in controller and '"$MEMREC_TASK_KIND" == ppo_compat' in worker
    assert '--protect-other-workloads' in controller
    assert 'critic.score.state_dict()' in text and 'critic.v_head' not in text
    assert 'freeze_native_critic(critic' in text and 'native_critic_values(output, ids' in text


def native_shapes():
    actor = {"model.language_model.weight": [3570052096], "model.visual.weight": [333514240],
             "lm_head.weight": [248320, 2560]}
    critic = {n: s for n, s in actor.items() if n != "lm_head.weight"}
    critic.update({"score.weight": [1, 2560], "score.bias": [1]})
    return actor, critic


def test_native_scalar_critic_not_equal_actor_lm_count_but_same_backbone():
    actor, critic = native_shapes()
    contract = native_critic_contract(actor, critic, 2560, official=True)
    assert contract["scope"] == {"text_backbone_parameters": 3570052096,
                                "frozen_vision_parameters": 333514240, "scalar_head_parameters": 2561}
    assert shape_scope(actor)["text_backbone_parameters"] == 4205751296


@pytest.mark.parametrize("mutation", ["missing_backbone", "extra_layer", "backbone_shape", "vision_shape", "nonscalar_head", "missing_bias"])
def test_native_shape_adapter_does_not_weaken_backbone_or_head_guards(mutation):
    actor, critic = native_shapes()
    if mutation == "missing_backbone": critic.pop("model.language_model.weight")
    if mutation == "extra_layer": critic["rule_head.weight"] = [2560]
    if mutation == "backbone_shape": critic["model.language_model.weight"] = [10]
    if mutation == "vision_shape": critic["model.visual.weight"] = [10]
    if mutation == "nonscalar_head": critic["score.weight"] = [2, 2560]
    if mutation == "missing_bias": critic.pop("score.bias")
    with pytest.raises(ValueError): native_critic_contract(actor, critic, 2560, official=True)


def test_native_cpu_receipt_is_required_for_retry(tmp_path):
    config, _ = load_compat_config(ROOT)
    token_file = tmp_path / "tokens.jsonl"
    token_file.write_text("".join(json.dumps(r) + "\n" for r in rows()))
    report = {"status": "FULL_ROLE_COMPAT_CPU_PREPARATION_PASS_GPU_GATES_REMAIN",
        "source_commit": "source", "config_sha256": "config", "code_sha256": {"code": "sha"},
        "gpu_requested": False, "model_weights_loaded": False, "training_ready": False,
        "artifact_sha256": {"tokens.jsonl": file_sha256(token_file)}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Native critic CPU interface receipt"):
        verify_preparation(tmp_path, commit="source", config_sha="config", code_sha={"code": "sha"}, config=config)


def test_retry_preserves_all_numerical_research_and_resource_settings():
    base, _ = load_compat_config(ROOT, version=1)
    retry, _ = load_compat_config(ROOT)
    changed = {key for key in base if base[key] != retry[key]}
    assert changed == {"schema_version", "prepare_run_id", "run_id"}
    assert retry["cpu_native_critic_samples"] == 20
