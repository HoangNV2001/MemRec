import json
from pathlib import Path

import pytest

from src.cmirank.format_sft import encode_completion
from src.cmirank.ppo_compat import load_compat_config, validate_compat_rows, verify_preparation
from src.cmirank.provenance import file_sha256
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
    config, _ = load_compat_config(ROOT)
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
