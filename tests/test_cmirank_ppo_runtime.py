import copy
import json
from pathlib import Path

import pytest

from src.cmirank.ppo_runtime import locked_requirements, require_new_environment, response_reward_layout, verify_resolution
from src.cmirank.rewards import mpss_rewards, ndcg_at_k


def profile():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/cmirank/ppo_runtime_v1.json").read_text())


def resolution():
    cfg = profile()
    rows = []
    for name, version in cfg["versions"].items():
        wheel = cfg["wheels"].get(name, {"url": f"https://files.pythonhosted.org/{name}.whl", "sha256": "a" * 64})
        rows.append({"metadata": {"name": name, "version": version},
                     "download_info": {"url": wheel["url"], "archive_info": {"hashes": {"sha256": wheel["sha256"]}}}})
    rows.append({"metadata": {"name": "verl", "version": "0.9.0"},
                 "download_info": {"url": "https://github.com/verl-project/verl.git",
                     "vcs_info": {"vcs": "git", "commit_id": cfg["verl_commit"]}}})
    return {"install": rows}


def test_exact_cuda12_profile_and_artifact_lock():
    cfg = profile()
    assert not cfg["training_ready"] and not cfg["gpu_requested"] and cfg["cpu_samples"] == 20
    assert cfg["gamma"] == 1 and cfg["algorithm"] == "ppo_with_gae"
    assert verify_resolution(resolution(), cfg)["vllm"] == "0.20.0+cu129"
    lock = locked_requirements(resolution())
    assert len(lock) == 8 and all("#sha256=" in r or cfg["verl_commit"] in r for r in lock)


@pytest.mark.parametrize("name,version", [("nvidia-cuda-runtime-cu13", "13.0"), ("cuda-bindings", "13.0.1"),
    ("cuda-toolkit", "12.9"), ("torch", "2.11.0+cu130")])
def test_cuda13_or_implicit_toolkit_is_rejected(name, version):
    report = resolution()
    report["install"].append({"metadata": {"name": name, "version": version},
        "download_info": {"url": "https://files.pythonhosted.org/bad.whl"}})
    with pytest.raises(ValueError):
        verify_resolution(report, profile())


@pytest.mark.parametrize("change", ["version", "sha", "verl", "http", "missing"])
def test_changed_primary_stack_and_untrusted_receipts_fail(change):
    report = copy.deepcopy(resolution())
    if change == "version": report["install"][0]["metadata"]["version"] = "2.13.0"
    if change == "sha": report["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"] = "b" * 64
    if change == "verl": report["install"][-1]["download_info"]["vcs_info"]["commit_id"] = "f" * 40
    if change == "http": report["install"][0]["download_info"]["url"] = "http://invalid"
    if change == "missing": report["install"].pop()
    with pytest.raises(ValueError):
        verify_resolution(report, profile())


def test_reward_layout_all_target_positions_and_nine_actions():
    labels = [f"C{i}" for i in range(10)]
    lengths = [5 + j % 3 for j in range(9)]
    for target in labels:
        rewards, rank = mpss_rewards(labels, labels[:-1], labels[-1], target)
        placed, positions = response_reward_layout(lengths, list(rewards))
        assert len(placed) == sum(lengths) and len(positions) == 9
        assert [placed[p] for p in positions] == list(rewards)
        assert all(r == 0 for i, r in enumerate(placed) if i not in positions)
        assert sum(placed) == pytest.approx(ndcg_at_k(rank), abs=1e-12)
    with pytest.raises(ValueError): response_reward_layout(lengths[:-1], [0] * 9)
    with pytest.raises(ValueError): response_reward_layout([0] * 9, [0] * 9)


def test_new_environment_cannot_overwrite_existing_or_other_project(tmp_path):
    env = require_new_environment(tmp_path, "cmirank-ppo-cu129-v1-hnv")
    env.mkdir(parents=True)
    with pytest.raises(ValueError): require_new_environment(tmp_path, env.name)
    with pytest.raises(ValueError): require_new_environment(tmp_path, "llm-hnv")
    with pytest.raises(ValueError): require_new_environment(tmp_path, "../omnidistill")


def test_cpu_prep_has_no_cancellation_gpu_or_full_weights_path():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/cmirank/17_prepare_ppo_runtime_cpu.py").read_text()
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != ""' in text
    assert '"--dry-run"' in text and '"--no-deps"' in text and '"pip", "check"' in text
    assert "handoff(" not in text and "scancel" not in text and "torch.cuda" not in text
