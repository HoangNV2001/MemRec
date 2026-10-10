import copy
import json
from pathlib import Path

import pytest

from src.cmirank.ppo_runtime import load_kernel_config, load_runtime_config, locked_requirements, require_new_environment, response_reward_layout, verify_resolution
from src.cmirank.rewards import mpss_rewards, ndcg_at_k


def profile():
    root = Path(__file__).resolve().parents[1]
    return load_runtime_config(root)[0]


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
    assert len(lock) == 11 and all("#sha256=" in r or cfg["verl_commit"] in r for r in lock)


def test_only_inspected_cuda12_runtime_metapackage_allowed_in_private_env():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = load_runtime_config(root, 1)
    with pytest.raises(ValueError, match="CUDA13/toolkit"):
        verify_resolution(resolution(), cfg)
    assert verify_resolution(resolution(), profile())["cuda-toolkit"] == "12.9.1"


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


def test_native_kernel_build_is_source_pinned_cpu_bounded_new_overlay():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/cmirank/ppo_kernel_build_v1.json").read_text())
    text = (root / "scripts/cmirank/19_build_ppo_kernel_cpu.py").read_text()
    assert config["cpus"] == 4 and config["max_jobs"] == 2 and config["nvcc_threads"] == 1
    assert config["cuda_arch"] == "90" and not config["gpu_requested"] and not config["training_ready"]
    assert '"--target", str(overlay)' in text and '"--no-deps"' in text
    assert 'overlay.exists()' in text and 'report["artifact_sha256"]' in text
    assert "scancel" not in text and "handoff(" not in text and ".cuda(" not in text


def test_kernel_retry_only_changes_walltime_and_owned_cache_not_training_scope():
    root = Path(__file__).resolve().parents[1]
    original, _ = load_kernel_config(root, 1)
    retry, _ = load_kernel_config(root, 2)
    for name in ("flash_attention_version", "flash_attention_commit", "cuda_home", "nvcc_release",
                 "cuda_arch", "max_jobs", "nvcc_threads", "cpus", "parent_report_sha256"):
        assert original[name] == retry[name]
    assert retry["compile_timeout_seconds"] == 5400 and retry["timeout_minutes"] == 95
    assert retry["overlay_name"] != original["overlay_name"] and not retry["gpu_requested"]


@pytest.mark.parametrize("mutation", [{"max_jobs": 4}, {"cuda_arch": "80"},
    {"resume_source_relative_path": "/usr/local/cuda"},
    {"resume_source_relative_path": "runs/../foreign"}])
def test_kernel_cache_retry_cannot_widen_scope(tmp_path, mutation):
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "configs/cmirank"
    directory.mkdir(parents=True)
    for version in (1, 2):
        path = f"configs/cmirank/ppo_kernel_build_v{version}.json"
        (tmp_path / path).write_bytes((root / path).read_bytes())
    path = directory / "ppo_kernel_build_v2.json"
    config = json.loads(path.read_text());config.update(mutation);path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="contract"):
        load_kernel_config(tmp_path)
