import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_fla_candidate_is_pure_pinned_overlay_not_a_framework_upgrade():
    config = json.loads((ROOT / "configs/cmirank/ppo_fla_overlay_v1.json").read_text())
    assert config["samples"] == 20 and not config["gpu_requested"] and not config["full_weights_loaded"]
    assert not config["base_environment_mutation"] and not config["training_ready"]
    assert {w["name"] for w in config["wheels"]} == {"flash-linear-attention", "fla-core"}
    for wheel in config["wheels"]:
        assert wheel["version"] == "0.5.2" and wheel["filename"].endswith("py3-none-any.whl")
        assert len(wheel["sha256"]) == 64 and wheel["url"].startswith("https://files.pythonhosted.org/packages/")


def test_cpu_overlay_probes_no_cuda_no_weights_no_dependency_or_job_mutation():
    source = (ROOT / "scripts/cmirank/25_prepare_fla_overlay_cpu.py").read_text()
    assert '"--no-index"' in source and '"--no-deps"' in source and '"--target"' in source
    assert "range(20)" in source and "assert not torch.cuda.is_initialized()" in source
    assert "hf.chunk_gated_delta_rule is chunk_gated_delta_rule" in source
    assert "base_before !=" in source and "overlay.exists()" in source
    assert "scancel" not in source and "handoff(" not in source and "from_pretrained(" not in source
    assert "optimizer.step" not in source and "CUDA_HOME=" not in source
