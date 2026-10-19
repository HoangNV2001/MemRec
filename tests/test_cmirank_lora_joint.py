import importlib
import json
from pathlib import Path

import pytest

from src.cmirank.lora_joint import load_joint_config, joint_gpu_budget
from src.cmirank.lora import load_lora_config

ROOT = Path(__file__).resolve().parents[1]


def test_joint_inherits_science_and_binds_the_actual_reviewed_cpu20():
    original = load_lora_config(ROOT)[0]
    c, _ = load_joint_config(ROOT)
    allowed = {"schema_version", "prepare_run_id", "parameterization", "optimizer_updates"}
    assert all(c[k] == v for k, v in original.items() if k not in allowed)
    assert c["role"] == "joint" and c["optimizer_updates"] == 2 and c["updates_per_role"] == 1
    assert c["gpu_index"] == 1 and c["rank"] == 32 and not c["training_ready"]
    assert c["cpu_parent_report_sha256"] == "565b51744ca72be75139bd662ed02900d62520932d893cce9c4c463518913663"
    roles = {"actor": {"persistent_train_bytes": 19195938816},
             "critic": {"persistent_train_bytes": 19195990036}}
    r = joint_gpu_budget(roles, 81559*2**20, c["workspace_reserve_bytes"])
    assert r["within_estimated_budget"] and not r["full_ppo_fit_proven"]
    assert r["both_roles_params_grad_mom_bytes"] == 38391928852
    assert not joint_gpu_budget(roles, 16*2**30, c["workspace_reserve_bytes"])["within_estimated_budget"]


@pytest.mark.parametrize("key,value", [("gpu_index", 0), ("optimizer_updates", 3), ("role", "critic"),
    ("rank", 64), ("cpu_parent_report_sha256", "0"*64), ("host_load_reserve_bytes", 2**30)])
def test_joint_cannot_expand_resources_or_drift_from_cpu_scope(tmp_path, key, value):
    folder = tmp_path/"configs/cmirank"
    folder.mkdir(parents=True)
    for version in (1, 2, 3, 4):
        (folder/f"ppo_lora_v{version}.json").write_bytes((ROOT/f"configs/cmirank/ppo_lora_v{version}.json").read_bytes())
    c = json.loads((ROOT/"configs/cmirank/ppo_lora_joint_v1.json").read_text())
    c[key] = value
    (folder/"ppo_lora_joint_v1.json").write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_joint_config(tmp_path)


def test_reuses_original_coordinator_and_keeper_with_native_only_updates():
    coordinator = importlib.import_module("scripts.cmirank.33_smoke_actor_optimizer_gpu")
    assert callable(coordinator.main)
    source = (ROOT/"scripts/cmirank/33_smoke_actor_optimizer_gpu.py").read_text()
    assert 'choices=("full", "lora_joint", "lora_checkpoint"), default="full"' in source
    assert 'prepared = verify_optimizer_preparation(private, ROOT, c, path, commit)' in source
    assert 'prepared = verify_joint_preparation(private, ROOT, c, path, commit)' in source
    assert 'child_exited=worker.poll() == 0, child_kind="native_fsdp2"' in source
    assert "import torch" not in source
    worker = (ROOT/"scripts/cmirank/38_native_lora_joint_gpu.py").read_text()
    assert 'engine.train_batch(data, loss_function=partial(loss_fn, config=loss_config))' in worker
    for forbidden in ('engine.optimizer.step(', 'torch.optim.AdamW(', 'p.requires_grad_(', 'scancel', 'vllm.serve'):
        assert forbidden not in worker
    assert 'world_size=1' in worker and 'torch.cuda.device_count() != 1' in worker
    assert '"full_ppo_fit_proven": False' in worker and '"training_ready": False' in worker
    for name in ("run_real_policy_smoke.sh", "run_real_policy_with_keeper.sh"):
        assert 'lora_optimizer' in (ROOT/"scripts/cmirank"/name).read_text()


def test_new_cpu_gate_uses_full_frozen_response_eos_and_source_bound_inputs():
    source = (ROOT/"scripts/cmirank/37_prepare_lora_joint_cpu.py").read_text()
    assert 'verified_parents(private, load_optimizer_config(ROOT)[0])' in source
    assert 'prepare_micro_batches(data' in source and 'current.grad[:p-1].count_nonzero()' in source
    assert '"gpu_requested": False' in source and '"optimizer_updates": 0' in source
    assert 'reviewed_lora_parent(private, ROOT, c)' in source
