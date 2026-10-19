from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from src.cmirank.lora_checkpoint import (load_checkpoint_config, checkpoint_budget,
    tensor_digest, train_state_fingerprint, run_checkpoint_roundtrip)
from src.cmirank.lora_joint import load_joint_config

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_inherits_approved_training_scope_and_preserves_critic_head():
    c, _ = load_checkpoint_config(ROOT)
    parent, _ = load_joint_config(ROOT)
    allowed = {"prepare_run_id", "run_id", "parameterization", "optimizer_updates", "updates_per_role", "worker_status"}
    assert all(c[k] == v for k, v in parent.items() if k not in allowed)
    assert c["gpu_index"] == 1 and c["rank"] == 32 and c["model_dtype"] == "fp32"
    assert not c["checkpoint_save_lora_only"] and c["checkpoint_contents"] == ["model", "optimizer", "extra"]
    assert c["optimizer_updates"] == 6 and c["restored_logical_updates_per_role"] == 2
    assert not c["training_ready"] and not c["books_outcomes_accessed"]


@pytest.mark.parametrize("key,value", [("checkpoint_save_lora_only", True), ("optimizer_updates", 4),
    ("checkpoint_contents", ["model"]), ("gpu_index", 0), ("rank", 64),
    ("joint_parent_report_sha256", "0"*64), ("schema_version", True)])
def test_checkpoint_contract_rejects_hidden_method_resource_or_provenance_changes(tmp_path, key, value):
    folder = tmp_path/"configs/cmirank"
    folder.mkdir(parents=True)
    for source in (ROOT/"configs/cmirank").glob("ppo_lora*.json"):
        (folder/source.name).write_bytes(source.read_bytes())
    profile = folder/"ppo_lora_checkpoint_v1.json"
    c = json.loads(profile.read_text())
    c[key] = value
    profile.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_checkpoint_config(tmp_path)


def example_roles():
    return {"actor": {"persistent_train_bytes": 19195938816, "parameter_bytes": 18416781312,
                      "adam_moment_bytes": 519438336},
            "critic": {"persistent_train_bytes": 19195990036, "parameter_bytes": 18416801800,
                       "adam_moment_bytes": 519458824}}


def test_checkpoint_budget_does_not_assume_native_fsdp2_is_cpu_offloaded():
    c, _ = load_checkpoint_config(ROOT)
    r = checkpoint_budget(example_roles(), 81559*2**20, 64*2**30, 4*2**30, 28373980*1024, 100*2**30, c)
    assert r["within_estimated_budget"] and not r["checkpoint_fit_proven"]
    assert r["gpu_required_bytes"] == 38391928852+18936260624+8*2**30
    assert not checkpoint_budget(example_roles(), 40*2**30, 64*2**30, 4*2**30, 28373980*1024, 100*2**30, c)["within_estimated_budget"]
    assert not checkpoint_budget(example_roles(), 81559*2**20, 48*2**30, 4*2**30, 28373980*1024, 100*2**30, c)["within_estimated_budget"]
    assert not checkpoint_budget(example_roles(), 81559*2**20, 64*2**30, 4*2**30, 28373980*1024, 10*2**30, c)["within_estimated_budget"]


def test_complete_tensor_fingerprint_includes_tail_not_only_first32_samples():
    torch = pytest.importorskip("torch")
    value = torch.zeros(128)
    original = tensor_digest(value)
    value[-1] = 1
    assert tensor_digest(value) != original
    assert tensor_digest(value) != tensor_digest(value.reshape(2, 64))


@pytest.mark.parametrize("omit_head", [False, True])
def test_native_resume_harness_detects_scalar_head_loss_and_restores_real_optimizer(tmp_path, monkeypatch, omit_head):
    torch = pytest.importorskip("torch")
    from src.cmirank import lora_checkpoint as module
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora = torch.nn.Linear(2, 2)
            self.score = torch.nn.Linear(2, 1)
            self.backbone = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
    model = Model()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    class Engine:
        module = model
        lr_scheduler = scheduler
        checkpoint_manager = SimpleNamespace()

        def __init__(self):
            self.optimizer = optimizer
            self.updates = 0

        def train_mode(self):
            return nullcontext()

        def lr_scheduler_step(self):
            scheduler.step()

        def train_batch(self, data, loss_function):
            self.updates += 1
            optimizer.zero_grad()
            model.score(model.lora(torch.ones(2, 2))).square().mean().backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            return {"metrics": {"grad_norm": norm.item()}}

        def save_checkpoint(self, directory, global_step, max_ckpt_to_keep):
            assert global_step == 1 and max_ckpt_to_keep is None
            self.saved = deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
            self.rng = (random.getstate(), torch.get_rng_state())
            Path(directory).mkdir(parents=True)
            (Path(directory)/"cpu-fixture-marker").write_text("native-api-test-double-only")

        def load_checkpoint(self, directory, del_local_after_load):
            assert not del_local_after_load
            values, optim, sched = deepcopy(self.saved)
            if omit_head:
                values = {k: v for k, v in values.items() if not k.startswith("score.")}
            model.load_state_dict(values, strict=not omit_head)
            optimizer.load_state_dict(optim)
            scheduler.load_state_dict(sched)
            random.setstate(self.rng[0])
            torch.set_rng_state(self.rng[1])
    engine = Engine()
    monkeypatch.setattr(module, "rng_probe", lambda: {"python": random.random(), "torch_cpu": torch.rand(8).tolist()})
    engine.train_batch(None, None)
    initial = train_state_fingerprint(engine)
    score = lambda active: [[float(x.detach())] for x in active.module.score(torch.ones(20, 2)).flatten()]
    if omit_head:
        with pytest.raises(ValueError, match="exact adapter/head/Adam/scheduler/RNG"):
            run_checkpoint_roundtrip(engine, None, None, {}, "critic", tmp_path, lambda x: None, score)
    else:
        result = run_checkpoint_roundtrip(engine, None, None, {}, "critic", tmp_path, lambda x: None, score)
        assert engine.updates == 3 and result["native_continuation_step"] == 2
        assert result["complete_trainable_and_adam_state_equal"] and result["response_readout_max_absolute_delta"] == 0
        assert train_state_fingerprint(engine) != initial
        audit = json.loads((tmp_path/"checkpoint-critic-audit.json").read_text())
        assert audit["train_state_before"] == audit["train_state_restored"] == initial


def test_checkpoint_gate_stays_on_original_controller_and_does_not_patch_native_framework():
    source = (ROOT/"src/cmirank/lora_checkpoint.py").read_text()
    for call in ('engine.save_checkpoint(', 'engine.load_checkpoint(', 'engine.train_batch(', 'engine.lr_scheduler_step('):
        assert call in source
    for forbidden in ('optimizer.step(', 'requires_grad_(', 'load_rng_state(', 'scancel', 'hf_model"'):
        assert forbidden not in source
    for filename in ("run_real_policy_smoke.sh", "run_real_policy_with_keeper.sh"):
        assert "lora_checkpoint" in (ROOT/"scripts/cmirank"/filename).read_text()
    coordinator = (ROOT/"scripts/cmirank/33_smoke_actor_optimizer_gpu.py").read_text()
    assert 'child_exited=worker.poll() == 0, child_kind="native_fsdp2"' in coordinator
    assert 'verify_checkpoint_preparation(private, ROOT, c, path, commit)' in coordinator
    assert 'gpu_index=1, protect_other_workloads=True' in coordinator
