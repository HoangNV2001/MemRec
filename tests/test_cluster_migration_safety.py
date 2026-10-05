"""Portable workspace and reserved-resource guards; no SSH/Slurm/CUDA calls."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cmirank.gpu_resources import GPUCard, generator_step_target, select_reserved_gpu1
from src.cluster_runtime import CLUSTER_ROOT, require_allocation, require_project_root


def migration_module():
    path = Path(__file__).resolve().parents[1] / "scripts/cluster/migrate_workspace.py"
    spec = importlib.util.spec_from_file_location("migration_hnv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reserved_selector_never_uses_free_gpu0_or_a_busy_gpu1():
    zero = GPUCard(0, "GPU-zero", "NVIDIA H100", 0, 1, 81559)
    one = GPUCard(1, "GPU-one", "NVIDIA H100", 0, 1, 81559)
    assert select_reserved_gpu1([zero, one], set()) == one
    with pytest.raises(ValueError):
        select_reserved_gpu1([zero, one], {"GPU-one"})
    with pytest.raises(ValueError):
        select_reserved_gpu1([zero], set())


@pytest.mark.parametrize("job,step,name,owner,state,pids,members", [
    ("21273", "batch", "omni-gen-1", "hoangnv242", "RUNNING", {7}, {7}),
    ("21273", "extern", "omni-gen-1", "hoangnv242", "RUNNING", {7}, {7}),
    ("21273", "", "omni-gen-1", "hoangnv242", "RUNNING", {7}, {7}),
    ("21273", "1", "omni-gen-0", "hoangnv242", "RUNNING", {7}, {7}),
    ("21273", "1", "omni-gen-1", "anhntc2", "RUNNING", {7}, {7}),
    ("21273", "1", "omni-gen-1", "hoangnv242", "RUNNING", {7, 8}, {7}),
    ("21273", "1", "omni-gen-1", "hoangnv242", "PENDING", {7}, {7}),
])
def test_step_handoff_cannot_target_job_batch_gpu0_or_foreign_work(job, step, name, owner, state, pids, members):
    with pytest.raises(ValueError):
        generator_step_target(job, step, name=name, owner=owner, state=state, gpu_pids=pids, step_pids=members)


def test_step_handoff_has_only_numeric_job_dot_step_target():
    assert generator_step_target("21273", "12", name="omni-gen-1", owner="hoangnv242", state="RUNNING",
                                 gpu_pids={7}, step_pids={7, 8}) == "21273.12"


def test_only_new_owner_and_exact_running_allocation(monkeypatch):
    import src.cluster_runtime as runtime
    monkeypatch.setenv("SLURM_JOB_ID", "21273")
    monkeypatch.setattr(runtime.pwd, "getpwuid", lambda _: SimpleNamespace(pw_name="hoangnv242"))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "hoangnv242 RUNNING senvoice-pro-opt\n")
    assert require_allocation() == "21273"
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "hoangnv242 RUNNING train_TTS_4gpu\n")
    with pytest.raises(RuntimeError):
        require_allocation()
    monkeypatch.setattr(runtime.pwd, "getpwuid", lambda _: SimpleNamespace(pw_name="anhntc2"))
    with pytest.raises(RuntimeError):
        require_allocation()


def test_project_root_cannot_fall_back_to_old_account_or_workspace_parent():
    assert require_project_root(CLUSTER_ROOT) == CLUSTER_ROOT
    for path in (CLUSTER_ROOT.parent, Path("/mnt/data/users/anhnct/memrec-hnv"), Path("/mnt/data")):
        with pytest.raises(RuntimeError):
            require_project_root(path)


def test_copy_smoke_selects_twenty_files_across_all_components():
    module = migration_module()
    rows = {f"{top}/file-{i}.json": {"kind": "file", "size": 20} for top in
            ("repo", "models", "envs", "cache", "runs", "logs") for i in range(8)}
    selected = module.smoke_selection(rows)
    assert len(selected) == len(set(selected)) == 20
    assert {path.split("/")[0] for path in selected} == {"repo", "models", "envs", "cache", "runs", "logs"}
    assert selected == module.smoke_selection(dict(reversed(list(rows.items()))))


def test_inventory_does_not_follow_external_links_and_fails_byte_mismatch(tmp_path):
    module = migration_module()
    root = tmp_path / "source"
    root.mkdir()
    (root / "data").write_text("immutable-data")
    (root / "external").symlink_to(tmp_path / "not-here")
    rows = module.inventory(root, hash_files=True)
    assert rows["external"]["kind"] == "symlink"
    assert rows["data"]["sha256"] == module.sha256(root / "data")
    module.compare(rows, dict(rows))
    different = {key: dict(value) for key, value in rows.items()}
    different["data"]["size"] = 1
    with pytest.raises(RuntimeError):
        module.compare(rows, different)


def test_relocation_changes_generated_runtime_only_and_does_not_mutate_hardlink_alias(tmp_path):
    import os
    module = migration_module()
    source, target, work = tmp_path / "old-user/project", tmp_path / "new-user/project", tmp_path / "work"
    source.mkdir(parents=True); target.mkdir(parents=True); work.mkdir()
    (target / "envs/e/bin").mkdir(parents=True)
    launcher = target / "envs/e/bin/pip"
    launcher.write_text(f"#!{source}/envs/e/bin/python\n")
    (target / "cache").mkdir()
    os.link(launcher, target / "cache/immutable-alias")
    (target / "models").mkdir()
    model = target / "models/weights.bin"
    model.write_bytes((str(source) + "\0weights").encode())
    (target / "runs").mkdir()
    historical = target / "runs/manifest.json"
    historical.write_text(f'{{"historical_root":"{source}"}}')
    (target / "internal-link").symlink_to(str(source / "models/weights.bin"))
    before = module.inventory(target, hash_files=True)
    changes = module.relocate_runtime(source, target, work)
    module.compare(module.relocated_expected(before, changes), module.inventory(target, hash_files=True))
    assert launcher.read_text().startswith(f"#!{target}")
    assert (target / "cache/immutable-alias").read_text().startswith(f"#!{source}")
    assert historical.read_text() == f'{{"historical_root":"{source}"}}'
    assert model.read_bytes() == (str(source) + "\0weights").encode()
    assert (target / "internal-link").resolve() == model
