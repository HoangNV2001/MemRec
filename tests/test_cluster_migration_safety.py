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
    ("21273", "0", "omni-gen-1", "hoangnv242", "RUNNING", {7}, {7}),
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


def test_only_runtime_unix_sockets_are_explicitly_inventoried_not_silently_skipped(tmp_path):
    import socket
    module = migration_module()
    root = tmp_path / "root"
    (root / "cache/tmp").mkdir(parents=True)
    endpoint = socket.socket(socket.AF_UNIX)
    endpoint.bind(str(root / "cache/tmp/ipc"))
    try:
        with pytest.raises(RuntimeError, match="Special file"):
            module.inventory(root, hash_files=True)
        rows = module.inventory(root, hash_files=True, record_runtime_sockets=True)
        assert rows["cache/tmp/ipc"]["kind"] == "nonportable_runtime_socket"
        assert "cache/tmp/ipc" not in module.portable_rows(rows)
    finally:
        endpoint.close()


def test_socket_outside_project_runtime_cache_remains_a_hard_failure(tmp_path):
    import socket
    module = migration_module()
    endpoint = socket.socket(socket.AF_UNIX)
    endpoint.bind(str(tmp_path / "ipc"))
    try:
        with pytest.raises(RuntimeError, match="Special file"):
            module.inventory(tmp_path, hash_files=True, record_runtime_sockets=True)
    finally:
        endpoint.close()


def test_named_reader_acl_adds_no_group_world_or_reader_write_permission():
    import struct
    path = Path(__file__).resolve().parents[1] / "scripts/cluster/grant_migration_read.py"
    spec = importlib.util.spec_from_file_location("grant_reader_hnv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blob = module.read_acl(1052)
    assert struct.unpack("<I", blob[:4])[0] == 2
    entries = [struct.unpack("<HHI", blob[start:start + 8]) for start in range(4, len(blob), 8)]
    assert entries == [(1, 6, 0xFFFFFFFF), (2, 4, 1052), (4, 0, 0xFFFFFFFF),
                       (16, 4, 0xFFFFFFFF), (32, 0, 0xFFFFFFFF)]


def test_private_donor_snapshot_is_hash_bound_and_requires_original_metadata(tmp_path):
    import os
    module = migration_module()
    source, staging = tmp_path / "source", tmp_path / "staging"
    source.mkdir(); staging.mkdir()
    original, private = source / "private.json", staging / "private.json"
    original.write_text("opaque-original-bytes"); original.chmod(0o600)
    private.write_bytes(original.read_bytes()); private.chmod(0o600)
    info = original.stat()
    rows = {"private.json": {"snapshot_path": str(private), "size": info.st_size,
        "mtime_ns": info.st_mtime_ns, "mode": 0o600, "sha256": module.sha256(private)}}
    assert module.inventory(source, hash_files=True, protected=rows)["private.json"]["sha256"] == module.sha256(original)
    private.write_text("tampered-copy")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        module.inventory(source, hash_files=True, protected=rows)
    private.write_bytes(original.read_bytes())
    os.utime(original, ns=(info.st_atime_ns, info.st_mtime_ns + 100))
    with pytest.raises(RuntimeError, match="source metadata changed"):
        module.inventory(source, hash_files=True, protected=rows)


def review_module():
    path = Path(__file__).resolve().parents[1] / "scripts/cluster/review_workspace_migration.py"
    spec = importlib.util.spec_from_file_location("review_migration_hnv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_rejects_non_generated_journal_edits(tmp_path):
    module = review_module()
    with pytest.raises(RuntimeError, match="immutable/non-generated"):
        module.review_journal({"models/weights.bin": {}}, [{"path": "models/weights.bin", "kind": "file"}],
                              tmp_path, tmp_path / "old", tmp_path / "new")


def test_review_rejects_symlink_journal_that_is_not_exact_root_rebase(tmp_path):
    module = review_module()
    with pytest.raises(RuntimeError, match="exact root substitution"):
        module.review_journal({"link": {}}, [{"path": "link", "kind": "symlink",
            "before": str(tmp_path / "old/x"), "after": str(tmp_path / "new/y")}],
            tmp_path, tmp_path / "old", tmp_path / "new")


def test_deployed_code_exception_does_not_hide_immutable_data_or_untracked_files():
    module = review_module()
    prefix = module.REPO_PREFIX
    expected = {prefix + "src/example.py": {"sha256": "old"}, prefix + "data/pinned.pkl": {"sha256": "data"},
                prefix + ".git/HEAD": {"sha256": "old-head"}, "models/weights": {"sha256": "weights"}}
    changed = {key: dict(row) for key, row in expected.items()}
    changed[prefix + "src/example.py"]["sha256"] = "new"
    changed[prefix + ".git/HEAD"]["sha256"] = "new-head"
    module.compare_after_deploy(expected, changed, {prefix + "src/example.py"})
    changed[prefix + "data/pinned.pkl"]["sha256"] = "tampered"
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        module.compare_after_deploy(expected, changed, {prefix + "src/example.py"})


def cleanup_module():
    path = Path(__file__).resolve().parents[1] / "scripts/cluster/cleanup_migrated_source.py"
    spec = importlib.util.spec_from_file_location("cleanup_migration_hnv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_path_reference_does_not_match_sibling_project():
    module = cleanup_module()
    root = Path("/mnt/data/users/anhnct/memrec-hnv")
    assert module.mentions_root(f"{root}/envs/bin/python\0--arg", root)
    assert module.mentions_root(f"{root} (deleted)", root)
    assert not module.mentions_root(f"{root}-other/envs/python", root)


def test_cleanup_git_read_cannot_trust_an_unrelated_repo():
    module = cleanup_module()
    with pytest.raises(RuntimeError, match="outside the exact"):
        module.read_git(Path("/mnt/data/users/hoangnv242"), "rev-parse", "HEAD")


def test_cleanup_rejects_source_metadata_changed_after_full_review(tmp_path):
    module = cleanup_module()
    (tmp_path / "data").write_text("frozen-data")
    original = module.inventory(tmp_path, hash_files=True)
    cutoff = max(path.lstat().st_ctime_ns for path in (tmp_path, tmp_path / "data")) + 1_000_000
    module.check_unchanged_metadata(tmp_path, original, cutoff)
    with pytest.raises(RuntimeError, match="changed after"):
        module.check_unchanged_metadata(tmp_path, original, cutoff - 10_000_000)


def test_cleanup_rejects_unreviewed_new_file(tmp_path):
    module = cleanup_module()
    (tmp_path / "data").write_text("frozen-data")
    original = module.inventory(tmp_path, hash_files=True)
    (tmp_path / "new-data").write_text("not copied")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        module.check_unchanged_metadata(tmp_path, original, 2**63 - 1)
