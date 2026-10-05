#!/usr/bin/env python3
"""CPU-only copy/verify/relocate smoke. NEVER deletes a source or cancels a job.

Run the tested GitHub checkout inside hoangnv242's existing reserved allocation.
Writes a delete-ready receipt only after full integrity + portability smoke.
Old-account cleanup is a separate, explicit verified operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import CLUSTER_ROOT, MIGRATION_SOURCE, require_allocation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def inventory(root: Path, *, hash_files: bool, record_runtime_sockets: bool = False) -> dict:
    """Never follow symlinks; reject special files and changing file contents."""
    rows, inode_hashes, queue = {}, {}, [root]
    while queue:
        directory = queue.pop()
        for path in sorted(directory.iterdir()):
            rel = path.relative_to(root).as_posix()
            before = path.lstat()
            row = {"mode": stat.S_IMODE(before.st_mode)}
            if stat.S_ISLNK(before.st_mode):
                row.update(kind="symlink", target=os.readlink(path))
            elif stat.S_ISDIR(before.st_mode):
                row.update(kind="directory")
                queue.append(path)
            elif stat.S_ISREG(before.st_mode):
                row.update(kind="file", size=before.st_size)
                if hash_files:
                    inode = (before.st_dev, before.st_ino)
                    row["sha256"] = inode_hashes.get(inode) or sha256(path)
                    inode_hashes[inode] = row["sha256"]
                    after = path.stat()
                    if (after.st_size, after.st_mtime_ns, after.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
                        raise RuntimeError(f"Source changed while hashing: {rel}")
            elif record_runtime_sockets and stat.S_ISSOCK(before.st_mode) and rel.startswith("cache/tmp/"):
                row.update(kind="nonportable_runtime_socket", reason="unix_ipc_endpoint_not_persistent_data")
            else:
                raise RuntimeError(f"Special file needs review, not a silent skip: {rel}")
            rows[rel] = row
    return rows


def portable_rows(rows: dict) -> dict:
    return {name: row for name, row in rows.items() if row["kind"] != "nonportable_runtime_socket"}


def smoke_selection(rows: dict) -> list[str]:
    candidates = [name for name, row in rows.items() if row["kind"] == "file" and 0 < row["size"] <= 1024 * 1024]
    ordered = sorted(candidates, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    selected = []
    for top in ("repo", "envs", "models", "cache", "runs", "logs"):
        selected.extend([name for name in ordered if name.startswith(top + "/")][:3])
    for name in ordered:
        if len(selected) == 20:
            break
        if name not in selected:
            selected.append(name)
    if len(selected) != 20:
        raise RuntimeError("Need twenty cross-component copy smoke files")
    return sorted(selected)


def compare(expected: dict, actual: dict) -> None:
    if expected != actual:
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        changed = [name for name in expected if name in actual and expected[name] != actual[name]][:5]
        raise RuntimeError(f"Tree integrity mismatch: missing={missing}, extra={extra}, changed={changed}")


def relocate_runtime(source: Path, destination: Path, work: Path) -> list[dict]:
    """Mechanical relocation of generated metadata only; immutable data untouched."""
    changes = []
    old, new = str(source), str(destination)
    rows = inventory(destination, hash_files=False)
    for rel, row in rows.items():
        path = destination / rel
        if row["kind"] == "symlink" and row["target"].startswith(old + "/"):
            target = new + row["target"][len(old):]
            before = row["target"]
            path.unlink()
            path.symlink_to(target)
            changes.append({"path": rel, "kind": "symlink", "before": before, "after": target})
        elif row["kind"] == "file" and rel.startswith("envs/") and row["size"] <= 2 * 1024 * 1024:
            in_bin = len(path.relative_to(destination / "envs").parts) >= 3 and path.relative_to(destination / "envs").parts[1] == "bin"
            generated = (in_bin or path.name == "pyvenv.cfg" or path.suffix == ".pth"
                         or path.name.startswith("__editable__"))
            if not generated:
                continue
            content = path.read_bytes()
            if b"\0" in content or old.encode() not in content:
                continue
            content.decode("utf-8")  # Fail, do not rewrite a binary/unknown codec.
            replacement = content.replace(old.encode(), new.encode())
            backup = work / "relocation-backup-hnv" / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            # Atomic replacement avoids modifying other aliases of a hardlinked
            # generated entrypoint in a package cache.
            temporary = path.with_name(path.name + ".relocate-hnv")
            temporary.write_bytes(replacement)
            temporary.chmod(row["mode"])
            os.replace(temporary, path)
            changes.append({"path": rel, "kind": "file", "before_sha256": hashlib.sha256(content).hexdigest(),
                            "after_sha256": hashlib.sha256(replacement).hexdigest(), "after_size": len(replacement)})
    for rel, row in inventory(destination, hash_files=False).items():
        if row["kind"] == "symlink" and row["target"].startswith(str(source.parent) + "/"):
            raise RuntimeError(f"Unrelocated old-account dependency: {rel}")
    return changes


def relocated_expected(original: dict, changes: list[dict]) -> dict:
    expected = {name: dict(row) for name, row in original.items()}
    for change in changes:
        row = expected[change["path"]]
        if change["kind"] == "symlink":
            if row["target"] != change["before"]:
                raise RuntimeError("Relocation symlink journal differs")
            row["target"] = change["after"]
        else:
            if row["sha256"] != change["before_sha256"]:
                raise RuntimeError("Relocation file journal differs")
            row.update(sha256=change["after_sha256"], size=change["after_size"])
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resume-empty-reviewed-failure", action="store_true")
    args = parser.parse_args()
    job = require_allocation()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Migration must remain CPU-only; leave both generator steps running")
    source, destination = MIGRATION_SOURCE, CLUSTER_ROOT
    if (source.is_symlink() or destination.is_symlink() or source.resolve() != source
            or destination.resolve() != destination or not source.is_dir() or destination.exists()
            or args.work_dir.resolve() != Path("/mnt/data/users/hoangnv242/memrec-migration-20261005-hnv")):
        raise RuntimeError("Unexpected/overlapping source, destination, workdir or existing target")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True):
        raise RuntimeError("Migration requires exact clean tested GitHub source")
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    if (work / "state.json").exists():
        previous = json.loads((work / "state.json").read_text())
        if (not args.resume_empty_reviewed_failure or previous.get("phase") != "FAILED_SOURCE_PRESERVED"
                or previous.get("source") != str(source) or previous.get("destination") != str(destination)
                or (work / "source-manifest.json").exists() or (work / "copy-smoke.json").exists()):
            raise RuntimeError("Existing nonempty/unreviewed attempt must not be overwritten")
        archive = work / "failed-precopy-attempt-hnv"
        archive.mkdir()
        shutil.copy2(work / "state.json", archive / "state.json")
        (work / "state.json").unlink()
    started = time.monotonic()
    def phase(name, **fields):
        record = {"phase": name, "elapsed_seconds": time.monotonic() - started,
                  "source": str(source), "destination": str(destination), "source_deleted": False,
                  "gpu_requested": False, "allocation_job_id": job, **fields}
        save(work / "state.json", record)
        print(json.dumps(record), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("migration timeout/interrupt")))
    try:
        phase("source_metadata_inventory")
        metadata = inventory(source, hash_files=False, record_runtime_sockets=True)
        excluded = {name: row for name, row in metadata.items() if row["kind"] == "nonportable_runtime_socket"}
        save(work / "nonportable-runtime-sockets.json", excluded)
        logical_bytes = sum(row.get("size", 0) for row in metadata.values())
        if shutil.disk_usage(destination.parent).free < logical_bytes + 10 * 2**30:
            raise RuntimeError("Insufficient space for full copy plus safety reserve")
        selected = smoke_selection(metadata)
        sample_root = work / "copy-smoke-hnv"
        sample_root.mkdir()
        file_list = work / "smoke-files-hnv.list"
        file_list.write_bytes(b"\0".join(name.encode() for name in selected) + b"\0")
        flags = ["-aH", "--no-owner", "--no-group"]
        phase("copy_smoke_20", source_entries=len(metadata), source_logical_bytes=logical_bytes)
        subprocess.run(["rsync", *flags, "--from0", f"--files-from={file_list}", str(source) + "/", str(sample_root) + "/"], check=True)
        for rel in selected:
            if sha256(source / rel) != sha256(sample_root / rel) or (source / rel).stat().st_size != (sample_root / rel).stat().st_size:
                raise RuntimeError(f"Twenty-file copy smoke failed: {rel}")
        save(work / "copy-smoke.json", {"status": "PASS", "files": selected, "count": 20, "rsync_flags": flags})
        phase("source_full_sha256", copy_smoke_passed=True)
        original = inventory(source, hash_files=True, record_runtime_sockets=True)
        save(work / "source-manifest.json", original)
        destination.mkdir()
        destination.chmod(stat.S_IMODE(source.stat().st_mode))
        phase("full_rsync", source_entries=len(original), source_logical_bytes=logical_bytes)
        with (work / "rsync-hnv.log").open("w") as log:
            if any(any(char in name for char in "*?[]") for name in excluded):
                raise RuntimeError("Runtime socket exclusion needs literal-name review")
            exclusions = [f"--exclude=/{name}" for name in excluded]
            subprocess.run(["rsync", *flags, *exclusions, "--partial", "--info=stats2", str(source) + "/", str(destination) + "/"],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        phase("destination_full_sha256")
        copied = inventory(destination, hash_files=True)
        compare(portable_rows(original), copied)
        save(work / "copied-manifest.json", copied)
        phase("verified_byte_identical_relocating_generated_runtime")
        changes = relocate_runtime(source, destination, work)
        save(work / "relocation-journal.json", changes)
        compare(relocated_expected(portable_rows(original), changes), inventory(destination, hash_files=True))
        target_repo = destination / "repo/MemRec-hnv"
        source_git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target_repo, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=target_repo, text=True):
            raise RuntimeError("Copied source has tracked edits; preserve instead of overwriting")
        # New source goes through GitHub, never patch-copy into the running repo.
        subprocess.run(["git", "fetch", "origin", "experiment/evo-multihop"], cwd=target_repo, check=True)
        subprocess.run(["git", "merge", "--ff-only", args.expected_commit], cwd=target_repo, check=True)
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target_repo, text=True).strip() != args.expected_commit:
            raise RuntimeError("New runtime repo is not the tested source")
        phase("portable_env_and_20_user_cpu_smoke", relocation_entries=len(changes))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MEMREC_ROOT=str(destination),
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   XDG_CACHE_HOME=str(destination / "cache"), HF_HOME=str(destination / "cache/huggingface"),
                   TMPDIR=str(destination / "cache/tmp"))
        env_checks = {}
        for name in ("llm-hnv", "cmirank-qwen35-t513-hnv", "sasrec-hnv"):
            python = destination / "envs" / name / "bin/python"
            output = subprocess.check_output([str(python), "-c",
                "import json,sys,torch,importlib.metadata as m; "
                "assert not torch.cuda.is_available() and not torch.cuda.is_initialized(); "
                "versions={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
                "print(json.dumps({'prefix':sys.prefix,'torch':torch.__version__,'transformers':versions.get('transformers')}))"], env=env, text=True)
            record = json.loads(output.strip().splitlines()[-1])
            if record["prefix"] != str(destination / "envs" / name):
                raise RuntimeError("Venv is still importing under the old workspace")
            env_checks[name] = record
        if (env_checks["llm-hnv"]["transformers"] != "4.55.4"
                or env_checks["cmirank-qwen35-t513-hnv"]["transformers"] != "5.13.0"):
            raise RuntimeError("Memory/policy environment versions differ from the frozen successful runs")
        save(work / "environment-smoke.json", env_checks)
        with (work / "cpu-smoke-hnv.log").open("w") as log:
            subprocess.run([str(destination / "envs/llm-hnv/bin/python"), str(ROOT / "scripts/cluster/smoke_relocated_workspace_cpu.py"),
                            "--repo", str(target_repo), "--output-dir", str(work / "cpu-smoke-hnv")],
                           env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        phase("final_source_unchanged_sha256")
        compare(original, inventory(source, hash_files=True, record_runtime_sockets=True))
        require_allocation()  # Receipt must not claim success after losing the reservation.
        receipt = {"status": "COPY_VERIFIED_RELOCATED_CPU_SMOKE_PASS_SOURCE_NOT_DELETED",
            "source_root": str(source), "destination_root": str(destination), "source_deleted": False,
            "source_entries": len(original), "source_logical_bytes": logical_bytes,
            "copied_entries": len(copied), "nonportable_runtime_socket_count": len(excluded),
            "runtime_socket_inventory_sha256": sha256(work / "nonportable-runtime-sockets.json"),
            "source_manifest_sha256": sha256(work / "source-manifest.json"),
            "copied_manifest_sha256": sha256(work / "copied-manifest.json"),
            "relocation_journal_sha256": sha256(work / "relocation-journal.json"),
            "cpu_smoke_report_sha256": sha256(work / "cpu-smoke-hnv/report.json"),
            "source_git_commit": source_git, "destination_git_commit": args.expected_commit,
            "environment_smoke": env_checks, "elapsed_seconds": time.monotonic() - started,
            "gpu_requested": False, "reserved_job_cancelled": False, "training_ready": False}
        save(work / "receipt.json", receipt)
        phase("delete_ready_independent_review_required", receipt_sha256=sha256(work / "receipt.json"))
    except BaseException as error:
        phase("FAILED_SOURCE_PRESERVED", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
