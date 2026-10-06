#!/usr/bin/env python3
"""Fresh, CPU-only integrity review of the completed migration; NEVER deletes."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import CLUSTER_ROOT, MIGRATION_SOURCE, require_allocation
from scripts.cluster.grant_migration_read import FILES
from scripts.cluster.migrate_workspace import compare, inventory, portable_rows, relocated_expected, save, sha256

WORK = Path("/mnt/data/users/hoangnv242/memrec-migration-20261005-hnv")
REPO_PREFIX = "repo/MemRec-hnv/"


def review_journal(original, changes, work, source, destination):
    """Check the actual backed-up bytes and reject edits outside generated metadata."""
    seen = set()
    for change in changes:
        name = change["path"]
        if name in seen or name not in original:
            raise RuntimeError("Duplicate/unknown relocation journal path")
        seen.add(name)
        if change["kind"] == "symlink":
            if (not change["before"].startswith(str(source) + "/")
                    or change["after"] != str(destination) + change["before"][len(str(source)):]):
                raise RuntimeError("Relocation symlink is not an exact root substitution")
        elif change["kind"] == "file":
            parts = Path(name).parts
            leaf = Path(name)
            if (len(parts) < 3 or parts[0] != "envs" or not (
                    (len(parts) >= 4 and parts[2] == "bin") or leaf.name == "pyvenv.cfg"
                    or leaf.suffix == ".pth" or leaf.name.startswith("__editable__"))):
                raise RuntimeError("Relocation changed immutable/non-generated content")
            backup = work / "relocation-backup-hnv" / name
            if sha256(backup) != change["before_sha256"]:
                raise RuntimeError("Relocation backup hash mismatch")
            content = backup.read_bytes()
            content.decode("utf-8")
            replacement = content.replace(str(source).encode(), str(destination).encode())
            import hashlib
            if (b"\0" in content or replacement == content
                    or hashlib.sha256(replacement).hexdigest() != change["after_sha256"]
                    or len(replacement) != change["after_size"]):
                raise RuntimeError("Journal is not an exact generated-prefix substitution")
        else:
            raise RuntimeError("Unknown relocation kind")
    return relocated_expected(original, changes)


def compare_after_deploy(expected, actual, approved_code_paths):
    """Only clean GitHub-deployed changed code and .git metadata may differ."""
    new_parents = {parent.as_posix() for name in approved_code_paths for parent in Path(name).parents
                   if parent.as_posix() not in expected and parent.as_posix() in actual
                   and actual[parent.as_posix()].get("kind") == "directory"}
    def retained(rows):
        return {name: row for name, row in rows.items()
                if name != REPO_PREFIX + ".git" and not name.startswith(REPO_PREFIX + ".git/")
                and name not in approved_code_paths and name not in new_parents}
    compare(retained(expected), retained(actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tool-commit", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--fresh-donor-manifest", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    job = require_allocation()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Review must leave GPUs/generators untouched")
    for root in (WORK, MIGRATION_SOURCE, CLUSTER_ROOT):
        if root.is_symlink() or root.resolve() != root or not root.is_dir():
            raise RuntimeError("Missing/aliased migration scope")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_tool_commit or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True):
        raise RuntimeError("Review tool must be the exact clean tested GitHub source")
    receipt_path = WORK / "receipt.json"
    if sha256(receipt_path) != args.receipt_sha256:
        raise RuntimeError("Receipt differs from the independently downloaded copy")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"] != "COPY_VERIFIED_RELOCATED_CPU_SMOKE_PASS_SOURCE_NOT_DELETED"
            or receipt["source_root"] != str(MIGRATION_SOURCE) or receipt["destination_root"] != str(CLUSTER_ROOT)
            or any(receipt[name] for name in ("source_deleted", "gpu_requested", "reserved_job_cancelled", "training_ready"))):
        raise RuntimeError("Wrong/incomplete migration receipt")
    for name, key in (("source-manifest.json", "source_manifest_sha256"),
                      ("copied-manifest.json", "copied_manifest_sha256"),
                      ("relocation-journal.json", "relocation_journal_sha256"),
                      ("nonportable-runtime-sockets.json", "runtime_socket_inventory_sha256"),
                      ("cpu-smoke-hnv/report.json", "cpu_smoke_report_sha256")):
        if sha256(WORK / name) != receipt[key]:
            raise RuntimeError("Migration proof hash changed")
    original = json.loads((WORK / "source-manifest.json").read_text())
    copied = json.loads((WORK / "copied-manifest.json").read_text())
    compare(portable_rows(original), copied)
    donor_path = args.fresh_donor_manifest
    if (donor_path.is_symlink() or donor_path.resolve() != WORK / "fresh-donor-review-20261006-hnv.json"
            or not 0 <= time.time() - donor_path.stat().st_mtime <= 1800):
        raise RuntimeError("Need a fresh private donor-side checksum review")
    fresh = json.loads(donor_path.read_text())
    staged = json.loads((WORK / "protected-source-hnv/donor-manifest.json").read_text())
    if fresh != staged or set(fresh["files"]) != set(FILES) or fresh["permissions_modified"]:
        raise RuntimeError("Fresh donor checksum/metadata does not match the migration snapshot")
    protected = {name: {**row, "snapshot_path": str(WORK / "protected-source-hnv" / name)}
                 for name, row in fresh["files"].items()}
    compare(original, inventory(MIGRATION_SOURCE, hash_files=True, record_runtime_sockets=True, protected=protected))
    journal = json.loads((WORK / "relocation-journal.json").read_text())
    expected = review_journal(copied, journal, WORK, MIGRATION_SOURCE, CLUSTER_ROOT)
    repo = CLUSTER_ROOT / REPO_PREFIX
    deployed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if deployed != receipt["destination_git_commit"] or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True):
        raise RuntimeError("Deployed repo changed/dirty after smoke")
    code_changes = subprocess.check_output(["git", "diff", "--name-only", "--no-renames", "-z",
        receipt["source_git_commit"], deployed], cwd=repo).decode().split("\0")
    allowed = {REPO_PREFIX + name for name in code_changes if name}
    actual = inventory(CLUSTER_ROOT, hash_files=True)
    compare_after_deploy(expected, actual, allowed)
    report = json.loads((WORK / "cpu-smoke-hnv/report.json").read_text())
    if (report["users"] != 20 or report["fake_logical_calls"] != 100 or report["real_llm_requests"] != 0
            or report["gpu_requested"] or report["training_ready"] or report["warmup_stage_w_calls"] != 20):
        raise RuntimeError("Portable twenty-user CPU smoke proof is incomplete")
    if json.loads((WORK / "environment-smoke.json").read_text()) != receipt["environment_smoke"]:
        raise RuntimeError("Environment smoke proof differs")
    # The SASRec environment inherits host packages: compare both prefixes on
    # this same node, without claiming historical package-version pinning.
    version_probe = "import json,sys,torch,transformers; print(json.dumps({'prefix':sys.prefix,'torch':torch.__version__,'transformers':transformers.__version__}))"
    versions = {}
    for label, root in (("source", MIGRATION_SOURCE), ("destination", CLUSTER_ROOT)):
        versions[label] = json.loads(subprocess.check_output([str(root / "envs/sasrec-hnv/bin/python"),
            "-c", version_probe], text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")))
        if versions[label]["prefix"] != str(root / "envs/sasrec-hnv"):
            raise RuntimeError("SASRec prefix portability mismatch")
    if any(versions["source"][key] != versions["destination"][key] for key in ("torch", "transformers")):
        raise RuntimeError("Inherited SASRec packages changed during relocation")
    require_allocation()
    output = WORK / "independent-review-20261006-hnv.json"
    if output.exists():
        raise RuntimeError("Do not overwrite an earlier independent review")
    save(output, {"status": "FRESH_SOURCE_AND_DESTINATION_FULL_HASH_REVIEW_PASS",
        "source_root": str(MIGRATION_SOURCE), "destination_root": str(CLUSTER_ROOT),
        "source_deleted": False, "receipt_sha256": args.receipt_sha256,
        "fresh_donor_manifest_sha256": sha256(donor_path), "protected_files_verified": len(protected),
        "relocation_entries_verified": len(journal), "source_entries": len(original),
        "destination_entries": len(actual), "github_code_change_paths_verified": len(allowed),
        "sasrec_same_node_inherited_versions": versions, "historical_sasrec_env_version_pin_proven": False,
        "tool_commit": head, "allocation_job_id": job, "node": socket.gethostname(),
        "gpu_requested": False, "reserved_job_cancelled": False, "training_ready": False,
        "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": "INDEPENDENT_REVIEW_PASS_SOURCE_NOT_DELETED", "report_sha256": sha256(output),
                      "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
