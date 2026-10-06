#!/usr/bin/env python3
"""Donor-account cleanup of ONE verified MemRec subtree; no job/GPU control.

Check process references on the login node first (--process-check-only), then
perform filesystem cleanup in the donor's existing allocation, with no CUDA.
The old allocation is used only for this explicitly requested source cleanup,
never for new research compute. Receipt is emitted to stdout, not the old tree.
"""

import argparse
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import CLUSTER_ROOT, MIGRATION_SOURCE
from scripts.cluster.grant_migration_read import FILES
from scripts.cluster.migrate_workspace import compare, inventory, sha256

WORK = Path("/mnt/data/users/hoangnv242/memrec-migration-20261005-hnv")
REVIEW_SHA = "1115f0bfd94c7e14a0047fcc2c1bce870e297eecae0971d2f64cc140be3bc4e3"
RECEIPT_SHA = "aad18840fa4050110bf6cbe66f580053c0c47bcaa00e01f9df1ba2f81a9dc1fb"


def read_git(repo, *args):
    # The donor intentionally reads the new-owned proof/tool checkout. Trust
    # only these two explicit paths. The donor's Git 2.34.1 does not honor
    # command-line safe.directory; explicit git-dir/work-tree avoids discovery
    # without widening global trust or refreshing either index.
    if repo not in (ROOT, CLUSTER_ROOT / "repo/MemRec-hnv") or repo.is_symlink():
        raise RuntimeError("Git read outside the exact migration tool/destination")
    return subprocess.check_output(["git", "--no-optional-locks", f"--git-dir={repo / '.git'}",
                                   f"--work-tree={repo}", *args],
                                   cwd=repo, text=True)


def mentions_root(text, root):
    # Match paths, not an unrelated sibling such as memrec-hnv-other.
    import re
    return re.search(re.escape(str(root)) + r"(?=/|[\s\x00]|$)", text) is not None


def is_ssh_transport(command, comm):
    import re
    # SSH session transports are non-dumpable; their shell/exec children are
    # inspected separately. Do NOT exempt internal-sftp or arbitrary commands.
    return comm == "sshd" and re.fullmatch(r"sshd: anhntc2@(notty|pts/\d+(?:,pts/\d+)*)", command.strip()) is not None


def process_references(root, *, proc=Path("/proc"), uid=None):
    """Read-only donor UID scan, without printing commands/environments/secrets."""
    uid = os.geteuid() if uid is None else uid
    references, scanned, transports = [], 0, []
    for path in proc.iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != uid:
                continue
            scanned += 1
            roles = []
            command = (path / "cmdline").read_bytes().decode(errors="replace").strip("\0 ")
            if is_ssh_transport(command, (path / "comm").read_text().strip()):
                transports.append(int(path.name))
                continue
            if mentions_root(command, root):
                roles.append("command_path")
            if mentions_root(os.readlink(path / "cwd"), root):
                roles.append("working_directory")
            if mentions_root((path / "maps").read_text(), root):
                roles.append("mapped_dependency")
            for fd in (path / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if mentions_root(target, root):
                    roles.append("open_file")
                    break
            if roles:
                references.append({"pid": int(path.name), "roles": sorted(set(roles))})
        except FileNotFoundError:
            continue  # Process exited during the diagnostic.
        # Permission errors for a live donor process are a hard failure.
    return {"node": socket.gethostname(), "donor_processes_scanned": scanned,
            "ssh_transport_pids_separately_classified": transports, "source_references": references}


def check_unchanged_metadata(root, original, review_finished_ns):
    metadata = inventory(root, hash_files=False, record_runtime_sockets=True)
    compare({name: {k: v for k, v in row.items() if k != "sha256"} for name, row in original.items()}, metadata)
    for name in ("", *metadata):
        info = (root / name).lstat()
        if max(info.st_mtime_ns, info.st_ctime_ns) > review_finished_ns:
            raise RuntimeError("Source changed after the independent full-hash review; do not delete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tool-commit", required=True)
    parser.add_argument("--process-check-only", action="store_true")
    parser.add_argument("--delete-verified-source", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if pwd.getpwuid(os.geteuid()).pw_name != "anhntc2":
        raise RuntimeError("Only the donor may check/clean this exact old source")
    if read_git(ROOT, "rev-parse", "HEAD").strip() != args.expected_tool_commit:
        raise RuntimeError("Cleanup requires the tested GitHub tool SHA")
    if read_git(ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Cleanup tool checkout is not clean")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Cleanup never uses/reclaims a GPU")
    for path in (MIGRATION_SOURCE, CLUSTER_ROOT, WORK):
        if path.is_symlink() or path.resolve() != path or not path.is_dir():
            raise RuntimeError("Missing/symlink/aliased source, destination or proof scope")
    if (MIGRATION_SOURCE.stat().st_uid != os.geteuid()
            or CLUSTER_ROOT.stat().st_uid != pwd.getpwnam("hoangnv242").pw_uid):
        raise RuntimeError("Unexpected source/destination ownership")
    processes = process_references(MIGRATION_SOURCE)
    if processes["source_references"]:
        print(json.dumps({"status": "BLOCKED_SOURCE_IN_USE", **processes}), flush=True)
        raise RuntimeError("Live donor process still references MemRec source")
    if args.process_check_only:
        if args.delete_verified_source:
            raise RuntimeError("Diagnostic mode cannot delete")
        print(json.dumps({"status": "DONOR_PROCESS_REFERENCE_CHECK_PASS", **processes}), flush=True)
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    if (not job.isdigit() or subprocess.check_output(["squeue", "-j", job, "-h", "-o", "%u %T"], text=True).strip()
            != "anhntc2 RUNNING"):
        raise RuntimeError("Filesystem cleanup must be in the donor's existing allocation")
    review_path = WORK / "independent-review-20261006-hnv.json"
    if sha256(review_path) != REVIEW_SHA or sha256(WORK / "receipt.json") != RECEIPT_SHA:
        raise RuntimeError("Independent integrity proof changed")
    review = json.loads(review_path.read_text())
    receipt = json.loads((WORK / "receipt.json").read_text())
    if (review["status"] != "FRESH_SOURCE_AND_DESTINATION_FULL_HASH_REVIEW_PASS"
            or review["receipt_sha256"] != RECEIPT_SHA or review["source_deleted"]
            or review["source_root"] != str(MIGRATION_SOURCE) or review["destination_root"] != str(CLUSTER_ROOT)):
        raise RuntimeError("Wrong review/scope or already cleaned source")
    if sha256(WORK / "source-manifest.json") != receipt["source_manifest_sha256"]:
        raise RuntimeError("Source snapshot proof changed")
    donor_path = WORK / "fresh-donor-review-20261006-hnv.json"
    if sha256(donor_path) != review["fresh_donor_manifest_sha256"]:
        raise RuntimeError("Reviewed protected-file checksum proof changed")
    donor = json.loads(donor_path.read_text())
    if set(donor["files"]) != set(FILES):
        raise RuntimeError("Protected source scope differs")
    for name, row in donor["files"].items():
        path = MIGRATION_SOURCE / name
        info = path.lstat()
        if (info.st_size != row["size"] or info.st_mtime_ns != row["mtime_ns"]
                or info.st_mode & 0o7777 != row["mode"] or sha256(path) != row["sha256"]):
            raise RuntimeError("Fresh donor checksum differs; preserve source")
    original = json.loads((WORK / "source-manifest.json").read_text())
    check_unchanged_metadata(MIGRATION_SOURCE, original, review_path.stat().st_mtime_ns)
    if read_git(CLUSTER_ROOT / "repo/MemRec-hnv", "rev-parse", "HEAD").strip() != receipt["destination_git_commit"]:
        raise RuntimeError("Verified destination changed before cleanup")
    final_processes = process_references(MIGRATION_SOURCE)
    if final_processes["source_references"] or not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Active source reference or unsafe recursive deletion implementation")
    if not args.delete_verified_source:
        print(json.dumps({"status": "FINAL_CLEANUP_GATE_PASS_SOURCE_NOT_DELETED", **final_processes}), flush=True)
        return
    # No caller-supplied target, broad variable, glob, job control or symlink
    # traversal. This fd-based deletion can only remove the reviewed old subtree.
    shutil.rmtree(MIGRATION_SOURCE)
    if MIGRATION_SOURCE.exists() or not CLUSTER_ROOT.is_dir():
        raise RuntimeError("Cleanup did not complete or destination missing")
    print(json.dumps({"status": "VERIFIED_OLD_MEMREC_SUBTREE_REMOVED", "source_root": str(MIGRATION_SOURCE),
        "destination_root": str(CLUSTER_ROOT), "source_deleted": True, "destination_preserved": True,
        "recovery_copy": str(CLUSTER_ROOT), "source_logical_bytes": receipt["source_logical_bytes"],
        "source_entries_removed": len(original), "independent_review_sha256": REVIEW_SHA,
        "tool_commit": args.expected_tool_commit, "gpu_requested": False, "jobs_cancelled": False,
        "elapsed_seconds": time.monotonic() - started, **final_processes}), flush=True)


if __name__ == "__main__":
    main()
