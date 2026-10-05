"""Current MemRec cluster scope; no job cancellation or GPU acquisition here."""

from pathlib import Path
import os
import pwd
import subprocess

CLUSTER_OWNER = "hoangnv242"
ALLOCATION_NAME = "senvoice-pro-opt"
CLUSTER_ROOT = Path("/mnt/data/users/hoangnv242/memrec-hnv")
MIGRATION_SOURCE = Path("/mnt/data/users/anhnct/memrec-hnv")


def require_allocation() -> str:
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or pwd.getpwuid(os.geteuid()).pw_name != CLUSTER_OWNER:
        raise RuntimeError("Require the approved owner's existing Slurm allocation")
    actual = subprocess.check_output(["squeue", "-j", job, "-h", "-o", "%u %T %j"], text=True).strip()
    if actual != f"{CLUSTER_OWNER} RUNNING {ALLOCATION_NAME}":
        raise RuntimeError("Unexpected allocation owner/state/name; never cancel or replace it")
    return job


def require_project_root(path: Path) -> Path:
    root = path.resolve()
    if root != CLUSTER_ROOT or path.is_symlink():
        raise RuntimeError("Project root must be the new isolated MemRec workspace")
    return root
