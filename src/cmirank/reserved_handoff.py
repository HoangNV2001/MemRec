"""One explicitly owner-attested GPU1 generator handoff; NEVER cancel a job.

NVML host PIDs may be invisible in the container's /proc. In that case this
path records the owner's CURRENT exclusive-step confirmation, not a fictional
PID mapping. All visible Slurm task UID/cgroup/commands are still verified.
"""

import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import time

from src.cluster_runtime import CLUSTER_OWNER, require_allocation
from src.cmirank.gpu_resources import compute_gpu_processes, parse_gpu_snapshot, select_reserved_gpu1

GENERATOR_ROOT = Path("/mnt/data/users/hoangnv242/omni-gen-hnv")


def step_rows(raw):
    rows = []
    for line in raw.splitlines():
        row = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=([^\s]+)", line))
        if row:
            rows.append(row)
    return rows


def confirmed_target(job, target, rows, uid):
    if not re.fullmatch(re.escape(job) + r"\.[1-9][0-9]*", target):
        raise ValueError("Confirmation must name one current numeric nonzero step, never a job/batch/extern")
    generators = [r for r in rows if r.get("Name") == "omni-gen-1"]
    if (len(generators) != 1 or generators[0].get("StepId") != target
            or generators[0].get("State") != "RUNNING" or generators[0].get("UserId") != str(uid)):
        raise ValueError("Current GPU1 generator differs from the owner-confirmed step")
    protected = [r for r in rows if r.get("Name") == "omni-gen-0"]
    if (len(protected) != 1 or protected[0].get("State") != "RUNNING"
            or protected[0].get("UserId") != str(uid) or protected[0].get("StepId") == target):
        raise ValueError("Cannot establish the protected GPU0 generator")
    return protected[0]["StepId"]


def validate_task(job, step, uid, argv, cgroup, env):
    expected_group = rf"/job_{re.escape(job)}/step_{re.escape(step)}(?:/|$)"
    if (uid != pwd.getpwnam(CLUSTER_OWNER).pw_uid or not re.search(expected_group, cgroup)
            or env.get("SLURM_JOB_ID") != job or env.get("SLURM_STEP_ID") != step
            or env.get("GPU") != "1"):
        raise ValueError("Generator task UID/cgroup/device/allocation does not match")
    if len(argv) < 2:
        raise ValueError("Missing generator command")
    if (argv[0] in ("bash", "/bin/bash", "/usr/bin/bash")
            and argv[1] == str(GENERATOR_ROOT / "omni_gen.sh")):
        # /proc/environ records the shell's initial reservation mask. The
        # actual CUDA actor is its Python child, which MUST expose only GPU1.
        if env.get("CUDA_VISIBLE_DEVICES") not in ("1", "0,1"):
            raise ValueError("Unexpected bootstrap mask for the approved GPU1 wrapper")
        return "wrapper"
    if (argv[0] == "/mnt/data/users/hoangnv242/envs/omnidistill/bin/python"
            and argv[1] == str(GENERATOR_ROOT / "omni_gen.py")):
        if env.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ValueError("Generator Python must expose only GPU1")
        return "python"
    raise ValueError("Unexpected task in the generator step; do not stop it")


def gpu_snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader,nounits"], text=True)
    return parse_gpu_snapshot(raw), compute_gpu_processes(apps), raw, apps


def inspect(job, target):
    uid = os.geteuid()
    steps = subprocess.check_output(["scontrol", "show", "step", job, "-o"], text=True)
    protected = confirmed_target(job, target, step_rows(steps), uid)
    step = target.split(".")[1]
    pids_text = subprocess.check_output(["scontrol", "listpids", target], text=True)
    members = []
    for line in pids_text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit() or fields[1:3] != [job, step]:
            raise ValueError("Unexpected Slurm step PID record")
        path = Path("/proc") / fields[0]
        task_uid = path.stat().st_uid
        argv = [s.decode() for s in (path / "cmdline").read_bytes().split(b"\0") if s]
        cgroup = (path / "cgroup").read_text()
        if task_uid == 0:
            if argv != [f"slurmstepd: [{target}] "] and " ".join(argv).strip() != f"slurmstepd: [{target}]":
                raise ValueError("Unknown privileged process in the generator step")
            continue  # Slurm infrastructure is NEVER directly signalled.
        allowed_env = {b"SLURM_JOB_ID", b"SLURM_STEP_ID", b"CUDA_VISIBLE_DEVICES", b"GPU"}
        environment = dict(part.decode().split("=", 1) for part in (path / "environ").read_bytes().split(b"\0")
                           if b"=" in part and part.split(b"=", 1)[0] in allowed_env)
        role = validate_task(job, step, task_uid, argv, cgroup, environment)
        start_ticks = (path / "stat").read_text().rsplit(") ", 1)[1].split()[19]
        members.append({"pid": int(path.name), "uid": task_uid, "role": role, "argv": argv,
                        "cgroup": cgroup, "start_ticks": start_ticks, "device_environment": environment})
    if sorted(r["role"] for r in members) != ["python", "wrapper"]:
        raise ValueError("Need exactly the approved wrapper and its generator Python")
    cards, apps, raw_cards, raw_apps = gpu_snapshot()
    one = [c for c in cards if c.index == 1]
    zero = [c for c in cards if c.index == 0]
    if (len(one) != 1 or len(zero) != 1 or "H100" not in one[0].name or one[0].total_mib < 80000
            or len(apps.get(one[0].uuid, set())) != 1
            or apps.get(one[0].uuid, set()) & apps.get(zero[0].uuid, set())):
        raise ValueError("GPU1 snapshot exceeds the owner-confirmed single-generator scope")
    return {"job": job, "target": target, "protected_step": protected, "gpu1_uuid": one[0].uuid,
            "nvml_gpu1_pids": sorted(apps[one[0].uuid]), "members": members,
            "steps": steps, "gpu_snapshot": raw_cards, "compute_snapshot": raw_apps,
            "pid_mapping": "NOT_OBSERVED_OWNER_ATTESTED_CURRENT_STEP_EXCLUSIVITY"}


def handoff(run, owner_confirmed_step):
    """Call ONLY after CPU preparation, immediately before real model compute."""
    job = require_allocation()
    if not re.fullmatch(re.escape(job) + r"\.[1-9][0-9]*", owner_confirmed_step):
        raise ValueError("Never cancel a job/batch/extern/step0 or a different allocation")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Handoff coordinator must not initialize CUDA")
    first = inspect(job, owner_confirmed_step)
    second = inspect(job, owner_confirmed_step)
    identity = ("target", "protected_step", "gpu1_uuid", "nvml_gpu1_pids", "members")
    if any(first[k] != second[k] for k in identity):
        raise RuntimeError("Generator identity changed during preflight; require a fresh handoff")
    record = {**second, "owner_confirmation": "Fresh researcher confirmation: GPU1 only runs this omni-gen-1 step",
              "step_cancelled": False, "reserved_job_cancelled": False, "gpu0_signalled": False}
    path = run / "gpu1-handoff.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    require_allocation()
    # One positional numeric job.step only: NEVER job-only, --user/name/all,
    # batch/extern, GPU0, NVML PID, shared STOP file or generator restart.
    subprocess.run(["scancel", owner_confirmed_step], check=True)
    record["step_cancelled"] = True
    path.write_text(json.dumps(record, indent=2) + "\n")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        cards, apps, raw_cards, raw_apps = gpu_snapshot()
        try:
            card = select_reserved_gpu1(cards, set(apps))
        except ValueError:
            time.sleep(2)  # Bounded handoff drain, not experiment result polling.
            continue
        rows = step_rows(subprocess.check_output(["scontrol", "show", "step", job, "-o"], text=True))
        if any(r.get("StepId") == owner_confirmed_step and r.get("State") == "RUNNING" for r in rows):
            time.sleep(2)
            continue
        protected = [r for r in rows if r.get("StepId") == record["protected_step"]]
        if (card.uuid != record["gpu1_uuid"] or len(protected) != 1 or protected[0].get("State") != "RUNNING"
                or protected[0].get("Name") != "omni-gen-0" or protected[0].get("UserId") != str(os.geteuid())):
            raise RuntimeError("GPU identity/protected generator changed; do not load a model")
        require_allocation()
        record.update(status="OWNER_CONFIRMED_STEP_ONLY_HANDOFF_PASS", gpu1_idle=True,
                      gpu_snapshot_after=raw_cards, compute_snapshot_after=raw_apps)
        path.write_text(json.dumps(record, indent=2) + "\n")
        return record
    record.update(status="HANDOFF_DRAIN_FAILED_NO_MODEL_LOAD", gpu1_idle=False)
    path.write_text(json.dumps(record, indent=2) + "\n")
    raise RuntimeError("GPU1 did not drain; no further cancellation or model load")
