#!/usr/bin/env python3
"""Probe/verify mandatory GPU1 keeper handback; no cancellation or CUDA load."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.gpu_resources import select_reserved_gpu1
from src.cmirank.reserved_handoff import gpu_snapshot, inspect, step_rows


def handback_state(job: str, rows: list[dict], uid: int) -> tuple[str, str | None, str]:
    protected = [r for r in rows if r.get("Name") == "omni-gen-0" and r.get("State") == "RUNNING"]
    active = [r for r in rows if r.get("Name") == "omni-gen-1" and r.get("State") == "RUNNING"]
    if len(protected) != 1 or protected[0].get("UserId") != str(uid):
        raise ValueError("Protected GPU0 generator is not uniquely established")
    if len(active) > 1 or any(r.get("UserId") != str(uid) for r in active):
        raise ValueError("Duplicate/foreign generator; do not start another keeper")
    for row in protected + active:
        target = row.get("StepId", "")
        parts = target.split(".")
        if len(parts) != 2 or parts[0] != job or not parts[1].isdigit() or int(parts[1]) == 0:
            raise ValueError("Not current numeric generator steps")
    return ("ALREADY_RUNNING" if active else "START_GPU1_KEEPER", active[0]["StepId"] if active else None,
            protected[0]["StepId"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not args.run_dir.resolve().is_relative_to(private / "runs"):
        raise ValueError("CPU-only scoped handback required")
    run = args.run_dir
    rows = step_rows(subprocess.check_output(["scontrol", "show", "step", job, "-o"], text=True))
    state, active, protected = handback_state(job, rows, os.geteuid())
    if not args.verify:
        if state == "START_GPU1_KEEPER":
            cards, apps, _, _ = gpu_snapshot()
            card = select_reserved_gpu1(cards, set(apps))
            wrapper = Path("/mnt/data/users/hoangnv242/omni-gen-hnv/omni_gen.sh")
            if not wrapper.is_file() or wrapper.with_name("STOP").exists():
                raise ValueError("Approved wrapper missing/shared STOP set; do not modify either")
            # Do not start while another workload owns any physical GPU1 VRAM/PID.
            evidence = {"status": state, "gpu_uuid": card.uuid, "protected_step": protected, "job": job}
        else:
            # Existing keeper must match current UID/cgroup/command/device guards.
            proof = inspect(job, active)
            evidence = {"status": state, "generator_step": active, "protected_step": protected,
                        "job": job, "gpu_uuid": proof["gpu1_uuid"], "pid_mapping": "NOT_OBSERVED"}
        if run.is_dir():
            (run / "keeper-probe.json").write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n")
        print(json.dumps(evidence, sort_keys=True), flush=True)
        return
    # Bounded startup readiness, in this one server-side SSH/Slurm command.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        rows = step_rows(subprocess.check_output(["scontrol", "show", "step", job, "-o"], text=True))
        state, active, protected = handback_state(job, rows, os.geteuid())
        if active:
            try:
                proof = inspect(job, active)
                cards, apps, _, _ = gpu_snapshot()
                card = next(c for c in cards if c.index == 1)
                if card.used_mib > 512 and len(apps.get(card.uuid, set())) == 1:
                    evidence = {"status": "KEEPER_GPU1_HANDBACK_VERIFIED", "job": job,
                        "generator_step": active, "protected_step": protected, "gpu_uuid": card.uuid,
                        "gpu_used_mib": card.used_mib, "gpu_utilization_percent": card.utilization,
                        "members": proof["members"], "pid_mapping": "NOT_OBSERVED",
                        "reserved_job_cancelled": False, "gpu0_signalled": False}
                    if run.is_dir():
                        (run / "keeper-handback.json").write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n")
                    print(json.dumps(evidence, sort_keys=True), flush=True)
                    return
            except (ValueError, FileNotFoundError):
                pass  # Step/its Python may still be starting; never change its code or kill it.
        time.sleep(3)
    raise RuntimeError("Keeper startup not verified within 90 s; do not kill job or foreign PID")


if __name__ == "__main__":
    main()
