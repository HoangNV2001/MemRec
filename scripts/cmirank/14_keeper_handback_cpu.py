#!/usr/bin/env python3
"""Probe/verify the selected keeper handback; no cancellation or CUDA load."""

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
from src.cmirank.gpu_resources import select_reserved_gpu
from src.cmirank.reserved_handoff import gpu_snapshot, inspect, keeper_names, step_rows, validate_protection


def handback_state(job: str, rows: list[dict], uid: int, *, gpu_index=1,
                   protect_other_workloads=False) -> tuple[str, str | None, str | None]:
    validate_protection(gpu_index, protect_other_workloads)
    selected_name, protected_name = keeper_names(gpu_index)
    protected = [r for r in rows if r.get("Name") == protected_name and r.get("State") == "RUNNING"]
    active = [r for r in rows if r.get("Name") == selected_name and r.get("State") == "RUNNING"]
    if protect_other_workloads:
        protected = []  # Every GPU1 workload is outside our mutation scope.
    elif len(protected) != 1 or protected[0].get("UserId") != str(uid):
        raise ValueError(f"Protected GPU{1 - gpu_index} generator is not uniquely established")
    if len(active) > 1 or any(r.get("UserId") != str(uid) for r in active):
        raise ValueError("Duplicate/foreign generator; do not start another keeper")
    for row in protected + active:
        target = row.get("StepId", "")
        parts = target.split(".")
        if len(parts) != 2 or parts[0] != job or not parts[1].isdigit() or int(parts[1]) == 0:
            raise ValueError("Not current numeric generator steps")
    return ("ALREADY_RUNNING" if active else f"START_GPU{gpu_index}_KEEPER", active[0]["StepId"] if active else None,
            protected[0]["StepId"] if protected else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), default=1)
    parser.add_argument("--protect-other-workloads", action="store_true")
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not args.run_dir.resolve().is_relative_to(private / "runs"):
        raise ValueError("CPU-only scoped handback required")
    run = args.run_dir
    rows = step_rows(subprocess.check_output(["scontrol", "show", "step", job, "-o"], text=True))
    state, active, protected = handback_state(job, rows, os.geteuid(), gpu_index=args.gpu_index,
                                             protect_other_workloads=args.protect_other_workloads)
    if not args.verify:
        if state == f"START_GPU{args.gpu_index}_KEEPER":
            cards, apps, _, _ = gpu_snapshot()
            card = select_reserved_gpu(cards, set(apps), gpu_index=args.gpu_index)
            wrapper = Path("/mnt/data/users/hoangnv242/omni-gen-hnv/omni_gen.sh")
            if not wrapper.is_file() or wrapper.with_name("STOP").exists():
                raise ValueError("Approved wrapper missing/shared STOP set; do not modify either")
            # Do not start while another workload owns the selected card's VRAM/PID.
            evidence = {"status": state, "gpu_uuid": card.uuid, "protected_step": protected, "job": job}
        else:
            # Existing keeper must match current UID/cgroup/command/device guards.
            proof = inspect(job, active, gpu_index=args.gpu_index, protect_other_workloads=args.protect_other_workloads)
            evidence = {"status": state, "generator_step": active, "protected_step": protected,
                        "job": job, "gpu_uuid": proof[f"gpu{args.gpu_index}_uuid"], "pid_mapping": "NOT_OBSERVED"}
        if run.is_dir():
            (run / "keeper-probe.json").write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n")
        print(json.dumps(evidence, sort_keys=True), flush=True)
        return
    # Bounded startup readiness, in this one server-side SSH/Slurm command.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        rows = step_rows(subprocess.check_output(["scontrol", "show", "step", job, "-o"], text=True))
        state, active, protected = handback_state(job, rows, os.geteuid(), gpu_index=args.gpu_index,
                                                 protect_other_workloads=args.protect_other_workloads)
        if active:
            try:
                proof = inspect(job, active, gpu_index=args.gpu_index, protect_other_workloads=args.protect_other_workloads)
                cards, apps, _, _ = gpu_snapshot()
                card = next(c for c in cards if c.index == args.gpu_index)
                if card.used_mib > 512 and len(apps.get(card.uuid, set())) == 1:
                    evidence = {"status": f"KEEPER_GPU{args.gpu_index}_HANDBACK_VERIFIED", "job": job,
                        "generator_step": active, "protected_step": protected, "gpu_uuid": card.uuid,
                        "gpu_used_mib": card.used_mib, "gpu_utilization_percent": card.utilization,
                        "members": proof["members"], "pid_mapping": "NOT_OBSERVED",
                        "protected_gpu_uuid": proof["protected_gpu_uuid"],
                        "protected_gpu_pids": proof["protected_gpu_pids"],
                        "protect_other_workloads": args.protect_other_workloads,
                        "reserved_job_cancelled": False, f"gpu{1 - args.gpu_index}_signalled": False}
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
