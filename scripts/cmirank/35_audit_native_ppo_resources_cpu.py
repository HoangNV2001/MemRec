#!/usr/bin/env python3
"""Read-only native layout/limits audit after the two role optimizer passes.

No OOM experiment, GPU handoff, model load, checkpoint I/O or resource change.
This does not prove impossibility of all one-card optimizations.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.optimizer_smoke import allocation_memory, standard_ppo_memory_budget
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private / "runs/cmirank-ppo-resource-audit-v1-20261015-hnv" or run.exists()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Fresh exact-source allocation-only CPU memory audit required")
    parents = [("cmirank-actor-optimizer-smoke-v2-20261014-hnv",
        "d2951f12d4f24d8c00ef3449aeee754c0834599e34d83e27581289a903b27ebe"),
        ("cmirank-critic-optimizer-smoke-v1-20261014-hnv",
         "e0a82a6e798550e03fd43488c52ba0d56c376dec059b4bb6e0624bd3f3fa5ff7")]
    reports, hashes = [], {}
    for rid, sha in parents:
        p = private / "runs" / rid
        if file_sha256(p / "report.json") != sha:
            raise ValueError("Independently reviewed role report changed")
        r = json.loads((p / "report.json").read_text())
        cleanup = json.loads((p / "cleanup.json").read_text())
        handback = json.loads((p / "keeper-handback.json").read_text())
        if (file_sha256(p / "native-worker-report.json") != r["worker_report_sha256"]
                or cleanup["process_exit_code"] or not cleanup["gpu_released"]
                or not cleanup["allocation_still_running"] or not cleanup["child_exited"]
                or handback["status"] != "KEEPER_GPU1_HANDBACK_VERIFIED"
                or handback["gpu0_signalled"] or handback["reserved_job_cancelled"]):
            raise ValueError("Successful native role plus actual release/keeper restoration required")
        reports.append(r)
        hashes[rid] = sha
    memory = allocation_memory(job)
    line = next(x for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    p = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")
    while p.name != f"job_{job}":
        if p == p.parent or p == Path("/sys/fs/cgroup"):
            raise ValueError("Own allocation cgroup unavailable")
        p = p.parent
    stats = dict(x.split() for x in (p / "memory.stat").read_text().splitlines())
    total_mib = int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.total",
        "--format=csv,noheader,nounits"], text=True).strip())
    budget = standard_ppo_memory_budget(*reports, host_bytes=memory["maximum_bytes"],
        gpu_bytes=total_mib*2**20, baseline_anon_bytes=int(stats["anon"]))
    native = Path(sys.prefix) / "lib/python3.10/site-packages/verl"
    native_sources = ("workers/engine/base.py", "workers/engine/fsdp/transformer_impl.py",
        "utils/fsdp_utils.py", "utils/checkpoint/fsdp_checkpoint_manager.py")
    result = {"status": "CPU_NATIVE_LAYOUT_AUDIT_REQUIRES_RESOURCE_CHOICE" if budget["all_checked_layouts_unsafe"]
            else "CPU_NATIVE_LAYOUT_AUDIT_MEASUREMENT_GATES_REMAIN",
        "source_commit": commit, "parent_report_sha256": hashes, "allocation_memory": memory,
        "own_allocation_anonymous_bytes": int(stats["anon"]), "physical_gpu_total_mib": total_mib,
        "budget": budget, "native_source_sha256": {n: file_sha256(native/n) for n in native_sources},
        "audit_source_sha256": {n: file_sha256(ROOT/n) for n in
            ("scripts/cmirank/35_audit_native_ppo_resources_cpu.py", "src/cmirank/optimizer_smoke.py")},
        "gpu_requested": False, "full_weights_loaded": False, "optimizer_updates": 0,
        "resources_changed": False, "training_ready": False, "keeper_stopped": False}
    run.mkdir()
    (run / "report.json").write_text(artifact_json_dumps(result))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
