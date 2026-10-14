#!/usr/bin/env python3
"""CPU coordinator owns one native actor worker; original controller restores keeper1."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.optimizer_smoke import (allocation_memory, load_optimizer_config, verify_optimizer_preparation)
from src.cmirank.corrected_rollout import load_corrected_config
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.ppo_parity import verify_fla_overlay, wait_owned_hf_release
from src.cmirank.cuda_toolchain import verify_cuda_toolchain
from src.cmirank.causal_conv import verify_causal_overlay
from src.cmirank.gpu_resources import canonical_gpu_uuid, select_reserved_gpu, verified_numeric_cuda_binding
from src.cmirank.reserved_handoff import gpu_snapshot, handoff
from src.cmirank.provenance import file_sha256
_spec = spec_from_file_location("memrec_owned_optimizer_runtime", ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py")
_owned = module_from_spec(_spec)
_spec.loader.exec_module(_owned)
save, append, stop_owned = _owned.save, _owned.append, _owned.stop_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner-confirmed-generator-step", required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_optimizer_config(ROOT)
    native, _ = load_corrected_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / c["run_id"] or run.exists()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or not os.environ.get("MEMREC_KEEPER_AUTHORITY_PATH")):
        raise ValueError("Fresh clean-source GPU1 standing-authority coordinator required")
    prepared = verify_optimizer_preparation(private, ROOT, c, path, commit)
    verify_fla_overlay(private, native)
    toolchain = verify_cuda_toolchain(private, ROOT, native)[0]
    causal = verify_causal_overlay(private, ROOT, native)[0]
    checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
    for n, stat in prepared["checkpoint_stat"].items():
        s = (checkpoint/n).stat()
        if {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino} != stat:
            raise ValueError("Initializer changed since CPU hashing")
    for n, sha in prepared["native_source_sha256"].items():
        if file_sha256(Path(sys.prefix)/n) != sha:
            raise ValueError("Native source changed since CPU preparation")
    for n, sha in prepared["overlay_sha256"].items():
        if file_sha256(private / "overlays" / compat["overlay_name"] / n) != sha:
            raise ValueError("FA2 changed since CPU preparation")
    host = allocation_memory(job)
    if host["maximum_bytes"]-host["current_bytes"] < c["host_load_reserve_bytes"]:
        raise ValueError("Own allocation host-load margin disappeared; no handoff")
    run.mkdir()
    worker = card = log = None
    exit_code = 1
    began = time.monotonic()
    marker = "scripts/cmirank/34_native_actor_optimizer_gpu.py"
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned actor task interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    manifest = {"source_commit": commit, "config": c, "config_sha256": file_sha256(path),
        "slurm_job_id": job, "allocation_memory_before": host,
        "preparation_report_sha256": file_sha256(private / "runs" / c["prepare_run_id"] / "report.json"),
        "physical_requests": 0, "training_ready": False}
    save(run / "manifest.json", manifest)
    try:
        manifest["gpu_handoff"] = handoff(run, args.owner_confirmed_generator_step,
            gpu_index=1, protect_other_workloads=True, task_timeout_seconds=c["timeout_minutes"]*60)
        cards, apps, _, _ = gpu_snapshot()
        card = select_reserved_gpu(cards, set(apps), gpu_index=1)
        if card.uuid != manifest["gpu_handoff"]["gpu1_uuid"]:
            raise ValueError("Authorized physical GPU changed")
        cards, apps, _, _ = gpu_snapshot()
        if select_reserved_gpu(cards, set(apps), gpu_index=1).uuid != card.uuid:
            raise ValueError("GPU1 no longer idle at load time")
        pci = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"], text=True)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = verified_numeric_cuda_binding(cards, card, pci)
        os.environ["MEMREC_SELECTED_GPU_UUID"] = card.uuid
        identity = json.loads(subprocess.check_output([sys.executable, "-c",
            "import json,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; "
            "print(json.dumps({'uuid':str(torch.cuda.get_device_properties(0).uuid),'devices':1}))"], text=True))
        if canonical_gpu_uuid(identity["uuid"]) != canonical_gpu_uuid(card.uuid):
            raise ValueError("Native actor CUDA binding is not GPU1")
        save(run / "cuda-check-release.json", wait_owned_hf_release(gpu_snapshot, native, card.uuid,
            child_exited=True, child_kind="cuda_identity_check", emit=lambda x: append(run, "cuda-check-release-snapshots", x)))
        manifest.update(gpu_uuid=card.uuid, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], cuda_identity_check=identity)
        save(run / "manifest.json", manifest)
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(causal),
            str(private / "overlays" / native["hf_kernel_overlay"]), os.environ["PYTHONPATH"])),
            "CUDA_HOME": str(toolchain), "CUDA_PATH": str(toolchain),
            "PATH": str(toolchain / "bin") + os.pathsep + os.environ["PATH"], "FLA_TILELANG": "1",
            "TILELANG_CACHE_DIR": str(private / "cache/runtime-hnv" / c["run_id"] / "tilelang"),
            "TILELANG_TMP_DIR": str(private / "cache/runtime-hnv" / c["run_id"] / "tilelang-tmp")}
        log = (run / "native-worker.log").open("x")
        worker = subprocess.Popen([sys.executable, "-u", marker, "--run-dir", str(run)], env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if worker.wait(timeout=c["worker_timeout_seconds"]) != 0:
            raise RuntimeError("Owned native actor optimizer failed; no automatic profile search")
        save(run / "native-release.json", wait_owned_hf_release(gpu_snapshot, native, card.uuid,
            child_exited=True, child_kind="native_actor_optimizer", emit=lambda x: append(run, "native-release-snapshots", x)))
        report = json.loads((run / "native-worker-report.json").read_text())
        if (report["status"] != "NATIVE_ACTOR_OPTIMIZER_ONE_UPDATE_PASS_CRITIC_MULTITURN_CHECKPOINT_GATES_REMAIN"
                or report["source_commit"] != commit or report["optimizer_updates"] != 1
                or report["samples"] != 20 or report["training_ready"]):
            raise ValueError("Native actor worker scope/result differs")
        save(run / "report.json", {**report, "worker_report_sha256": file_sha256(run / "native-worker-report.json"),
            "config_sha256": file_sha256(path), "elapsed_seconds": time.monotonic()-began})
        exit_code = 0
    except BaseException as error:
        save(run / "failure.json", {"error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.monotonic()-began})
        raise
    finally:
        stop_owned(worker, marker)
        if log:
            log.close()
        save(run / "child-cleanup.json", {"process_exit_code": exit_code, "gpu_uuid": card.uuid if card else None,
            "owned_hf_worker_exited": worker is None or worker.poll() is not None,
            "training_ready": False, "update_discarded": True})


if __name__ == "__main__":
    main()
