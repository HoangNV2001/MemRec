#!/usr/bin/env python3
"""20 actual stochastic responses, then native FSDP2 scoring; one guarded GPU1."""

import argparse
import importlib.metadata
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.corrected_rollout import (corrected_server_options, generated_observation,
    generated_request, load_corrected_config, verify_corrected_preparation)
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_parity import verify_fla_overlay, wait_owned_hf_release, completion_request, response_prompt_logprobs
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.cuda_toolchain import verify_cuda_toolchain
from src.cmirank.causal_conv import verify_causal_overlay
from src.cmirank.gpu_resources import canonical_gpu_uuid, select_reserved_gpu, verified_numeric_cuda_binding
from src.cmirank.parser import parse_action
from src.cmirank.provenance import file_sha256
from src.cmirank.reserved_handoff import gpu_snapshot, handoff
_spec = spec_from_file_location("memrec_owned_parity_runtime", ROOT / "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py")
_owned_runtime = module_from_spec(_spec)
_spec.loader.exec_module(_owned_runtime)
append, save, stop_owned = _owned_runtime.append, _owned_runtime.save, _owned_runtime.stop_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner-confirmed-generator-step", required=True)
    parser.add_argument("--parameterization", choices=("full", "lora"), default="full")
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_corrected_config(ROOT)
    if args.parameterization == "lora":
        from src.cmirank.lora_rollout import (load_lora_rollout_config, verify_lora_rollout_preparation,
            lora_server_options, rollout_host_budget)
        from src.cmirank.lora_checkpoint import checkpoint_budget
        c, path = load_lora_rollout_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private / "runs" / c["run_id"] or run.exists()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or not os.environ.get("MEMREC_KEEPER_AUTHORITY_PATH")
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]):
        raise ValueError("Fresh exact-source GPU1 standing-authority coordinator required")
    for n, v in runtime["versions"].items():
        if importlib.metadata.version(n) != v:
            raise ValueError("Pinned runtime changed")
    rows, prepared = (verify_lora_rollout_preparation(private, ROOT, c, path, commit)
        if args.parameterization == "lora" else verify_corrected_preparation(private, ROOT, c, path, commit))
    verify_fla_overlay(private, c)
    toolchain = verify_cuda_toolchain(private, ROOT, c)[0]
    causal = verify_causal_overlay(private, ROOT, c)[0]
    checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
    for n, expected in prepared["checkpoint_stat"].items():
        s = (checkpoint / n).stat()
        if {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino} != expected:
            raise ValueError("Checkpoint changed after CPU hashing")
    for n, sha in prepared["native_source_sha256"].items():
        if file_sha256(Path(sys.prefix) / n) != sha:
            raise ValueError("Native source changed after CPU qualification")
    for n, sha in prepared["overlay_sha256"].items():
        if file_sha256(private / "overlays" / compat["overlay_name"] / n) != sha:
            raise ValueError("FA2 changed after CPU qualification")
    if args.parameterization == "lora":
        import shutil
        host = rollout_host_budget(job, c)
        previous_budget = prepared["checkpoint_budget"]
        # Reuse the conservatively measured checkpoint-role RSS allowance from
        # CPU admission, with the live anonymous allocation footprint.
        fixed_rss = (previous_budget["host_required_bytes"]
            -prepared["host_budget"]["stats_bytes"]["anon"]
            -previous_budget["largest_role_staging_bytes"]-c["checkpoint_host_workspace_bytes"])
        estimate = checkpoint_budget(prepared["roles"], previous_budget["gpu_bytes"],
            host["allocation_memory"]["maximum_bytes"], host["stats_bytes"]["anon"], fixed_rss,
            shutil.disk_usage(private/"runs").free, c)
        if not host["assessment"]["within_estimated_budget"] or not estimate["within_estimated_budget"]:
            raise ValueError("Sequential sampler/native restore budget disappeared; no handoff")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", c["port"]))
    run.mkdir()
    server = worker = card = server_log = worker_log = None
    exit_code, physical = 1, 0
    began = time.monotonic()
    worker_marker = "scripts/cmirank/30_native_logprob_worker_gpu.py"
    if args.parameterization == "lora":
        worker_marker = "scripts/cmirank/40_native_lora_rollout_gpu.py"
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned corrected-rollout task interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    manifest = {"source_commit": commit, "config": c, "config_sha256": file_sha256(path),
        "slurm_job_id": job, "preparation_report_sha256": file_sha256(private / "runs" / c["prepare_run_id"] / "report.json"),
        "optimizer_updates": 0, "training_ready": False,
        "compiler_cache_paths": {k: os.environ.get(k) for k in
            ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH",
             "FLASHINFER_WORKSPACE_BASE", "DG_JIT_CACHE_DIR")}}
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
        manifest.update(gpu_uuid=card.uuid, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
        # Fresh short-lived child validates full physical CUDA UUID BEFORE sampler weights.
        identity = json.loads(subprocess.check_output([sys.executable, "-c",
            "import json,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; "
            "print(json.dumps({'uuid':str(torch.cuda.get_device_properties(0).uuid),'devices':1}))"], text=True))
        if canonical_gpu_uuid(identity["uuid"]) != canonical_gpu_uuid(card.uuid):
            raise ValueError("Sampler CUDA binding is not the authorized physical GPU1")
        manifest["cuda_identity_check"] = identity
        save(run / "cuda-check-release.json", wait_owned_hf_release(gpu_snapshot, c, card.uuid,
            child_exited=True, child_kind="cuda_identity_check", emit=lambda x: append(run, "cuda-check-release-snapshots", x)))
        save(run / "manifest.json", manifest)
        server_log = (run / "vllm-server.log").open("x")
        environment = {**os.environ, **c["server_environment"]}
        server = subprocess.Popen([str(private / "envs" / runtime["environment_name"] / "bin/vllm"),
            "serve", str(checkpoint), *(lora_server_options(c, private/"runs"/c["prepare_run_id"]/"actor-adapter")
                if args.parameterization == "lora" else corrected_server_options(c))], env=environment,
            stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        (run / "server.pid").write_text(str(server.pid) + "\n")
        base = f"http://127.0.0.1:{c['port']}"
        deadline = time.monotonic() + c["server_startup_timeout_seconds"]
        while True:
            if server.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Owned sampler failed readiness or timed out")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(5)
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
            expected_models = {c["served_model_name"]}
            if args.parameterization == "lora":
                expected_models.add(c["served_adapter_name"])
            if {m["id"] for m in json.load(response)["data"]} != expected_models:
                raise ValueError("Local server identity differs")
        if args.parameterization == "lora":
            for model in (c["served_model_name"], c["served_adapter_name"]):
                for row in rows:
                    if physical >= c["physical_request_cap"]:
                        raise ValueError("Fixed60 request budget exhausted; no retries")
                    payload = {**completion_request(row, c), "model": model}
                    physical += 1
                    append(run, "physical-requests", {"event": "request", "attempt": physical,
                        "user_id": row["user_id"], "phase": "forced-control", "payload": payload})
                    request = urllib.request.Request(base+"/v1/completions", data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=c["request_timeout_seconds"]) as response:
                        raw = json.load(response)
                    append(run, "physical-requests", {"event": "response", "attempt": physical,
                        "user_id": row["user_id"], "phase": "forced-control", "response": raw})
                    append(run, "sampler-controls", {"model": model, "user_id": row["user_id"],
                        "logprobs": response_prompt_logprobs(raw, row)})
        observations = []
        for row in rows:
            if physical >= c["physical_request_cap"]:
                raise ValueError("Fixed20 request cap exhausted; no retry")
            payload = generated_request(row, c)
            if args.parameterization == "lora":
                payload["model"] = c["served_adapter_name"]
            physical += 1
            append(run, "physical-requests", {"event": "request", "attempt": physical, "user_id": row["user_id"], "payload": payload})
            request = urllib.request.Request(base + "/v1/completions", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=c["request_timeout_seconds"]) as response:
                raw = json.load(response)
            append(run, "physical-requests", {"event": "response", "attempt": physical, "user_id": row["user_id"], "response": raw})
            obs = generated_observation(raw, row, c, prepared["eos_token_id"])
            parsed = parse_action(obs["raw_text"], [f"C{i:02d}" for i in range(10)])
            obs.update(format_valid=parsed.valid, format_failure=parsed.failure_reason)
            append(run, "generated", obs)
            observations.append(obs)
        # No model remains loaded during CPU preparation or native scorer loading.
        stop_owned(server, str(checkpoint))
        save(run / "server-release.json", wait_owned_hf_release(gpu_snapshot, c, card.uuid,
            child_exited=server.poll() is not None, child_kind="vllm", emit=lambda x: append(run, "server-release-snapshots", x)))
        manifest["generated_sha256"] = file_sha256(run / "generated.jsonl")
        save(run / "manifest.json", manifest)
        worker_environment = {**os.environ,
            "PYTHONPATH": os.pathsep.join((str(causal), str(private / "overlays" / c["hf_kernel_overlay"]), os.environ["PYTHONPATH"])),
            "CUDA_HOME": str(toolchain), "CUDA_PATH": str(toolchain),
            "PATH": str(toolchain / "bin") + os.pathsep + os.environ["PATH"], "FLA_TILELANG": "1",
            "TILELANG_CACHE_DIR": str(private / "cache/runtime-hnv" / c["run_id"] / "tilelang"),
            "TILELANG_TMP_DIR": str(private / "cache/runtime-hnv" / c["run_id"] / "tilelang-tmp")}
        worker_log = (run / "native-worker.log").open("x")
        worker = subprocess.Popen([sys.executable, "-u", worker_marker, "--run-dir", str(run)],
            env=worker_environment, stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
        if worker.wait(timeout=600) != 0:
            raise RuntimeError("Owned native FSDP2 scorer failed; no optimizer promotion")
        save(run / "native-release.json", wait_owned_hf_release(gpu_snapshot, c, card.uuid,
            child_exited=worker.poll() == 0, child_kind="native_fsdp2", emit=lambda x: append(run, "native-release-snapshots", x)))
        worker_report = json.loads((run/"native-worker-report.json").read_text())
        if args.parameterization == "lora" and (worker_report["status"] != c["worker_status"]
                or worker_report["source_commit"] != commit or not worker_report["cold_process_restart_proven"]
                or worker_report["optimizer_updates"] or physical != 60):
            raise ValueError("Native cold-restore/reference or physical-request scope differs")
        save(run / "report.json", {"status": ("LORA_ROLLOUT_NATIVE_ENGINE_PASS_CPU_CORRECTION_REVIEW_REQUIRED"
            if args.parameterization == "lora" else "GENERATED_ROLLOUT_NATIVE_ENGINE_PASS_CPU_CORRECTION_REVIEW_REQUIRED"),
            "source_commit": commit, "config_sha256": file_sha256(path), "samples": 20,
            "physical_requests": physical, "generated_response_tokens": sum(len(x["response_token_ids"]) for x in observations),
            "format_valid_samples": sum(x["format_valid"] for x in observations),
            "truncated_samples": sum(x["finish_reason"] == "length" for x in observations),
            "optimizer_updates": 0, "training_ready": False, "ranking_metrics_computed": False,
            "books_outcomes_accessed": False, "old_strict_cross_engine_gate_status": "FAIL_UNCHANGED",
            "parameterization": args.parameterization,
            "worker_report_sha256": file_sha256(run/"native-worker-report.json"),
            "elapsed_seconds": time.monotonic()-began})
        exit_code = 0
    except BaseException as error:
        save(run / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "physical_requests": physical, "elapsed_seconds": time.monotonic()-began})
        raise
    finally:
        stop_owned(server, str(checkpoint))
        stop_owned(worker, worker_marker)
        if server_log:
            server_log.close()
        if worker_log:
            worker_log.close()
        save(run / "child-cleanup.json", {"process_exit_code": exit_code, "gpu_uuid": card.uuid if card else None,
            "owned_server_exited": server is None or server.poll() is not None,
            "owned_hf_worker_exited": worker is None or worker.poll() is not None,
            "optimizer_updates": 0, "training_ready": False})


if __name__ == "__main__":
    main()
