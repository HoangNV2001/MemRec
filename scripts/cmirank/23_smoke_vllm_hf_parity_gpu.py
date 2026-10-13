#!/usr/bin/env python3
"""Twenty raw-token vLLM/HF parity probes; CPU coordinator, one authorized GPU.

HF child exits/release-audits before vLLM starts. No held model waits for CPU
preparation or researcher input. Outer controller restores the authorized keeper.
"""

import argparse
import importlib.metadata
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
from src.cmirank.gpu_resources import select_reserved_gpu, verified_numeric_cuda_binding
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_parity import (compare_logprobs, completion_request, diagnostic_parent, load_parity_config,
                                    parity_code_hashes, response_prompt_logprobs, server_options,
                                    verify_fla_overlay, verify_parity_preparation)
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.cuda_toolchain import verify_cuda_toolchain
from src.cmirank.reserved_handoff import gpu_snapshot, handoff


def save(path, value):
    path.write_text(artifact_json_dumps(value))


def append(run, name, row):
    with (run / f"{name}.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stop_owned(process, marker):
    if process is None or process.poll() is not None:
        return
    command = subprocess.check_output(["ps", "-p", str(process.pid), "-o", "args="], text=True)
    if marker not in command or os.getpgid(process.pid) != process.pid:
        raise RuntimeError("Owned process identity changed; refusing to signal unknown target")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner-confirmed-generator-step", required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config, path = load_parity_config(ROOT)
    gpu_index = config["gpu_index"]
    if gpu_index == 1 and not os.environ.get("MEMREC_KEEPER_AUTHORITY_PATH"):
        raise ValueError("Current GPU1 resource policy requires the recorded researcher standing authority")
    compat, _ = load_compat_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / config["run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]):
        raise ValueError("Fresh exact-source CPU coordinator required")
    for name, version in runtime["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned parity runtime changed")
    rows, prepared = verify_parity_preparation(private / "runs" / config["prepare_run_id"], config,
        commit=commit, config_sha=file_sha256(path), code_sha=parity_code_hashes(ROOT))
    if config.get("hf_kernel_overlay"):
        verify_fla_overlay(private, config)  # File/hash preparation BEFORE handoff.
    toolchain = verify_cuda_toolchain(private, ROOT, config)[0] if config.get("hf_cuda_toolchain") else None
    checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
    for name, expected in prepared["checkpoint_stat"].items():
        stat = (checkpoint / name).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino} != expected:
            raise ValueError("Checkpoint changed since CPU hash verification")
    for name, sha in prepared["overlay_sha256"].items():
        if file_sha256(private / "overlays" / compat["overlay_name"] / name) != sha:
            raise ValueError("Native kernel overlay changed since CPU preparation")
    for name, sha in prepared["native_source_sha256"].items():
        if file_sha256(Path(sys.prefix) / name) != sha:
            raise ValueError("Native runtime source changed since CPU inspection")
    parent = None
    if config.get("hf_precision_diagnostic"):
        parent = diagnostic_parent(private / "runs" / config["diagnostic_parent_run_id"], config, rows)
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", config["port"]))  # Do not call another project's service.
    run.mkdir()
    worker = server = card = hf_log = server_log = None
    exit_code, physical_requests = 1, 0
    began = time.monotonic()
    worker_marker = "scripts/cmirank/24_hf_parity_worker_gpu.py"
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned parity task interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    manifest = {"source_commit": commit, "config": config, "config_sha256": file_sha256(path),
                "preparation_report_sha256": file_sha256(private / "runs" / config["prepare_run_id"] / "report.json"),
                "slurm_job_id": job, "optimizer_updates": 0, "ranking_metrics_computed": False,
                "compiler_cache_paths": {k: os.environ.get(k) for k in
                    ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH",
                     "FLASHINFER_WORKSPACE_BASE", "DG_JIT_CACHE_DIR")}}
    save(run / "manifest.json", manifest)
    try:
        manifest["gpu_handoff"] = handoff(run, args.owner_confirmed_generator_step,
                                            gpu_index=gpu_index, protect_other_workloads=True,
                                            task_timeout_seconds=config["timeout_minutes"] * 60)
        cards, apps, raw, raw_apps = gpu_snapshot()
        card = select_reserved_gpu(cards, set(apps), gpu_index=gpu_index)
        if card.uuid != manifest["gpu_handoff"][f"gpu{gpu_index}_uuid"]:
            raise ValueError("Physical GPU changed after authorized handoff")
        cards, apps, raw, raw_apps = gpu_snapshot()
        if select_reserved_gpu(cards, set(apps), gpu_index=gpu_index).uuid != card.uuid:
            raise ValueError("Authorized GPU no longer idle immediately before worker initialization")
        pci = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id",
                                       "--format=csv,noheader,nounits"], text=True)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = verified_numeric_cuda_binding(cards, card, pci)
        os.environ["MEMREC_SELECTED_GPU_UUID"] = card.uuid
        manifest.update(gpu_uuid=card.uuid, physical_gpu_index=gpu_index, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
        save(run / "manifest.json", manifest)
        hf_log = (run / "hf-worker.log").open("x")
        hf_environment = dict(os.environ)
        if config.get("hf_kernel_overlay"):
            hf_environment["PYTHONPATH"] = str(private / "overlays" / config["hf_kernel_overlay"]) + os.pathsep + os.environ["PYTHONPATH"]
        if toolchain:
            hf_environment.update(config["hf_worker_environment"])
            hf_environment.update(CUDA_HOME=str(toolchain), CUDA_PATH=str(toolchain),
                PATH=str(toolchain / "bin") + os.pathsep + os.environ["PATH"],
                TILELANG_CACHE_DIR=str(private / "cache/runtime-hnv" / config["run_id"] / "tilelang"),
                TILELANG_TMP_DIR=str(private / "cache/runtime-hnv" / config["run_id"] / "tilelang-tmp"))
            manifest["hf_private_compiler"] = {"cuda_home": str(toolchain),
                "preparation_report_sha256": file_sha256(private / "runs" / config["toolchain_prepare_run_id"] / "report.json"),
                "environment": {k: hf_environment[k] for k in ("CUDA_HOME", "CUDA_PATH", "FLA_TILELANG", "TILELANG_CACHE_DIR", "TILELANG_TMP_DIR")}}
            save(run / "manifest.json", manifest)
        worker = subprocess.Popen([sys.executable, "-u", worker_marker, "--run-dir", str(run)], env=hf_environment,
            stdout=hf_log, stderr=subprocess.STDOUT, start_new_session=True)
        if worker.wait(timeout=600) != 0:
            raise RuntimeError("Owned HF forward-only worker failed; no vLLM load")
        cards, apps, raw, raw_apps = gpu_snapshot()
        if select_reserved_gpu(cards, set(apps), gpu_index=gpu_index).uuid != card.uuid:
            raise ValueError("HF child did not completely release the authorized GPU before vLLM load")
        save(run / "hf-release.json", {"child_exited": True, "gpu_released": True, "gpu_uuid": card.uuid})
        if config.get("hf_precision_diagnostic"):
            # The only model is ALREADY unloaded. All review below is CPU-only.
            controls = [json.loads(s) for s in (run / "precision-logprobs.jsonl").read_text().splitlines()]
            keys = [(r["variant"], r["user_id"]) for r in controls]
            expected = {(v, r["user_id"]) for v in config["diagnostic_variants"] for r in rows}
            if len(controls) != 100 or len(set(keys)) != 100 or set(keys) != expected:
                raise ValueError("Every fixed precision control must include all twenty unchanged inputs")
            original = {r["user_id"]: r for r in parent}
            summary = {}
            for variant in config["diagnostic_variants"]:
                probes = []
                for control in (r for r in controls if r["variant"] == variant):
                    prior = original[control["user_id"]]
                    comparison = compare_logprobs(control["logprobs"], prior["vllm"], config)
                    baseline_delta = max(abs(a-b) for a,b in zip(control["logprobs"], prior["hf"]))
                    append(run, "precision-comparison", {"variant": variant, "user_id": control["user_id"],
                        "max_absolute_delta_to_original_hf": baseline_delta, **comparison})
                    if variant == "baseline_bf16_fa2" and baseline_delta != 0:
                        raise ValueError("Diagnostic BF16/FA2 repeat did not reproduce the original baseline exactly")
                    probes.append(comparison)
                summary[variant] = {"samples": len(probes), "within_original_bounds": sum(r["passed"] for r in probes),
                    "max_token_absolute_logprob_delta": max(r["max_token_absolute_logprob_delta"] for r in probes),
                    "max_mean_absolute_logprob_delta": max(r["mean_absolute_logprob_delta"] for r in probes),
                    "max_action_absolute_logratio": max(abs(r["action_log_likelihood_ratio"]) for r in probes)}
            save(run / "report.json", {"status": "PRECISION_DIAGNOSIS_COMPLETE_ORIGINAL_PARITY_GATE_REMAINS_FAILED",
                "source_commit": commit, "config_sha256": file_sha256(path), "variants": summary,
                "samples_per_variant": 20, "total_forward_vectors": 100, "physical_requests": 0,
                "optimizer_updates": 0, "generation_calls": 0, "ranking_metrics_computed": False,
                "books_outcomes_accessed": False, "primary_profile_selected": False, "training_ready": False,
                "elapsed_seconds": time.monotonic() - began})
            exit_code = 0
            return  # finally still audits child cleanup; outer restores keeper1.
        hf_rows = [json.loads(s) for s in (run / "hf-logprobs.jsonl").read_text().splitlines()]
        if len(hf_rows) != 20 or [r["user_id"] for r in hf_rows] != [r["user_id"] for r in rows]:
            raise ValueError("HF worker did not record the exact twenty probes in order")
        server_log = (run / "vllm-server.log").open("x")
        server_args = [str(private / "envs" / runtime["environment_name"] / "bin/vllm"), "serve", str(checkpoint),
                       *server_options(config)]
        server_environment = {**os.environ, **config.get("server_environment", {})}
        manifest["server_environment_overrides"] = config.get("server_environment", {})
        save(run / "manifest.json", manifest)
        server = subprocess.Popen(server_args, env=server_environment,
            stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        (run / "server.pid").write_text(str(server.pid) + "\n")
        deadline = time.monotonic() + config["server_startup_timeout_seconds"]
        base = f"http://127.0.0.1:{config['port']}"
        while True:
            if server.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Owned vLLM server failed readiness/timed out")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(5)  # Bounded service readiness, not external result polling.
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
            models = json.load(response)
        if [m["id"] for m in models["data"]] != [config["served_model_name"]]:
            raise ValueError("Local service is not the task-owned initializer")
        comparisons = []
        for index, (row, hf) in enumerate(zip(rows, hf_rows)):
            if physical_requests >= config["physical_request_cap"]:
                raise ValueError("Fixed physical request cap exhausted; no retry/repair")
            payload = completion_request(row, config)
            physical_requests += 1
            append(run, "physical-requests", {"event": "request", "attempt": physical_requests,
                                               "user_id": row["user_id"], "payload": payload})
            request = urllib.request.Request(base + "/v1/completions", data=json.dumps(payload).encode(),
                                              headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=config["request_timeout_seconds"]) as response:
                result = json.load(response)
            append(run, "physical-requests", {"event": "response", "attempt": physical_requests,
                                               "user_id": row["user_id"], "response": result})
            observed = response_prompt_logprobs(result, row)
            comparison = {"user_id": row["user_id"], "sample": index,
                          **compare_logprobs(hf["logprobs"], observed, config)}
            append(run, "comparison", comparison)
            comparisons.append(comparison)
            if (index + 1) % 5 == 0:
                print(json.dumps({"completed_samples": index + 1, "elapsed_seconds": time.monotonic() - began}), flush=True)
        # Release service weights BEFORE report/client/artifact housekeeping.
        stop_owned(server, str(checkpoint))
        passed = all(r["passed"] for r in comparisons)
        report = {"status": "VLLM_HF_RESPONSE_PARITY_PASS_CLEANUP_REVIEW_AND_PPO_WORKER_GATES_REMAIN" if passed
                            else "VLLM_HF_RESPONSE_PARITY_FAIL_NUMERICAL_DIAGNOSIS_REQUIRED",
            "source_commit": commit, "config_sha256": file_sha256(path), "samples": len(comparisons),
            "physical_requests": physical_requests, "generation_calls": physical_requests,
            "generated_tokens_scope": "one_unscored_extension_per_teacher_forced_probe_not_ranking_rollout",
            "optimizer_updates": 0, "ranking_metrics_computed": False, "books_outcomes_accessed": False,
            "ppo_worker_proven": False, "training_ready": False,
            "initial_shared_weight_parity_only": True, "production_optimized_rollout_proven": False,
            "server_mode": "native_vllm_eager_no_prefix_cache_bf16_tp1",
            "language_model_only": config.get("language_model_only", False),
            "gdn_prefill_backend": config.get("gdn_prefill_backend", "auto"),
            "server_environment_overrides": config.get("server_environment", {}),
            "max_token_absolute_logprob_delta": max(r["max_token_absolute_logprob_delta"] for r in comparisons),
            "max_mean_absolute_logprob_delta": max(r["mean_absolute_logprob_delta"] for r in comparisons),
            "max_action_absolute_logratio": max(abs(r["action_log_likelihood_ratio"]) for r in comparisons),
            "elapsed_seconds": time.monotonic() - began}
        save(run / "report.json", report)
        exit_code = 0 if passed else 1
    except BaseException as error:
        save(run / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "physical_requests": physical_requests, "elapsed_seconds": time.monotonic() - began})
        raise
    finally:
        stop_owned(server, str(checkpoint))
        stop_owned(worker, worker_marker)
        if hf_log:
            hf_log.close()
        if server_log:
            server_log.close()
        save(run / "child-cleanup.json", {"process_exit_code": exit_code, "gpu_uuid": card.uuid if card else None,
            "owned_server_exited": server is None or server.poll() is not None,
            "owned_hf_worker_exited": worker is None or worker.poll() is not None,
            "optimizer_updates": 0, "training_ready": False})
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
