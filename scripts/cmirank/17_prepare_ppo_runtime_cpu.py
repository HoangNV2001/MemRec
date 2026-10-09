#!/usr/bin/env python3
"""Create one isolated pinned PPO candidate env; CPU-only, keep GPUs untouched.

Resolver verification precedes installation. Never reuse/overwrite a partial
environment, call a scheduler cancellation, initialize CUDA or load 4B weights.
Tiny class probes are expressly not production PPO or full one-H100 proof.
"""

from __future__ import annotations

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
from src.cmirank.ppo_runtime import load_runtime_config, locked_requirements, require_new_environment, verify_resolution
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile-version", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config, config_path = load_runtime_config(ROOT, args.profile_version)
    run = args.run_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / config["run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != config["cpus"]
            or config["gpu_requested"] or config["training_ready"]):
        raise ValueError("Fresh source-matched scoped CPU step required")
    if sys.version_info[:2] != (3, 10):
        raise ValueError("Pinned official wheels require Python3.10")
    env = require_new_environment(private, config["environment_name"])
    run.mkdir()
    started = time.monotonic()
    subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    python = str(env / "bin/python")
    child_env = {**os.environ, "PIP_CACHE_DIR": str(private / "cache/ppo-runtime-v1/pip"),
                 "TMPDIR": str(run / "tmp"), "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled",
                 "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                 "RAYON_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    (run / "tmp").mkdir()
    def command(argv, label, timeout=1800):
        print(f"CPU runtime stage: {label}", flush=True)
        with (run / f"{label}.log").open("w") as log:
            result = subprocess.run(argv, env=child_env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see scoped run log (exit {result.returncode})")
    manifest = {"source_commit": commit, "config_sha256": file_sha256(config_path), "profile_version": args.profile_version,
                "base_config_sha256": file_sha256(ROOT / "configs/cmirank/ppo_runtime_v1.json"),
                "slurm_job_id": job, "python_version": sys.version, "environment": str(env),
                "gpu_requested": False, "gpu_step_cancelled": False, "model_weights_loaded": False,
                "existing_envs_mutated": False, "training_ready": False}
    (run / "manifest.json").write_text(artifact_json_dumps(manifest))
    try:
        command([python, "-m", "pip", "install", "pip==25.3", "setuptools==80.9.0", "wheel==0.45.1"], "bootstrap")
        requirements = [f"{n} @ {v['url']}#sha256={v['sha256']}" for n, v in config["wheels"].items()]
        requirements += [f"{n}=={v}" for n, v in config["versions"].items() if n not in config["wheels"]]
        requirements += [f"verl @ git+https://github.com/verl-project/verl.git@{config['verl_commit']}"]
        (run / "requested-requirements.txt").write_text("\n".join(requirements) + "\n")
        command([python, "-m", "pip", "install", "--dry-run", "--report", str(run / "resolution.json"),
                 "-r", str(run / "requested-requirements.txt")], "resolve")
        resolution = json.loads((run / "resolution.json").read_text())
        versions = verify_resolution(resolution, config)
        (run / "requirements.lock").write_text("\n".join(locked_requirements(resolution)) + "\n")
        command([python, "-m", "pip", "install", "--no-deps", "--report", str(run / "installation.json"),
                 "-r", str(run / "requirements.lock")], "install")
        actual = verify_resolution(json.loads((run / "installation.json").read_text()), config)
        if actual != versions:
            raise ValueError("Installed dependency graph differs from inspected resolution")
        command([python, "-m", "pip", "check"], "pip-check", timeout=60)
        with (run / "pip-freeze.txt").open("w") as handle:
            subprocess.run([python, "-m", "pip", "freeze", "--all"], env=child_env, stdout=handle, check=True)
        command([python, str(ROOT / "scripts/cmirank/18_smoke_ppo_classes_cpu.py"),
                 "--run-dir", str(run), "--profile-version", str(args.profile_version)], "miniature-cpu-smoke", timeout=600)
        report = {**manifest, "status": "PPO_DEPENDENCY_AND_MINIATURE_CPU_GATE_PASS_GPU_GATES_REMAIN",
                  "versions": versions, "elapsed_seconds": time.monotonic() - started,
                  "artifact_sha256": {p.name: file_sha256(p) for p in run.iterdir() if p.is_file()}}
        (run / "report.json").write_text(artifact_json_dumps(report))
        print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "training_ready")}), flush=True)
    except BaseException as error:
        (run / "failure.json").write_text(artifact_json_dumps({**manifest, "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.monotonic() - started,
            "partial_environment_preserved_do_not_reuse": True}))
        raise


if __name__ == "__main__":
    main()
