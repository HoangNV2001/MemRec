#!/usr/bin/env python3
"""Build the missing native critic kernel in a new overlay, keep GPUs untouched.

Compile official FA2 against the verified private Torch ABI. Existing env is
read-only; no unverified wheel for a different Torch version or global install.
CPU import/symbol checks do not establish CUDA kernel or PPO-worker correctness.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config_path = ROOT / "configs/cmirank/ppo_kernel_build_v1.json"
    config = json.loads(config_path.read_text())
    runtime, runtime_path = load_runtime_config(ROOT)
    run = args.run_dir.resolve()
    overlay = private / "overlays" / config["overlay_name"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / config["run_id"] or run.exists() or overlay.exists() or overlay.is_symlink()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("PYTHONPATH")
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != config["cpus"]
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]):
        raise ValueError("Fresh isolated CPU build on exact tested source required")
    parent = private / "runs" / config["parent_runtime_run_id"]
    if file_sha256(parent / "report.json") != config["parent_report_sha256"]:
        raise ValueError("Wrong completed miniature CPU runtime receipt")
    report = json.loads((parent / "report.json").read_text())
    if (report["status"] != "PPO_DEPENDENCY_AND_MINIATURE_CPU_GATE_PASS_GPU_GATES_REMAIN"
            or report["config_sha256"] != file_sha256(runtime_path)
            or any(file_sha256(parent / n) != sha for n, sha in report["artifact_sha256"].items())):
        raise ValueError("CPU gate artifacts changed")
    for name, version in report["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Verified CPU environment changed")
    try:
        importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("Do not overwrite an existing FlashAttention installation")
    import torch
    if torch.cuda.is_initialized() or torch.version.cuda != "12.9" or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise ValueError("Pinned CPU-only Torch CUDA family/ABI changed")
    compiler = Path(config["cuda_home"]) / "bin/nvcc"
    nvcc = subprocess.check_output([str(compiler), "--version"], text=True)
    if f"release {config['nvcc_release']}," not in nvcc:
        raise ValueError("Available compiler differs; no system/compiler mutation permitted")
    run.mkdir()
    (run / "tmp").mkdir()
    (run / "wheels").mkdir()
    started = time.monotonic()
    child_env = {**os.environ, "CUDA_HOME": config["cuda_home"],
        "MAX_JOBS": str(config["max_jobs"]), "NVCC_THREADS": str(config["nvcc_threads"]),
        "FLASH_ATTN_CUDA_ARCHS": config["cuda_arch"], "TORCH_CUDA_ARCH_LIST": "9.0",
        "FLASH_ATTENTION_FORCE_BUILD": "TRUE", "FLASH_ATTENTION_FORCE_CXX11_ABI": "TRUE",
        "PIP_CACHE_DIR": str(private / "cache/ppo-runtime-v1/pip"), "TMPDIR": str(run / "tmp"),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "TORCH_EXTENSIONS_DIR": str(run / "torch-extensions")}
    manifest = {"source_commit": commit, "config_sha256": file_sha256(config_path), "config": config,
        "slurm_job_id": job, "parent_report_sha256": config["parent_report_sha256"],
        "torch_version": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "nvcc_version": nvcc, "cuda_compiler_runtime_minor_mismatch": True,
        "gpu_requested": False, "gpu_initialized": False, "existing_environment_mutated": False,
        "driver_or_system_mutated": False, "training_ready": False}
    (run / "manifest.json").write_text(artifact_json_dumps(manifest))
    def command(argv, name, timeout=2100):
        print(f"CPU kernel stage: {name}", flush=True)
        with (run / f"{name}.log").open("w") as log:
            result = subprocess.run(argv, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{name} failed, see run log (exit {result.returncode})")
    try:
        command([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
            "--wheel-dir", str(run / "wheels"),
            f"git+https://github.com/Dao-AILab/flash-attention.git@{config['flash_attention_commit']}"], "compile")
        wheels = list((run / "wheels").glob("flash_attn-2.8.3-*.whl"))
        if len(wheels) != 1 or f"commit {config['flash_attention_commit']}" not in (run / "compile.log").read_text():
            raise ValueError("Missing unique wheel/pinned official source provenance")
        command([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(overlay),
                 "--report", str(run / "overlay-installation.json"), str(wheels[0])], "overlay-install", timeout=180)
        child_env["PYTHONPATH"] = str(overlay)
        command([sys.executable, "-c", "import importlib.metadata,json,torch,flash_attn; "
            "assert importlib.metadata.version('flash-attn') == '2.8.3'; "
            "assert not torch.cuda.is_initialized(); "
            "print(json.dumps({'flash_attn_version':flash_attn.__version__,'module':flash_attn.__file__,"
            "'cuda_initialized':False}))"], "cpu-import", timeout=60)
        command([sys.executable, "-m", "pip", "check"], "pip-check", timeout=60)
        for name, version in report["versions"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError("Build mutated existing verified dependencies")
        record = {**manifest, "status": "FA2_CPU_BUILD_AND_IMPORT_PASS_CUDA_AND_WORKER_GATES_REMAIN",
            "overlay": str(overlay), "wheel_sha256": file_sha256(wheels[0]), "wheel_name": wheels[0].name,
            "elapsed_seconds": time.monotonic() - started,
            "artifact_sha256": {p.name: file_sha256(p) for p in run.iterdir() if p.is_file()}}
        (run / "report.json").write_text(artifact_json_dumps(record))
        print(json.dumps({k: record[k] for k in ("status", "elapsed_seconds", "training_ready")}), flush=True)
    except BaseException as error:
        (run / "failure.json").write_text(artifact_json_dumps({**manifest, "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.monotonic() - started}))
        raise


if __name__ == "__main__":
    main()
