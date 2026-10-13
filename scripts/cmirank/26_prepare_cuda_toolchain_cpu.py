#!/usr/bin/env python3
"""CPU-only private NVCC12.9 + headers; replay exact failed TileLang CUDA source.

No driver/runfile/pip install, no system CUDA changes, no CUDA context or model.
Official NVIDIA archives are pinned by URL/size/SHA before any GPU measurement.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.cuda_toolchain import failed_cuda_source, validate_archive_members
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    path = ROOT / "configs/cmirank/cuda_toolchain_v1.json"
    config = json.loads(path.read_text())
    runtime, _ = load_runtime_config(ROOT)
    run = args.run_dir.resolve()
    toolchain = private / "toolchains" / config["toolchain_name"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / config["run_id"] or run.exists() or toolchain.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 4
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or config["gpu_requested"] or config["base_environment_mutation"]
            or config["driver_or_system_cuda_mutation"]):
        raise ValueError("Fresh exact-source private CPU compiler preparation required")
    started = time.monotonic()
    before = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    for name, version in runtime["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned PPO runtime changed; compiler preparation cannot upgrade it")
    system_nvcc = Path("/usr/local/cuda/bin/nvcc").resolve()
    system_sha = file_sha256(system_nvcc)
    failed_log = private / "runs" / config["failed_run_id"] / "hf-worker.log"
    if file_sha256(failed_log) != config["failed_log_sha256"]:
        raise ValueError("Reviewed failed compiler source changed")
    run.mkdir()
    toolchain.mkdir(parents=True)
    try:
        with urllib.request.urlopen(config["manifest_url"], timeout=30) as response:
            manifest_bytes = response.read()
        (run / "nvidia-redistrib.json").write_bytes(manifest_bytes)
        manifest = json.loads(manifest_bytes)
        for component in config["components"]:
            remote = manifest[component["name"]]
            entry = remote["linux-x86_64"]
            if (remote["version"] != component["version"]
                    or any(str(entry[k]) != str(component[k]) for k in ("relative_path", "size", "sha256"))):
                raise ValueError("Official compiler manifest differs from frozen pins")
            archive_path = run / Path(component["relative_path"]).name
            url = "https://developer.download.nvidia.com/compute/cuda/redist/" + component["relative_path"]
            with urllib.request.urlopen(url, timeout=90) as response, archive_path.open("xb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if archive_path.stat().st_size != component["size"] or file_sha256(archive_path) != component["sha256"]:
                raise ValueError("Private compiler archive size/SHA differs")
            unpack = run / component["name"]
            unpack.mkdir()
            with tarfile.open(archive_path, "r:xz") as archive:
                members = archive.getmembers()
                prefix = validate_archive_members(members)
                archive.extractall(unpack)
            payload = unpack / prefix
            # Merge only this fresh task's verified payload; never overwrite
            # an existing toolkit or a different component's conflicting bytes.
            for source in sorted(payload.rglob("*")):
                relative = source.relative_to(payload)
                if relative.parts[0] not in ("bin", "include", "nvvm", "lib", "lib64"):
                    continue  # Per-component licenses remain in retained archives/unpack.
                destination = toolchain / relative
                if source.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif source.is_file():
                    if destination.exists():
                        if file_sha256(destination) != file_sha256(source):
                            raise ValueError("Conflicting compiler component file")
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
        compiler = toolchain / "bin/nvcc"
        version = subprocess.check_output([str(compiler), "--version"], text=True)
        if "V12.9.86" not in version:
            raise ValueError("Private NVCC version differs from pin")
        (run / "nvcc-version.txt").write_text(version)
        source_path = run / "failed-kernel-replay.cu"
        source_path.write_text(failed_cuda_source(failed_log.read_text()))
        tile_root = Path(sys.prefix) / "lib/python3.10/site-packages/tilelang"
        args = [str(compiler), "--cubin", "-O3", "-lineinfo", "-arch=sm_90a", "-std=c++17",
                "-I" + str(tile_root / "src"), "-I" + str(tile_root / "3rdparty/cutlass/include"),
                "--use_fast_math", "-o", str(run / "failed-kernel-replay.cubin"), str(source_path)]
        # Installed templates actually live at <tilelang>/3rdparty/../src.
        with (run / "compile-replay.log").open("x") as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        if not (run / "failed-kernel-replay.cubin").stat().st_size:
            raise ValueError("Empty compiler replay output")
        # Native toolkit lookup happens at import time in the owned CPU child.
        probe = "import json,torch;from tilelang.env import CUDA_HOME;assert CUDA_HOME==" + repr(str(toolchain)) + ";assert not torch.cuda.is_initialized();print(json.dumps({'cuda_initialized':False,'native_tilelang_cuda_home':CUDA_HOME}))"
        environment = {**os.environ, "CUDA_HOME": str(toolchain), "CUDA_PATH": str(toolchain),
                       "TILELANG_CACHE_DIR": str(run / "tilelang-cache"), "TILELANG_TMP_DIR": str(run / "tilelang-tmp")}
        observed = json.loads(subprocess.check_output([sys.executable, "-c", probe], env=environment, text=True, timeout=60))
        unchanged = before == {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        if not unchanged or file_sha256(system_nvcc) != system_sha:
            raise ValueError("System compiler/base environment changed")
        report = {"status": "PRIVATE_CUDA129_CPU_REPLAY_PASS_GPU_KERNEL_GATES_REMAIN",
                  "source_commit": commit, "config_sha256": file_sha256(path), "slurm_job_id": job,
                  "code_sha256": {n: file_sha256(ROOT / n) for n in (
                      "src/cmirank/cuda_toolchain.py", "scripts/cmirank/26_prepare_cuda_toolchain_cpu.py")},
                  "toolchain_name": config["toolchain_name"], "failed_log_sha256": config["failed_log_sha256"],
                  "exact_failed_kernel_cpu_compile_passed": True, "compile_command": args,
                  "base_environment_unchanged": unchanged, "system_nvcc_unchanged": True,
                  "system_nvcc_sha256": system_sha, "gpu_requested": False, "cuda_initialized": observed["cuda_initialized"],
                  "native_tilelang_cuda_home": observed["native_tilelang_cuda_home"], "training_ready": False,
                  "elapsed_seconds": time.monotonic() - started,
                  "toolchain_sha256": {str(p.relative_to(toolchain)): file_sha256(p)
                                       for p in toolchain.rglob("*") if p.is_file()},
                  "artifact_sha256": {n: file_sha256(run / n) for n in (
                      "nvidia-redistrib.json", "nvcc-version.txt", "failed-kernel-replay.cu",
                      "failed-kernel-replay.cubin", "compile-replay.log")}}
        (run / "report.json").write_text(artifact_json_dumps(report))
        print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "gpu_requested")}), flush=True)
    except BaseException as error:
        (run / "failure.json").write_text(artifact_json_dumps({"error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.monotonic() - started, "gpu_requested": False}))
        raise


if __name__ == "__main__":
    main()
