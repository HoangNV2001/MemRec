#!/usr/bin/env python3
"""Build pinned official causal-conv1d against private Torch2.11/CUDA12.9 on CPU.

The existing environment and both GPU keepers stay untouched. Import is not a
GPU execution or gradient correctness proof. Native source is never patched.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.cuda_toolchain import validate_archive_members, verify_cuda_toolchain
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    path = ROOT / "configs/cmirank/causal_conv_overlay_v1.json"
    config = json.loads(path.read_text())
    runtime, _ = load_runtime_config(ROOT)
    run, overlay = args.run_dir.resolve(), private / "overlays" / config["overlay_name"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (run != private / "runs" / config["run_id"] or run.exists() or overlay.exists() or overlay.is_symlink()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("PYTHONPATH")
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != config["cpus"]
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or config["gpu_requested"] or config["base_environment_mutation"] or config["training_ready"]):
        raise ValueError("Fresh exact-source isolated CPU convolution build required")
    for name, version in runtime["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned base runtime changed")
    toolchain, _ = verify_cuda_toolchain(private, ROOT, config)
    before = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if "causal-conv1d" in before:
        raise ValueError("Do not overwrite an existing base convolution installation")
    run.mkdir()
    (run / "wheels").mkdir()
    (run / "tmp").mkdir()
    began = time.monotonic()
    environment = {**os.environ, "CUDA_HOME": str(toolchain), "CUDA_PATH": str(toolchain),
        "PATH": str(toolchain / "bin") + os.pathsep + os.environ["PATH"],
        "MAX_JOBS": str(config["max_jobs"]), "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
        "PIP_CACHE_DIR": str(private / "cache/pip"), "TMPDIR": str(run / "tmp"),
        "TORCH_EXTENSIONS_DIR": str(run / "torch-extensions"),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned CPU build interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def command(argv, log_name, timeout):
        with (run / log_name).open("x") as log:
            child = subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
                raise
        if code:
            raise RuntimeError(f"Owned native build stage failed: {log_name} exit{code}")
    try:
        archive_path = run / "causal_conv1d-1.7.0.tar.gz"
        with urllib.request.urlopen(config["source_url"], timeout=60) as response, archive_path.open("xb") as out:
            out.write(response.read(config["source_size"] + 1))
        if archive_path.stat().st_size != config["source_size"] or file_sha256(archive_path) != config["source_sha256"]:
            raise ValueError("Official native convolution source changed")
        source_directory = run / "source"
        source_directory.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            prefix = validate_archive_members(members)
            archive.extractall(source_directory)
        source = source_directory / prefix
        source_hashes = {str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*")
                         if p.is_file() and (p.suffix in (".py", ".cu", ".cpp", ".h", ".cuh") or p.name == "pyproject.toml")}
        print("CPU_CAUSALCONV_OFFICIAL_BUILD_START max_jobs=1 native_nvcc_threads=4", flush=True)
        command([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-build-isolation", "--no-deps",
                 "--verbose", "--wheel-dir", str(run / "wheels"), str(source)], "compile.log", config["compile_timeout_seconds"])
        for name, digest in source_hashes.items():
            if file_sha256(source / name) != digest:
                raise ValueError("Build patched immutable official source")
        wheels = list((run / "wheels").glob("causal_conv1d-1.7.0-*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one native Torch ABI wheel")
        command([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                 "--target", str(overlay), str(wheels[0])], "overlay-install.log", 120)
        environment["PYTHONPATH"] = str(overlay)
        probe = """import json,torch,importlib.metadata,causal_conv1d,causal_conv1d_cuda
assert importlib.metadata.version('causal-conv1d')=='1.7.0'
assert torch.version.cuda=='12.9' and torch._C._GLIBCXX_USE_CXX11_ABI
assert not torch.cuda.is_initialized()
for _ in range(20):
 assert callable(causal_conv1d.causal_conv1d_fn) and callable(causal_conv1d.causal_conv1d_update)
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
assert hf.causal_conv1d_fn is None and hf.causal_conv1d_update is None
assert not torch.cuda.is_initialized()
print(json.dumps({'samples':20,'cuda_initialized':False,'hf_cpu_cuda_availability_guard_respected':True,
 'extension_path':causal_conv1d_cuda.__file__,'torch_version':torch.__version__}))
"""
        command([sys.executable, "-c", probe], "cpu-import.log", 60)
        unchanged = before == {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        if not unchanged:
            raise ValueError("CPU native build changed the base environment")
        # Build log/import probe include the real Torch2.11/cu129 ABI; no guessed
        # prebuilt wheel, fake CUDA availability, or GPU model construction.
        report = {"status": "CAUSALCONV_CPU_BUILD_IMPORT_PASS_GPU_FULL_FASTPATH_GATES_REMAIN",
            "source_commit": commit, "config_sha256": file_sha256(path), "slurm_job_id": job,
            "source_sha256": config["source_sha256"], "source_files_sha256": source_hashes,
            "code_sha256": {n: file_sha256(ROOT / n) for n in (
                "src/cmirank/causal_conv.py", "scripts/cmirank/27_build_causal_conv_cpu.py")},
            "version": config["version"], "wheel_sha256": file_sha256(wheels[0]),
            "elapsed_seconds": time.monotonic() - began, "samples":20,
            "gpu_requested":False, "cuda_initialized":False, "training_ready":False,
            "base_environment_unchanged":unchanged, "overlay_name":config["overlay_name"],
            "toolchain_report_sha256": config["toolchain_prepare_report_sha256"],
            "overlay_sha256": {str(p.relative_to(overlay)): file_sha256(p) for p in overlay.rglob("*")
                               if p.is_file() and (p.suffix in (".py", ".so") or p.name in ("METADATA", "WHEEL", "RECORD"))},
            "artifact_sha256": {n:file_sha256(run/n) for n in ("compile.log","overlay-install.log","cpu-import.log")}}
        (run / "report.json").write_text(artifact_json_dumps(report))
        print(json.dumps({k:report[k] for k in ("status","elapsed_seconds","samples")}), flush=True)
    except BaseException as error:
        (run / "failure.json").write_text(artifact_json_dumps({"error_type":type(error).__name__,
            "error":str(error),"elapsed_seconds":time.monotonic()-began,"gpu_requested":False}))
        raise


if __name__ == "__main__":
    main()
