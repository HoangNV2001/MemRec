"""Private compiler receipts; no GPU selection or driver/environment installation."""

import json
import os
from pathlib import Path, PurePosixPath

from .provenance import file_sha256


def validate_archive_members(members):
    """Reject archive traversal, special files and links outside its single root."""
    roots = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe compiler archive path")
        roots.add(path.parts[0])
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError("Compiler archives cannot contain special devices")
        if member.issym() or member.islnk():
            link = PurePosixPath(member.linkname)
            if link.is_absolute():
                raise ValueError("Absolute compiler archive link")
            parts = list(path.parent.parts) if member.issym() else []
            for part in link.parts:
                if part == "..":
                    if len(parts) <= 1:
                        raise ValueError("Compiler archive link escapes component root")
                    parts.pop()
                elif part != ".":
                    parts.append(part)
            if not parts or parts[0] != path.parts[0]:
                raise ValueError("Compiler archive link changes component root")
    if len(roots) != 1:
        raise ValueError("Compiler component needs exactly one archive root")
    return next(iter(roots))


def failed_cuda_source(log):
    start = log.index("RuntimeError: #include ") + len("RuntimeError: ")
    end = log.index("\nCompilation error:", start)
    source = log[start:end].strip() + "\n"
    if "cp.async.bulk.tensor" not in log or "kernel_kernel" not in source:
        raise ValueError("Not the reviewed TileLang TMA compiler regression")
    return source


def verify_cuda_toolchain(private, root, config):
    contract_path = root / "configs/cmirank/cuda_toolchain_v1.json"
    contract = json.loads(contract_path.read_text())
    run = private / "runs" / contract["run_id"]
    report = json.loads((run / "report.json").read_text())
    if (config["hf_cuda_toolchain"] != contract["toolchain_name"]
            or config["toolchain_prepare_run_id"] != contract["run_id"]
            or report["status"] != "PRIVATE_CUDA129_CPU_REPLAY_PASS_GPU_KERNEL_GATES_REMAIN"
            or report["config_sha256"] != file_sha256(contract_path)
            or report["source_commit"] != os.environ["MEMREC_EXPECTED_COMMIT"]
            or report["toolchain_name"] != contract["toolchain_name"]
            or report["gpu_requested"] or report["cuda_initialized"] or report["training_ready"]
            or not report["base_environment_unchanged"] or not report["system_nvcc_unchanged"]
            or report["failed_log_sha256"] != contract["failed_log_sha256"]
            or not report["exact_failed_kernel_cpu_compile_passed"]):
        raise ValueError("Private compiler requires completed CPU-only regression replay")
    for name in ("src/cmirank/cuda_toolchain.py", "scripts/cmirank/26_prepare_cuda_toolchain_cpu.py"):
        if report["code_sha256"].get(name) != file_sha256(root / name):
            raise ValueError("Private compiler CPU harness changed")
    toolchain = private / "toolchains" / contract["toolchain_name"]
    for name, digest in report["toolchain_sha256"].items():
        path = toolchain / name
        if not path.resolve().is_relative_to(toolchain) or file_sha256(path) != digest:
            raise ValueError("Private compiler bytes changed after CPU qualification")
    for name, digest in report["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(run / name) != digest:
            raise ValueError("Private compiler replay artifact changed")
    return toolchain, report
