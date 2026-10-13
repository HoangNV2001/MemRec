"""Receipt guards for the official causal-conv1d private ABI overlay."""

import json
import os
from pathlib import Path

from .provenance import file_sha256


def verify_causal_overlay(private, root, config):
    contract_path = root / "configs/cmirank/causal_conv_overlay_v1.json"
    contract = json.loads(contract_path.read_text())
    run = private / "runs" / contract["run_id"]
    report = json.loads((run / "report.json").read_text())
    if (config["hf_causal_overlay"] != contract["overlay_name"]
            or report["status"] != "CAUSALCONV_CPU_BUILD_IMPORT_PASS_GPU_FULL_FASTPATH_GATES_REMAIN"
            or report["source_commit"] != os.environ["MEMREC_EXPECTED_COMMIT"]
            or report["config_sha256"] != file_sha256(contract_path)
            or report["source_sha256"] != contract["source_sha256"]
            or report["version"] != contract["version"] or report["gpu_requested"]
            or report["cuda_initialized"] or report["samples"] != 20
            or not report["base_environment_unchanged"] or report["training_ready"]):
        raise ValueError("Native causal convolution requires completed source-bound CPU build/import")
    for name in ("src/cmirank/causal_conv.py", "scripts/cmirank/27_build_causal_conv_cpu.py"):
        if report["code_sha256"].get(name) != file_sha256(root / name):
            raise ValueError("Causal convolution preparation code changed")
    overlay = private / "overlays" / contract["overlay_name"]
    for name, digest in report["overlay_sha256"].items():
        path = overlay / name
        if not path.resolve().is_relative_to(overlay) or file_sha256(path) != digest:
            raise ValueError("Native convolution overlay bytes changed")
    for name, digest in report["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(run / name) != digest:
            raise ValueError("Native convolution CPU receipt changed")
    return overlay, report
