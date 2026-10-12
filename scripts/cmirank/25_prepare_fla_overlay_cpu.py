#!/usr/bin/env python3
"""Install two pinned pure native-kernel wheels into a NEW private overlay.

No GPU/model weights/framework patch/base-env mutation or CUDA compilation.
CPU import/signature probes cannot establish numerical/backward GPU agreement.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract-version", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config = json.loads((ROOT / "configs/cmirank/ppo_fla_overlay_v1.json").read_text())
    path = ROOT / f"configs/cmirank/ppo_fla_overlay_v{args.contract_version}.json"
    if args.contract_version == 2:
        retry = json.loads(path.read_text())
        if (set(retry) != {"schema_version", "inherits", "run_id", "reuse_installation_run_id", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_fla_overlay_v1.json"
                or retry["reuse_installation_run_id"] != config["run_id"]):
            raise ValueError("CPU guard retry must preserve the pinned wheel/profile contract")
        config.update(retry)
    runtime, _ = load_runtime_config(ROOT)
    run, overlay = args.run_dir.resolve(), private / "overlays" / config["overlay_name"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    reuse = config.get("reuse_installation_run_id")
    if (run != private / "runs" / config["run_id"] or run.exists()
            or (not reuse and overlay.exists()) or (reuse and not overlay.is_dir())
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 4
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]
            or config["gpu_requested"] or config["full_weights_loaded"] or config["base_environment_mutation"]
            or config["training_ready"] or config["samples"] != 20):
        raise ValueError("Only a fresh exact-source private CPU overlay preparation is allowed")
    for name, version in runtime["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Base candidate runtime changed")
    base_before = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    started = time.monotonic()
    run.mkdir()
    try:
        from packaging.requirements import Requirement
        expected_versions = {w["name"]: w["version"] for w in config["wheels"]}
        wheel_paths = []
        installed_candidates = {}
        for wheel in config["wheels"]:
            if (not wheel["url"].startswith("https://files.pythonhosted.org/packages/")
                    or Path(wheel["filename"]).name != wheel["filename"]
                    or not wheel["filename"].endswith("py3-none-any.whl")):
                raise ValueError("Only official pinned pure Python wheels may enter the overlay")
            destination = (private / "runs" / reuse if reuse else run) / wheel["filename"]
            if not reuse:
                with urllib.request.urlopen(wheel["url"], timeout=60) as response, destination.open("xb") as out:
                    while block := response.read(1024*1024): out.write(block)
            if destination.stat().st_size != wheel["size"] or file_sha256(destination) != wheel["sha256"]:
                raise ValueError("Official kernel wheel differs from its pre-measurement pin")
            with zipfile.ZipFile(destination) as archive:
                metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
                if len(metadata) != 1 or any(n.endswith((".so", ".pyd")) for n in archive.namelist()):
                    raise ValueError("Overlay must not introduce compiled/native ABI wheels")
                from email.parser import Parser
                headers = Parser().parsestr(archive.read(metadata[0]).decode())
                if headers["Name"] != wheel["name"] or headers["Version"] != wheel["version"]:
                    raise ValueError("Wheel name/version metadata differs")
                for dependency in headers.get_all("Requires-Dist", []):
                    requirement = Requirement(dependency)
                    if requirement.marker and not requirement.marker.evaluate({"extra": ""}): continue
                    installed = expected_versions.get(requirement.name) or importlib.metadata.version(requirement.name)
                    if installed not in requirement.specifier:
                        raise ValueError("Kernel wheel requires a base dependency change; do not auto-upgrade")
                if reuse:
                    import hashlib
                    for name in archive.namelist():
                        if name.endswith("/") or name.endswith(".dist-info/RECORD"): continue
                        if Path(name).is_absolute() or ".." in Path(name).parts:
                            raise ValueError("Wheel contains an unsafe installed path")
                        installed_candidates.setdefault(name, set()).add(hashlib.sha256(archive.read(name)).hexdigest())
            wheel_paths.append(str(destination))
        if reuse:
            previous = private / "runs" / reuse
            if not (previous / "failure.json").is_file() or not (previous / "pip-overlay.log").is_file():
                raise ValueError("Reuse requires the known failed CPU probe's installation evidence")
            for name, digests in installed_candidates.items():
                if not (overlay / name).is_file() or file_sha256(overlay / name) not in digests:
                    raise ValueError("Installed overlay differs from both official pinned wheel payloads")
        else:
            with (run / "pip-overlay.log").open("x") as log:
                subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-index",
                                "--no-deps", "--target", str(overlay), *wheel_paths], stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
        code = """import json,inspect,torch,importlib.metadata
from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
from fla.modules import FusedRMSNormGated
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.utils.import_utils import is_flash_linear_attention_available
# HF explicitly requires CUDA availability: respect it, never fake dispatch on CPU.
assert not is_flash_linear_attention_available()
assert hf.chunk_gated_delta_rule is None and hf.fused_recurrent_gated_delta_rule is None
assert hf.FusedRMSNormGated is None
assert not torch.cuda.is_initialized()
for i in range(20):
 inspect.signature(chunk_gated_delta_rule).bind(object(),object(),object(),g=object(),beta=object(),
     initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True,cu_seqlens=None)
assert not torch.cuda.is_initialized()
print(json.dumps({'samples':20,'imported_fla_chunk_module':chunk_gated_delta_rule.__module__,
 'imported_fla_norm_module':FusedRMSNormGated.__module__,'hf_cuda_only_availability_guard_respected':True,
 'hf_cuda_dispatch_proven':False,'causal_conv1d_available':hf.causal_conv1d_fn is not None,
 'cuda_initialized':False,'full_weights_loaded':False,'versions':{n:importlib.metadata.version(n) for n in
 ('flash-linear-attention','fla-core','torch','transformers','triton')}}))
"""
        environment = {**os.environ, "PYTHONPATH": str(overlay) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        output = subprocess.check_output([sys.executable, "-c", code], env=environment, text=True, timeout=180)
        (run / "import-probe.stdout").write_text(output)
        probe = json.loads(output.splitlines()[-1])
        if base_before != {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}:
            raise ValueError("Base environment was mutated by overlay installation")
        files = {str(p.relative_to(overlay)): file_sha256(p) for p in sorted(overlay.rglob("*"))
                 if p.is_file() and (p.suffix == ".py" or p.name in ("METADATA", "WHEEL", "RECORD"))}
        report = {"status": "FRESH_NATIVE_FLA_CPU_IMPORT_PASS_GPU_KERNEL_AND_PARITY_GATES_REMAIN",
            "source_commit": commit, "config_sha256": file_sha256(path), "slurm_job_id": job,
            "overlay_name": config["overlay_name"], "overlay_sha256": files, "probe": probe,
            "reuse_installation_from_run_id": reuse,
            "base_environment_unchanged": True, "gpu_requested": False, "full_weights_loaded": False,
            "optimizer_updates": 0, "training_ready": False, "elapsed_seconds": time.monotonic()-started}
        (run / "report.json").write_text(artifact_json_dumps(report))
        print(json.dumps({k:report[k] for k in ("status","overlay_name","elapsed_seconds")}), flush=True)
    except BaseException as error:
        (run / "failure.json").write_text(artifact_json_dumps({"error_type":type(error).__name__,
            "error":str(error),"elapsed_seconds":time.monotonic()-started,"gpu_requested":False}))
        raise


if __name__ == "__main__":
    main()
