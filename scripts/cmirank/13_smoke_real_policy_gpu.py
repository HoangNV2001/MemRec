#!/usr/bin/env python3
"""One-card real N−1/direct functional diagnostic on reviewed secondary inputs.

Prepare/check tokenizer and all full prompts on CPU before generator handoff.
No target file, ranking metric, repair, optimizer or checkpoint is accessed.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
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
from src.cmirank.gpu_resources import (
    canonical_gpu_uuid, compute_gpu_processes, parse_gpu_snapshot, select_reserved_gpu1,
    verified_numeric_cuda_binding,
)
from src.cmirank.labels import make_labels
from src.cmirank.policy_smoke import load_secondary_inputs, run_functional_smoke, verify_cpu_token_audit
from src.cmirank.prompts import render_direct_prompt, render_step_prompt
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.memory_smoke import object_sha256


def save(path, value):
    path.write_text(artifact_json_dumps(value))


def capture(run, label):
    cards = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    (run / f"gpus-{label}.csv").write_text(cards)
    (run / f"apps-{label}.csv").write_text(apps)
    return parse_gpu_snapshot(cards), compute_gpu_processes(apps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner-confirmed-generator-step")
    parser.add_argument("--cpu-audit-only", action="store_true")
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    if not args.cpu_audit_only and not args.owner_confirmed_generator_step:
        raise ValueError("A fresh current generator ownership confirmation is required for GPU use")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (commit != os.environ["MEMREC_EXPECTED_COMMIT"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 4
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Exact clean source, CPU preparation and authorized allocation required")
    config_path = ROOT / "configs/cmirank/real_policy_smoke_v1.json"
    config = json.loads(config_path.read_text())
    if (config["users"] != 20 or config["generation_cap"] != 200 or config["nominal_generations"] != 200
            or config["training_ready"] or config["model_updates"] or config["ranking_metrics_computed"]
            or config["output_repair"] or config["do_sample"] or config["enable_thinking"]):
        raise ValueError("Not a locked secondary functional smoke")
    run = args.run_dir.resolve()
    expected_run_id = "cmirank-qwen35-real-policy-cpu-v1-20261008-hnv" if args.cpu_audit_only else config["run_id"]
    if run != private / "runs" / expected_run_id or run.exists():
        raise ValueError("Fresh scoped output required")
    contract = json.loads((ROOT / config["model_contract"]).read_text())
    if contract["model_revision"] != config["model_revision"]:
        raise ValueError("Primary policy revision changed")
    requests, provenance = load_secondary_inputs(private / "runs" / config["memory_run_id"],
        private / "runs" / config["memory_review_id"] / "review.json", expected_review_sha=config["memory_review_sha256"])
    model_path = private / f"models/Qwen3.5-4B-{contract['model_revision'][:7]}-hnv"
    marker = json.loads((model_path / "download-complete-hnv.json").read_text())
    if marker["model_revision"] != config["model_revision"] or not marker["sha256"]:
        raise ValueError("Pinned completed policy checkpoint required")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    if versions != {"torch": "2.8.0", "transformers": "5.13.0"}:
        # Distribution metadata uses 2.8.0 (runtime torch.__version__ has +cu128).
        if versions != {"torch": "2.8.0+cu128", "transformers": "5.13.0"}:
            raise ValueError("Existing Qwen3.5 smoke environment changed")
    run.mkdir()
    model, card, torch, exit_code = None, None, None, 1
    started = time.monotonic()

    def interrupted(signum, frame):
        raise InterruptedError(f"Owned smoke interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.cpu_audit_only:
            import torch
            from transformers import AutoTokenizer
            if torch.cuda.is_initialized():
                raise RuntimeError("CPU audit must not initialize CUDA")
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            audit = []
            for request in requests:
                for mode, prompt in (("iterative", render_step_prompt(request, make_labels(range(10)))),
                                     ("direct", render_direct_prompt(request))):
                    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    length = len(tokenizer.encode(rendered, add_special_tokens=False))
                    if length > config["max_input_tokens"]:
                        raise ValueError("Real full prompt exceeds locked context cap; no truncation")
                    audit.append({"user_id": request.user_id, "mode": mode, "tokens": length,
                                  "rendered_prompt_sha256": object_sha256(rendered)})
        else:
            # The CPU tokenizer lives in a separate, exited process. Do not
            # import torch/transformers in this process until CVD is bound.
            audit = verify_cpu_token_audit(private / "runs/cmirank-qwen35-real-policy-cpu-v1-20261008-hnv",
                commit=commit, config_sha=file_sha256(config_path),
                prompt_sha=file_sha256(ROOT / "src/cmirank/prompts.py"), parser_sha=file_sha256(ROOT / "src/cmirank/parser.py"),
                marker_sha=file_sha256(model_path / "download-complete-hnv.json"), provenance=provenance,
                max_input=config["max_input_tokens"])
        save(run / "cpu-token-audit.json", audit)
        manifest = {"config": config, "config_sha256": file_sha256(config_path), "source_commit": commit,
            "slurm_job_id": job, "versions": versions, "checkpoint_sha256": marker["sha256"],
            "checkpoint_marker_sha256": file_sha256(model_path / "download-complete-hnv.json"),
            "prompt_code_sha256": file_sha256(ROOT / "src/cmirank/prompts.py"),
            "parser_code_sha256": file_sha256(ROOT / "src/cmirank/parser.py"), **provenance}
        manifest["cpu_audit_only"] = args.cpu_audit_only
        save(run / "manifest.json", manifest)
        if args.cpu_audit_only:
            save(run / "report.json", {"status": "CPU_REAL_POLICY_INPUT_TOKEN_AUDIT_PASS_NO_GPU",
                "source_commit": commit, "config_sha256": file_sha256(config_path), "users": 20,
                "prompts_audited": len(audit), "input_tokens_min": min(row["tokens"] for row in audit),
                "input_tokens_max": max(row["tokens"] for row in audit), "max_input_tokens": config["max_input_tokens"],
                "gpu_requested": False, "model_weights_loaded": False, "model_updates": 0,
                "token_audit_sha256": file_sha256(run / "cpu-token-audit.json"),
                "ranking_metrics_computed": False, "training_ready": False,
                "elapsed_seconds": time.monotonic() - started, **provenance})
            exit_code = 0
            print((run / "report.json").read_text(), flush=True)
            return
        from src.cmirank.reserved_handoff import handoff
        manifest["gpu1_handoff"] = handoff(run, args.owner_confirmed_generator_step)
        cards, apps = capture(run, "before")
        card = select_reserved_gpu1(cards, set(apps))
        cards, apps = capture(run, "load-time")
        card = select_reserved_gpu1(cards, set(apps))
        pci = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"], text=True)
        (run / "gpu-pci-load-time.csv").write_text(pci)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = verified_numeric_cuda_binding(cards, card, pci)
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        if torch.cuda.device_count() != 1 or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid)) != canonical_gpu_uuid(card.uuid):
            raise RuntimeError("Policy CUDA device does not match the selected physical GPU1 UUID")
        torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"], 0)
        torch.manual_seed(config["seed"])
        torch.set_num_threads(4)
        manifest.update(gpu_uuid=card.uuid, physical_gpu_index=card.index,
                        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], cpu_preparation_seconds=time.monotonic() - started)
        save(run / "manifest.json", manifest)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation=config["attn_implementation"], output_loading_info=True)
        save(run / "loading-info.json", loading)
        if (loading.get("missing_keys") or loading.get("error_msgs")
                or any("mtp" not in key for key in loading.get("unexpected_keys", []))):
            raise ValueError("Official policy weights did not fully load")
        model.eval()
        calls = 0

        def generate(prompt):
            nonlocal calls
            if calls >= config["generation_cap"]:
                raise RuntimeError("Functional generation cap reached; no repair")
            calls += 1
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            length = inputs["input_ids"].shape[-1]
            if length > config["max_input_tokens"]:
                raise ValueError("Policy turn exceeds locked context cap; no truncation")
            began = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=config["max_new_tokens"],
                                        use_cache=True, pad_token_id=tokenizer.pad_token_id)
            ids = output[0, length:].tolist()
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            result = {"raw_output": tokenizer.decode(ids, skip_special_tokens=True),
                "complete": bool(ids and ids[-1] in eos_ids), "input_tokens": length,
                "output_tokens": len(ids), "seconds": time.monotonic() - began,
                "rendered_prompt_sha256": object_sha256(rendered), "physical_generation": calls}
            del inputs, output
            return result

        def emit(name, row):
            with (run / f"{name}.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if name == "progress":
                print(json.dumps({**row, "elapsed_seconds": time.monotonic() - started}), flush=True)

        result = run_functional_smoke(requests, generate=generate, emit=emit)
        result.update(source_commit=commit, config_sha256=file_sha256(config_path),
            elapsed_seconds=time.monotonic() - started, peak_vram_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_vram_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, input_hashes=provenance["rank_request_hashes"])
        save(run / "report.json", result)
        exit_code = 0 if result["status"] == "FUNCTIONAL_SMOKE_PASS" else 3
    except BaseException as error:
        save(run / "failure.json", {"error_type": type(error).__name__, "message": str(error),
                                   "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        model = None
        gc.collect()
        if card is not None and torch is not None:
            torch.cuda.empty_cache()
        save(run / "child-cleanup.json", {"process_exit_code": exit_code,
            "owned_model_unloaded": model is None, "gpu_uuid": card.uuid if card else None,
            "training_ready": False})
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
