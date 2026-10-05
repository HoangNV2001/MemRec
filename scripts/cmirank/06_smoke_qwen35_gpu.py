#!/usr/bin/env python3
"""One-H100 model/format/backward smoke, without Books labels or PPO claims."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cmirank.labels import make_labels
from src.cmirank.parser import parse_action
from src.cmirank.prompts import render_step_prompt
from src.cmirank.provenance import artifact_json_dumps
from src.cmirank.smoke_fixtures import synthetic_rank_requests


def save_json(path: Path, value: object) -> None:
    path.write_text(artifact_json_dumps(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Require an authorized single-GPU Slurm step")
    storage = Path("/mnt/data/users/hoangnv242/memrec-hnv")
    if not args.run_dir.resolve().is_relative_to(storage / "runs"):
        raise RuntimeError("Run directory outside MemRec-owned storage")
    config_path = ROOT / "configs/cmirank/qwen35_gpu_smoke.json"
    config = json.loads(config_path.read_text())
    contract = json.loads((ROOT / config["model_contract"]).read_text())
    model_path = storage / f"models/Qwen3.5-4B-{contract['model_revision'][:7]}-hnv"
    marker = json.loads((model_path / "download-complete-hnv.json").read_text())
    if marker["model_revision"] != contract["model_revision"]:
        raise RuntimeError("Model revision differs from approved contract")
    if config["scope"] != "synthetic_infrastructure_only" or config["batch_size"] != 1:
        raise RuntimeError("Unexpected smoke scope/batch")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "process.pid").write_text(str(os.getpid()) + "\n")
    save_json(args.run_dir / "config.json", config)
    save_json(args.run_dir / "model-contract.json", contract)
    started = time.monotonic()

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke must see exactly one GPU")
    properties = torch.cuda.get_device_properties(0)
    if "H100" not in properties.name or properties.total_memory < 79_000_000_000:
        raise RuntimeError("Selected device is not an 80 GB H100")
    torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"], 0)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    save_json(args.run_dir / "manifest.json", {
        "run_id": config["run_id"], "status": "running",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
        "model_id": contract["model_id"], "model_revision": contract["model_revision"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": properties.name, "gpu_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "research_checkpoint_created": False, "books_outcomes_accessed": False,
        "checkpoint_sha256": marker["sha256"],
    })
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
    )
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=config["attn_implementation"], output_loading_info=True,
    )
    save_json(args.run_dir / "loading-info.json", loading)
    if loading.get("missing_keys") or loading.get("error_msgs"):
        raise RuntimeError("Official checkpoint did not fully initialize model")
    unexpected = [key for key in loading.get("unexpected_keys", []) if "mtp" not in key]
    if unexpected:
        raise RuntimeError("Unexpected non-MTP checkpoint keys")
    vision_parameters = 0
    for name, parameter in model.named_parameters():
        if ".visual." in name:
            parameter.requires_grad_(False)
            vision_parameters += parameter.numel()
    if not vision_parameters:
        raise RuntimeError("Cannot establish vision/text training boundary")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"MODEL LOADED: text_trainable={trainable} frozen_vision={vision_parameters}", flush=True)
    requests = synthetic_rank_requests(config["examples"], config["seed"])
    labels = make_labels(range(10))
    prompt_tokens = []
    valid = 0
    responses_path = args.run_dir / "responses.jsonl"
    first_input = None
    model.eval()
    inference_started = time.monotonic()
    with responses_path.open("w") as output:
        for index, request in enumerate(requests):
            prompt = render_step_prompt(request, labels)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=config["enable_thinking"],
            )
            inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")
            length = inputs["input_ids"].shape[-1]
            if length > config["max_input_tokens"]:
                raise RuntimeError("Smoke prompt exceeds locked context cap; no truncation")
            prompt_tokens.append(length)
            if first_input is None:
                first_input = inputs["input_ids"].detach().clone()
            step_started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=config["do_sample"],
                    max_new_tokens=config["max_new_tokens"], use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            raw = tokenizer.decode(generated[0, length:], skip_special_tokens=True)
            action = parse_action(raw, labels)
            valid += int(action.valid)
            output.write(json.dumps({
                "sample": index, "rank_request_sha256": request.sha256(),
                "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "prompt_tokens": length, "output_tokens": generated.shape[-1] - length,
                "raw_output": raw, "valid": action.valid, "label": action.label,
                "failure_reason": action.failure_reason,
                "seconds": time.monotonic() - step_started,
            }) + "\n")
            output.flush()
            del generated, inputs
            if (index + 1) % 5 == 0:
                print(f"FORMAT SMOKE: {index + 1}/{len(requests)} valid={valid}", flush=True)

    inference_seconds = time.monotonic() - inference_started
    # One arbitrary answer teaches only format for an infrastructure update;
    # there is no positive/negative outcome label in these synthetic fixtures.
    answer_ids = tokenizer.encode("<answer>C09</answer>" + tokenizer.eos_token,
                                  add_special_tokens=False)
    suffix = torch.tensor([answer_ids], device="cuda:0")
    input_ids = torch.cat([first_input, suffix], dim=1)
    supervision = input_ids.clone()
    supervision[:, :first_input.shape[-1]] = -100
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["optimizer_lr"], foreach=False,
    )
    backward_started = time.monotonic()
    result = model(input_ids=input_ids, labels=supervision, use_cache=False)
    loss = result.loss
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite text-backbone loss")
    loss_value = loss.detach().float().item()
    loss.backward()
    del result, loss
    gradient_parameters = 0
    nonzero_gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError("Nonfinite text-backbone gradient")
            gradient_parameters += parameter.numel()
            nonzero_gradient_tensors += int(torch.count_nonzero(parameter.grad).item() > 0)
    if not nonzero_gradient_tensors:
        raise RuntimeError("Backward produced no learning signal")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    summary = {
        "status": "INFRASTRUCTURE_SMOKE_PASS" if valid == len(requests) else "FORMAT_SMOKE_FAILED",
        "format_valid": valid, "format_total": len(requests),
        "trainable_text_parameters": trainable, "frozen_vision_parameters": vision_parameters,
        "gradient_parameters": gradient_parameters,
        "nonzero_gradient_tensors": nonzero_gradient_tensors,
        "format_surrogate_loss": loss_value, "optimizer_updates_discarded": 1,
        "inference_seconds": inference_seconds,
        "backward_optimizer_seconds": time.monotonic() - backward_started,
        "prompt_tokens_min": min(prompt_tokens), "prompt_tokens_max": max(prompt_tokens),
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "elapsed_seconds": time.monotonic() - started,
        "ppo_proven": False, "research_checkpoint_created": False,
        "books_quality_measured": False,
    }
    save_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    del optimizer, model, first_input, input_ids, supervision, suffix
    gc.collect()
    torch.cuda.empty_cache()
    if valid != len(requests):
        raise SystemExit(3)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # No credentials or external data are present in this infrastructure run.
        if "--run-dir" in sys.argv:
            failure_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1])
            if failure_dir.is_dir():
                save_json(failure_dir / "failure.json", {
                    "type": type(error).__name__, "message": str(error),
                })
        raise
