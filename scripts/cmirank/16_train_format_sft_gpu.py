#!/usr/bin/env python3
"""Bounded full-text format SFT; GPU0 only, no Books outcome or PPO update."""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.format_sft import (
    format_code_hashes, load_format_config, load_prepared_format_data, verify_sft_smoke,
)
from src.cmirank.gpu_resources import canonical_gpu_uuid, select_reserved_gpu, verified_numeric_cuda_binding
from src.cmirank.memory_smoke import object_sha256
from src.cmirank.policy_smoke import load_secondary_inputs, run_functional_smoke
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.request import RankRequest
from src.cmirank.reserved_handoff import gpu_snapshot, handoff


def save(path, value):
    path.write_text(artifact_json_dumps(value))


def train(model, rows, torch, config, phase, emit):
    vision = 0
    for name, parameter in model.named_parameters():
        if ".visual." in name:
            parameter.requires_grad_(False)
            vision += parameter.numel()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if vision != 333514240 or sum(p.numel() for p in parameters) != 4205751296:
        raise ValueError("Previously audited full-text/frozen-vision boundary changed")
    if any(p.dtype != torch.bfloat16 for p in parameters):
        raise ValueError("Declared BF16 full-text parameter/state recipe changed")
    if not hasattr(model, "lm_head"):
        raise ValueError("Cannot audit actual text-head weight update")
    response_ids = sorted({token for row in rows for token in row["input_ids"][row["prompt_tokens"]:]})
    head_indices = torch.tensor(response_ids, device="cuda:0")
    before = model.lm_head.weight.detach().index_select(0, head_indices).float().cpu()
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], betas=tuple(config["betas"]),
        eps=config["eps"], weight_decay=config["weight_decay"], foreach=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    optimizer.zero_grad(set_to_none=True)
    updates, losses, total_examples = 0, [], 0
    accumulation = config[f"{phase}_gradient_accumulation"]
    began = time.monotonic()
    optimizer_dtype = None
    for epoch in range(config[f"{phase}_epochs"]):
        order = list(range(len(rows)))
        random.Random(config["seed"] + epoch).shuffle(order)
        for micro, index in enumerate(order):
            row = rows[index]
            ids = torch.tensor([row["input_ids"]], device="cuda:0")
            labels = torch.tensor([row["labels"]], device="cuda:0")
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
            loss = result.loss
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite structural SFT loss")
            loss_value = loss.detach().float().item()
            (loss / accumulation).backward()
            losses.append(loss_value)
            total_examples += 1
            del loss, result, ids, labels
            if (micro + 1) % accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
                if not torch.isfinite(norm).item() or norm.item() <= 0:
                    raise ValueError("Missing/nonfinite text-backbone gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                state = next(s for s in optimizer.state.values() if "exp_avg" in s)
                optimizer_dtype = str(state["exp_avg"].dtype)
                if optimizer_dtype != "torch.bfloat16" or str(state["exp_avg_sq"].dtype) != optimizer_dtype:
                    raise ValueError("Declared BF16 AdamW state recipe changed")
                emit("training", {"epoch": epoch, "optimizer_update": updates, "examples_processed": total_examples,
                    "mean_microbatch_loss": sum(losses[-accumulation:]) / accumulation,
                    "gradient_norm_before_clipping": float(norm.item()), "elapsed_seconds": time.monotonic() - began})
    if updates != config[f"{phase}_updates"]:
        raise ValueError("Fixed warm-start optimizer budget differs")
    delta = (model.lm_head.weight.detach().index_select(0, head_indices).float().cpu() - before).abs().max().item()
    if delta == 0 or any(p.grad is not None for name, p in model.named_parameters() if ".visual." in name):
        raise ValueError("No actual text update or frozen vision received gradients")
    summary = {"optimizer_updates": updates, "examples_processed": total_examples,
        "first_microbatch_loss": losses[0], "last_microbatch_loss": losses[-1],
        "trainable_text_parameters": sum(p.numel() for p in parameters), "frozen_vision_parameters": vision,
        "response_head_weight_delta_max": delta, "optimizer_state_dtype": optimizer_dtype,
        "train_seconds": time.monotonic() - began}
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    del optimizer, parameters, before, head_indices, state
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def checkpoint_probe(model, rows, torch):
    logits = []
    with torch.inference_mode():
        for row in rows[:20]:
            ids = torch.tensor([row["input_ids"][:row["prompt_tokens"]]], device="cuda:0")
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits.append(output.logits[0, -1].float().cpu())
            del output, ids
    return torch.stack(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--owner-confirmed-generator-step", required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    config, path = load_format_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private / "runs" / config[f"{args.phase}_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 4
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Fresh scoped CPU coordinator and exact clean deployed source required")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    if versions["transformers"] != "5.13.0" or versions["torch"] not in ("2.8.0", "2.8.0+cu128"):
        raise ValueError("Existing isolated SFT runtime changed")
    base = private / f"models/Qwen3.5-4B-{config['model_revision'][:7]}-hnv"
    data = private / "runs" / config["data_run_id"]
    rows, data_receipt = load_prepared_format_data(data, commit=commit, config_sha=file_sha256(path),
        code_sha=format_code_hashes(ROOT), marker_sha=file_sha256(base / "download-complete-hnv.json"), config=config)
    if args.phase == "full":
        smoke = verify_sft_smoke(private / "runs" / config["smoke_run_id"], commit=commit,
            config_sha=file_sha256(path), data_report_sha=file_sha256(data / "report.json"))
    else:
        smoke = None
        rows = rows[:config["smoke_train_examples"]]
    synthetic = [RankRequest.from_dict(json.loads(s)) for s in (data / "synthetic-holdout.jsonl").read_text().splitlines()]
    real = None
    if args.phase == "full":
        diagnostic = json.loads((ROOT / config["real_diagnostic_contract"]).read_text())
        real, proof = load_secondary_inputs(private / "runs" / diagnostic["memory_run_id"],
            private / "runs" / diagnostic["memory_review_id"] / "review.json", expected_review_sha=diagnostic["memory_review_sha256"])
        if proof != data_receipt["secondary_books_provenance"]:
            raise ValueError("Read-only secondary diagnostic state changed")
    run.mkdir()
    started = time.monotonic()
    model, card, torch, exit_code = None, None, None, 1
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned SFT interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def emit(name, row):
        with (run / f"{name}.jsonl").open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if name == "training" and row["optimizer_update"] % 8 == 0:
            print(json.dumps({"event": "SFT_UPDATE", **row}), flush=True)
        if name.endswith("progress") and row["completed_users"] % 5 == 0:
            print(json.dumps({"event": name, **row}), flush=True)
    try:
        manifest = {"config": config, "config_sha256": file_sha256(path), "source_commit": commit,
            "phase": args.phase, "slurm_job_id": job, "versions": versions,
            "data_report_sha256": file_sha256(data / "report.json"), "code_sha256": format_code_hashes(ROOT),
            "checkpoint_marker_sha256": file_sha256(base / "download-complete-hnv.json"),
            "books_training_data_accessed": False, "ranking_metrics_computed": False,
            "ppo_proven": False, "training_ready": False, "matching_smoke": smoke}
        save(run / "manifest.json", manifest)
        manifest["gpu0_handoff"] = handoff(run, args.owner_confirmed_generator_step, gpu_index=0,
            protect_other_workloads=config.get("resource_policy") == "gpu0_only_gpu1_all_workloads_read_only")
        cards, apps, raw, raw_apps = gpu_snapshot()
        card = select_reserved_gpu(cards, set(apps), gpu_index=0)
        if card.uuid != manifest["gpu0_handoff"]["gpu0_uuid"]:
            raise ValueError("Handoff and selected physical GPU differ")
        (run / "gpus-before.csv").write_text(raw)
        (run / "apps-before.csv").write_text(raw_apps)
        cards, apps, raw, raw_apps = gpu_snapshot()
        current = select_reserved_gpu(cards, set(apps), gpu_index=0)
        if current.uuid != card.uuid:
            raise ValueError("Selected GPU changed before model initialization")
        card = current
        (run / "gpus-load-time.csv").write_text(raw)
        (run / "apps-load-time.csv").write_text(raw_apps)
        pci = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"], text=True)
        (run / "gpu-pci-load-time.csv").write_text(pci)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = verified_numeric_cuda_binding(cards, card, pci)
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        if torch.cuda.device_count() != 1 or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid)) != canonical_gpu_uuid(card.uuid):
            raise ValueError("CUDA logical0 is not the authorized physical GPU0")
        torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"], 0)
        torch.manual_seed(config["seed"])
        torch.set_num_threads(4)
        manifest.update(gpu_uuid=card.uuid, physical_gpu_index=card.index, cuda_visible_devices="0")
        save(run / "manifest.json", manifest)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        def load(model_path):
            loaded, info = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0},
                attn_implementation=config["attn_implementation"], output_loading_info=True)
            if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
                raise ValueError("Full official/checkpoint loading parity failed")
            loaded.eval()
            return loaded, info
        model, loading = load(base)
        save(run / "loading-info.json", loading)
        training = train(model, rows, torch, config, args.phase, emit)
        # Save/reload is GPU-task verification, not CPU artifact hashing. No
        # optimizer/gradients or second model remains resident across reload.
        before = checkpoint_probe(model, rows, torch)
        checkpoint = run / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(checkpoint)
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        model, loading = load(checkpoint)
        save(run / "reload-loading-info.json", loading)
        after = checkpoint_probe(model, rows, torch)
        logit_delta = (before - after).abs().max().item()
        if not torch.isfinite(after).all().item() or logit_delta > config["checkpoint_logit_atol"]:
            raise ValueError("Checkpoint changed the twenty audited full-vocabulary logits")
        del before, after
        calls = 0
        cohort_calls = 0
        def generate(prompt):
            nonlocal calls, cohort_calls
            if cohort_calls >= config["functional_generation_cap_per_cohort"]:
                raise ValueError("Format generation cap reached; no repair")
            cohort_calls += 1
            calls += 1
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            n = inputs["input_ids"].shape[-1]
            if n > config["max_input_tokens"]:
                raise ValueError("Functional prompt too long; no truncation")
            began = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, use_cache=True,
                    max_new_tokens=config["max_new_tokens"], pad_token_id=tokenizer.pad_token_id)
            tokens = output[0, n:].tolist()
            eos = model.generation_config.eos_token_id
            valid_eos = eos if isinstance(eos, list) else [eos]
            value = {"raw_output": tokenizer.decode(tokens, skip_special_tokens=True),
                "complete": bool(tokens and tokens[-1] in valid_eos), "input_tokens": n,
                "output_tokens": len(tokens), "seconds": time.monotonic() - began,
                "rendered_prompt_sha256": object_sha256(rendered), "physical_generation": calls}
            del output, inputs
            return value
        synthetic_result = run_functional_smoke(synthetic, generate=generate,
            emit=lambda name, row: emit(f"synthetic-{name}", row))
        real_result = None
        if real is not None:
            cohort_calls = 0
            real_result = run_functional_smoke(real, generate=generate,
                emit=lambda name, row: emit(f"secondary-books-{name}", row))
        status = "FORMAT_SFT_INFRASTRUCTURE_SMOKE_PASS" if args.phase == "smoke" else "FORMAT_SFT_FIXED_BUDGET_COMPLETE_REVIEW_REQUIRED"
        # Infrastructure smoke permits the predeclared full LEARNING schedule,
        # not quality promotion, even when five format updates aren't sufficient.
        report = {"status": status, "phase": args.phase, "source_commit": commit,
            "config_sha256": file_sha256(path), "data_report_sha256": file_sha256(data / "report.json"),
            **training, "checkpoint_roundtrip_pass": True, "checkpoint_probe_examples": 20,
            "checkpoint_max_absolute_logit_difference": logit_delta,
            "synthetic_functional_diagnostic": synthetic_result, "secondary_books_functional_diagnostic": real_result,
            "generation_calls": calls, "elapsed_seconds": time.monotonic() - started,
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "checkpoint_role": "discarded_infrastructure_smoke" if args.phase == "smoke" else "format_initialization_only_not_ppo_rank_checkpoint",
            "books_training_data_accessed": False, "ranking_metrics_computed": False,
            "primary_provider_replaced": False, "ppo_proven": False, "training_ready": False}
        save(run / "report.json", report)
        exit_code = 0
        print(json.dumps({k: report[k] for k in ("status", "optimizer_updates", "checkpoint_roundtrip_pass",
                           "generation_calls", "elapsed_seconds", "peak_vram_allocated_mib")}), flush=True)
    except BaseException as error:
        save(run / "failure.json", {"error_type": type(error).__name__, "message": str(error),
                                   "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        model = None
        gc.collect()
        if card is not None and torch is not None:
            torch.cuda.empty_cache()
        save(run / "child-cleanup.json", {"process_exit_code": exit_code, "owned_model_unloaded": model is None,
            "gpu_uuid": card.uuid if card else None, "training_ready": False})


if __name__ == "__main__":
    main()
