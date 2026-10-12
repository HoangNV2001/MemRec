#!/usr/bin/env python3
"""Owned HF forward-only worker; exits fully before the vLLM service loads."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager, nullcontext

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.gpu_resources import canonical_gpu_uuid
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_parity import load_parity_config, parity_code_hashes, verify_parity_preparation
from src.cmirank.provenance import artifact_json_dumps, file_sha256


@contextmanager
def ieee_matmul(torch):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def precision_diagnosis(model, rows, run, config, torch, output_name="precision-logprobs.jsonl"):
    """All five fixed controls, no optimizer/profile selection or weight save."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    report = {"variants": config["diagnostic_variants"], "samples_per_variant": len(rows),
              "native_gdn_chunk_callable": model.model.language_model.layers[0].linear_attn.chunk_gated_delta_rule.__name__,
              "native_fast_path_available": native.is_fast_path_available,
              "tf32_matmul_at_baseline": torch.backends.cuda.matmul.allow_tf32,
              "primary_profile_selected": False, "optimizer_updates": 0, "generation_calls": 0}
    def record(handle, variant, row, logits, ids):
        probabilities = logits.float().log_softmax(-1).gather(
            -1, ids[:, row["prompt_tokens"]:].unsqueeze(-1)).squeeze(-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite precision-control response/EOS probabilities")
        handle.write(json.dumps({"variant": variant, "user_id": row["user_id"],
                                 "logprobs": probabilities[0].cpu().tolist()}) + "\n")
        handle.flush()
    device = next(model.parameters()).device
    with torch.no_grad(), (run / output_name).open("x") as handle:
        for row in rows:
            hidden = {}
            def capture(module, arguments):
                hidden["value"] = arguments[0][:, row["prompt_tokens"] - 1:-1].detach()
            hook = model.lm_head.register_forward_pre_hook(capture)
            try:
                ids = torch.tensor([row["input_ids"]], device=device)
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                record(handle, "baseline_bf16_fa2", row, output.logits[:, row["prompt_tokens"] - 1:-1], ids)
                # Only the readout shape/precision changes; same captured BF16 backbone.
                logits = torch.nn.functional.linear(hidden["value"], model.lm_head.weight)
                record(handle, "readout_bf16_response_only", row, logits, ids)
                del logits
                with ieee_matmul(torch):
                    logits = torch.nn.functional.linear(hidden["value"].float(), model.lm_head.weight.float())
                record(handle, "readout_fp32_response_only", row, logits, ids)
                del ids, output, logits
            finally:
                hook.remove()
                hidden.clear()
        model.set_attn_implementation("sdpa")
        for variant in ("bf16_sdpa", "fp32_sdpa"):
            if variant == "fp32_sdpa":
                # Ephemeral diagnostic cast of the SAME values, never an initializer save.
                model.to(dtype=torch.float32)
            context = ieee_matmul(torch) if variant == "fp32_sdpa" else nullcontext()
            with context:
                for row in rows:
                    ids = torch.tensor([row["input_ids"]], device=device)
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    record(handle, variant, row, output.logits[:, row["prompt_tokens"] - 1:-1], ids)
                    del ids, output
    return report


def rehearse_precision_cpu(official, run, config):
    """Native miniature SDPA proxy for control flow/hooks/dtype, not FA2 proof."""
    from copy import deepcopy
    import torch
    from transformers import AutoModelForImageTextToText
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tiny = deepcopy(official.to_dict())
    tiny["text_config"].update(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0,
                         "mrope_section": [5, 5, 6], "mrope_interleaved": True})
    tiny["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64,
        num_heads=2, out_hidden_size=64, num_position_embeddings=16)
    model = AutoModelForImageTextToText.from_config(type(official).from_dict(tiny),
        attn_implementation="sdpa", dtype=torch.bfloat16).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    rows = [{"user_id": -20001-i, "input_ids": [1, 3+i, 4, 5, 6, 7, 8, 2], "prompt_tokens": 3}
            for i in range(20)]
    report = precision_diagnosis(model, rows, run, config, torch, "precision-cpu-logprobs.jsonl")
    model = None
    del parameter
    gc.collect()
    if torch.cuda.is_initialized():
        raise ValueError("CPU precision-control rehearsal initialized CUDA")
    report.update(scope="MINIATURE_SDPA_CPU_CONTROL_FLOW_ONLY_BASELINE_FA2_NOT_EXECUTED",
                  official_weights_loaded=False, gpu_requested=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    require_allocation()
    config, path = load_parity_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    run = args.run_dir.resolve()
    if (run != private / "runs" / config["run_id"] or not run.is_dir()
            or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit()
            or not os.environ.get("MEMREC_SELECTED_GPU_UUID")):
        raise ValueError("Only the owned UUID-bound HF worker may initialize CUDA")
    rows, _ = verify_parity_preparation(private / "runs" / config["prepare_run_id"], config,
        commit=os.environ["MEMREC_EXPECTED_COMMIT"], config_sha=file_sha256(path), code_sha=parity_code_hashes(ROOT))
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    if (torch.cuda.device_count() != 1 or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
            != canonical_gpu_uuid(os.environ["MEMREC_SELECTED_GPU_UUID"])):
        raise ValueError("HF worker logical CUDA0 does not match the selected physical GPU UUID")
    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    started = time.monotonic()
    model = None
    try:
        checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
        model, info = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint,
            local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="flash_attention_2", output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("HF parity initializer loading keys changed")
        scope = {"text": 0, "vision": 0}
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(False)
            if parameter.dtype != torch.bfloat16:
                raise ValueError("Parity model is not the pinned BF16 parameterization")
            scope["vision" if ".visual." in name else "text"] += parameter.numel()
        if scope != {"text": 4205751296, "vision": 333514240}:
            raise ValueError("Full official text/frozen-vision scope changed")
        model.eval()
        diagnostic = None
        if config.get("hf_precision_diagnostic"):
            diagnostic = precision_diagnosis(model, rows, run, config, torch)
        else:
            with torch.no_grad(), (run / "hf-logprobs.jsonl").open("x") as handle:
                for row in rows:
                    ids = torch.tensor([row["input_ids"]], device="cuda:0")
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    logits = output.logits[:, row["prompt_tokens"] - 1:-1].float()
                    logprobs = logits.log_softmax(-1).gather(-1, ids[:, row["prompt_tokens"]:].unsqueeze(-1)).squeeze(-1)
                    if not torch.isfinite(logprobs).all():
                        raise ValueError("HF response/EOS logprobs are nonfinite")
                    handle.write(json.dumps({"user_id": row["user_id"], "logprobs": logprobs[0].cpu().tolist()}) + "\n")
                    handle.flush()
                    del ids, output, logits, logprobs
        report = {"samples": len(rows), "elapsed_seconds": time.monotonic() - started, "scope": scope,
                  "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "optimizer_updates": 0, "generation_calls": 0}
        if diagnostic:
            report["precision_diagnosis"] = diagnostic
        model = None
        del parameter
        gc.collect()
        torch.cuda.empty_cache()
        (run / "hf-worker-report.json").write_text(artifact_json_dumps(report))
    finally:
        model = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
