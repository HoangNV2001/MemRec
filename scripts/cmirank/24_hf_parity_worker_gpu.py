#!/usr/bin/env python3
"""Owned HF forward-only worker; exits fully before the vLLM service loads."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.gpu_resources import canonical_gpu_uuid
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_parity import load_parity_config, parity_code_hashes, verify_parity_preparation
from src.cmirank.provenance import artifact_json_dumps, file_sha256


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
