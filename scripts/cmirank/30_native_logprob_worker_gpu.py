#!/usr/bin/env python3
"""Full native FSDP2 forward-only scorer; no optimizer or custom PPO worker."""

import argparse
from datetime import timedelta
import gc
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
from src.cmirank.corrected_rollout import load_corrected_config, native_microbatch, verify_corrected_preparation
from src.cmirank.gpu_resources import canonical_gpu_uuid
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_parity import compare_logprobs
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    require_allocation()
    c, path = load_corrected_config(ROOT)
    run = args.run_dir.resolve()
    if run != private / "runs" / c["run_id"]:
        raise ValueError("Wrong bounded native scorer run")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rows, _ = verify_corrected_preparation(private, ROOT, c, path, commit)
    if (commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Exact clean deployed scorer source required")
    observations = list(map(json.loads, (run / "generated.jsonl").read_text().splitlines()))
    manifest = json.loads((run / "manifest.json").read_text())
    if (len(observations) != 20 or [x["user_id"] for x in observations] != [r["user_id"] for r in rows]
            or file_sha256(run / "generated.jsonl") != manifest["generated_sha256"]):
        raise ValueError("All20 actual sampled responses required")
    for row, obs in zip(rows, observations):
        if obs["input_ids"][:obs["prompt_tokens"]] != row["input_ids"][:row["prompt_tokens"]]:
            raise ValueError("Scorer prompt differs from behavior prompt")
    def interrupted(signum, frame):
        raise InterruptedError(f"Owned native scorer interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    import torch
    import torch.distributed as dist
    from verl.workers.config.model import HFModelConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig
    from verl.trainer.config import CheckpointConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
                != canonical_gpu_uuid(os.environ["MEMREC_SELECTED_GPU_UUID"])):
        raise ValueError("Native scorer must bind only the authorized physical GPU1")
    torch.set_num_threads(4)
    torch.manual_seed(c["seed"])
    started = time.monotonic()
    engine = None
    try:
        dist.init_process_group(backend="nccl", init_method="file://" + str(run / "native-rendezvous"),
            rank=0, world_size=1, timeout=timedelta(seconds=120), device_id=torch.device("cuda:0"))
        compat, _ = load_compat_config(ROOT)
        checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
        model_config = HFModelConfig(path=str(checkpoint), load_tokenizer=False, use_remove_padding=False,
            use_fused_kernels=False, enable_gradient_checkpointing=False)
        engine_config = FSDPEngineConfig(strategy="fsdp2", model_dtype="bfloat16", forward_only=True,
            use_torch_compile=False, use_dynamic_bsz=False, infer_micro_batch_size_per_gpu=1)
        engine = FSDPEngineWithLMHead(model_config, engine_config, FSDPOptimizerConfig(), CheckpointConfig())
        engine.initialize()
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native_hf
        layer = engine.module.model.language_model.layers[0].linear_attn
        if (not native_hf.is_fast_path_available
                or layer.chunk_gated_delta_rule is not native_hf.chunk_gated_delta_rule
                or not layer.chunk_gated_delta_rule.__module__.startswith("fla.")
                or layer.causal_conv1d_fn is not native_hf.causal_conv1d_fn):
            raise ValueError("Full native scorer did not select the audited complete GDN/causal fast path")
        if engine.optimizer is not None or engine.lr_scheduler is not None:
            raise ValueError("No optimizer may be constructed during likelihood qualification")
        if sum(p.numel() for p in engine.module.parameters()) != 4539265536:
            raise ValueError("Full initializer parameter scope changed")
        if any(p.dtype != torch.bfloat16 for p in engine.module.parameters()):
            raise ValueError("Explicit native BF16 parameter profile required")
        results = []
        with engine.eval_mode(), torch.no_grad():
            for obs in observations:
                ids = torch.tensor(obs["input_ids"], device="cuda:0")
                data = native_microbatch(ids, c, torch)
                passes = []
                for _ in range(2):
                    _, result = engine.forward_step(data, loss_function=None, forward_only=True)
                    values = result["model_output"]["log_probs"].values()
                    response = values[obs["prompt_tokens"]-1:len(obs["input_ids"])-1]
                    if response.numel() != len(obs["response_token_ids"]) or not torch.isfinite(response).all():
                        raise ValueError("Native sampled response/EOS likelihood alignment failed")
                    passes.append(response.float().cpu().tolist())
                    del result, values, response
                repeat = compare_logprobs(*passes, c)
                if not repeat["passed"]:
                    raise ValueError("Same native training engine repeat exceeded original numerical bounds")
                results.append({"user_id": obs["user_id"], "old_logprobs": passes[0],
                    "recomputed_logprobs": passes[1], "reference_logprobs": passes[0],
                    "reference_scope": "same_initial_checkpoint_no_update_identity_not_separate_reference_worker",
                    "response_tokens_including_eos": len(passes[0]), "native_repeat": repeat})
                del data, ids
        stats = {"peak_vram_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                 "peak_vram_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
        # Unload before result serialization/CPU correction; parent audits actual release.
        engine = None
        gc.collect()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        (run / "native-logprobs.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in results))
        (run / "native-worker-report.json").write_text(artifact_json_dumps({
            "status": "NATIVE_FSDP2_FULL_MODEL_FORWARD_ONLY_PASS_OPTIMIZER_UNPROVEN",
            "source_commit": commit, "samples": 20, "scoring_passes": 40,
            "native_engine_initialized": True, "native_engine_strategy": "fsdp2",
            "model_dtype": "bfloat16", "optimizer_constructed": False, "optimizer_updates": 0,
            "separate_reference_worker_proven": False, "training_ready": False,
            "elapsed_seconds": time.monotonic()-started, **stats}))
    finally:
        engine = None
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
