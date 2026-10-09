#!/usr/bin/env python3
"""Twenty miniature official-Qwen class probes, not full 4B/PPO/GPU training.

Keep the official vocabulary/tokenizer; shrink hidden dimensions for CPU only.
Use upstream VeRL GAE and TRL's existing value head, not a custom PPO optimizer.
No Books files/outcome labels, 4B weight load, rollout or model update.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.ppo_runtime import response_reward_layout
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.rewards import mpss_rewards, ndcg_at_k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    require_allocation()
    config = json.loads((ROOT / "configs/cmirank/ppo_runtime_v1.json").read_text())
    run = args.run_dir.resolve()
    if (run != private / "runs" / config["run_id"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 4
            or Path(sys.prefix).resolve() != private / "envs" / config["environment_name"]):
        raise ValueError("Scoped isolated CPU-only environment required")
    for name, version in config["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Wrong pinned {name} version")
    started = time.monotonic()
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    from verl.utils.model import get_hf_auto_model_class, patch_valuehead_model
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if torch.cuda.is_initialized():
        raise ValueError("CPU probe initialized CUDA")
    model_path = private / f"models/Qwen3.5-4B-{config['model_revision'][:7]}-hnv"
    if file_sha256(model_path / "download-complete-hnv.json") != config["checkpoint_marker_sha256"]:
        raise ValueError("Official checkpoint marker changed")
    marker = json.loads((model_path / "download-complete-hnv.json").read_text())
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if file_sha256(model_path / name) != marker["sha256"][name]:
            raise ValueError("Official config/tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    official = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    auto_class = get_hf_auto_model_class(official)
    if auto_class is not AutoModelForImageTextToText or official.model_type != "qwen3_5":
        raise ValueError("VeRL did not select the official multimodal Qwen3.5 class")
    with torch.device("meta"):
        full_meta = auto_class.from_config(official, attn_implementation="sdpa")
    full_scope = {"text_parameters": sum(p.numel() for n, p in full_meta.named_parameters() if ".visual." not in n),
                  "vision_parameters": sum(p.numel() for n, p in full_meta.named_parameters() if ".visual." in n)}
    if full_scope != {"text_parameters": 4205751296, "vision_parameters": 333514240}:
        raise ValueError("Official full-text/vision shape differs from reviewed SFT boundary")
    del full_meta
    tiny = deepcopy(official.to_dict())
    tiny["text_config"].update(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [5, 5, 6], "mrope_interleaved": True})
    tiny["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64,
        num_heads=2, out_hidden_size=64, num_position_embeddings=16)
    mini_config = type(official).from_dict(tiny)
    actor = auto_class.from_config(mini_config, attn_implementation="sdpa").eval()
    reference = auto_class.from_config(deepcopy(mini_config), attn_implementation="sdpa").eval()
    reference.load_state_dict(actor.state_dict(), strict=True)
    for model in (actor, reference):
        for name, param in model.named_parameters():
            param.requires_grad_(model is actor and ".visual." not in name)
    critic_base = auto_class.from_config(deepcopy(mini_config), attn_implementation="sdpa").eval()
    critic_base.config.hidden_size = mini_config.text_config.hidden_size
    critic = AutoModelForCausalLMWithValueHead.from_pretrained(critic_base, summary_dropout_prob=0.0).eval()
    patch_valuehead_model(critic)
    for name, param in critic.named_parameters():
        if ".visual." in name:
            param.requires_grad_(False)
    rows = []
    labels = [f"C{i:02d}" for i in range(10)]
    for i in range(config["cpu_samples"]):
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Exclude one: C00, C01."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        response = tokenizer.encode(f"<answer>C{i % 2:02d}</answer>" + tokenizer.eos_token, add_special_tokens=False)
        tokens = torch.tensor([prompt_ids + response], dtype=torch.long)
        prefix = len(prompt_ids)
        actor_out = actor(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
        with torch.no_grad():
            ref_out = reference(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
        indices = tokens[:, prefix:]
        def logprobs(out):
            return out.logits[:, prefix - 1:-1].log_softmax(-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        actor_lp, ref_lp = logprobs(actor_out), logprobs(ref_out)
        delta = (actor_lp.detach() - ref_lp).abs().max().item()
        if delta > 1e-6 or not torch.isfinite(actor_lp).all():
            raise ValueError("Identical-weight CPU actor/reference log probabilities differ")
        (-actor_lp.mean()).backward()
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
                   for p in actor.parameters() if p.requires_grad):
            raise ValueError("Actor has no finite nonzero text gradients")
        if any(p.grad is not None for p in reference.parameters()):
            raise ValueError("Reference accumulated gradients")
        actor.zero_grad(set_to_none=True)
        del actor_out, ref_out, actor_lp, ref_lp
        _, _, values = critic(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
        if values.shape != tokens.shape or not torch.isfinite(values).all():
            raise ValueError("Existing TRL/VeRL critic value head shape/nonfinite output")
        values[:, prefix:].square().mean().backward()
        if critic.v_head.summary.weight.grad is None or not torch.isfinite(critic.v_head.summary.weight.grad).all():
            raise ValueError("Critic value head gradients failed")
        if any(p.grad is not None for n, p in critic.named_parameters() if ".visual." in n):
            raise ValueError("Unused vision branch received gradients")
        critic.zero_grad(set_to_none=True)
        rewards, rank = mpss_rewards(labels, labels[:-1], labels[-1], labels[i % 10])
        placed, eos_positions = response_reward_layout([5 + j % 3 for j in range(9)], list(rewards))
        reward_tensor = torch.tensor([placed], dtype=torch.float32)
        _, returns = compute_gae_advantage_return(token_level_rewards=reward_tensor,
            values=torch.zeros_like(reward_tensor), response_mask=torch.ones_like(reward_tensor), gamma=1.0, lam=1.0)
        oracle = reward_tensor.flip(-1).cumsum(-1).flip(-1)
        error = (returns - oracle).abs().max().item()
        if error > 1e-6 or abs(sum(placed) - ndcg_at_k(rank)) > 1e-12:
            raise ValueError("VeRL GAE response-only episode continuity/MPSS return differs")
        rows.append({"sample": i, "prompt_tokens": prefix, "response_tokens": len(response),
                     "actor_ref_logprob_max_delta": delta, "gae_return_max_delta": error,
                     "synthetic_target_rank": rank, "action_eos_indices": eos_positions})
        if (i + 1) % 10 == 0:
            print(f"CPU miniature class/logprob/gradient/GAE probes: {i+1}/20", flush=True)
    if torch.cuda.is_initialized():
        raise ValueError("CPU probe unexpectedly initialized CUDA")
    report = {"status": "MINIATURE_QWEN_CLASS_CPU_PASS_NOT_FULL_MODEL_OR_PPO_PROOF",
        "cpu_samples": len(rows), "official_tokenizer_revision": config["model_revision"],
        "full_model_meta_shapes": full_scope, "miniature_only": True,
        "actor_class": type(actor).__name__, "critic_wrapper": type(critic).__name__,
        "full_model_weights_loaded": False, "gpu_initialized": False, "ppo_updates": 0,
        "books_outcomes_accessed": False, "training_ready": False,
        "production_valuehead_loader_tested": False, "vllm_rollout_tested": False,
        "probe_rows": rows, "remaining_gates": config["not_proven_by_cpu_gate"],
        "elapsed_seconds": time.monotonic() - started}
    (run / "miniature-cpu-report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "cpu_samples", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
