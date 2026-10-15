#!/usr/bin/env python3
"""CPU-only20-case native LoRA scope/critic adapter/round-trip qualification.

Official4B shapes are meta only. Tiny random fixtures are not Books training,
full-model optimizer/GAE/rollout or single-GPU fit evidence. Keepers stay running.
"""

import argparse
from copy import deepcopy
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.lora import load_lora_config, lora_code_hashes, linear_targets, audit_scope, make_critic_adapter
from src.cmirank.optimizer_smoke import allocation_memory, load_optimizer_config, verified_parents
from src.cmirank.ppo_compat import load_compat_config
from src.cmirank.ppo_runtime import load_runtime_config
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def miniature_config(official):
    tiny = deepcopy(official.to_dict())
    tiny["text_config"].update(vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0, "mrope_section": [5, 5, 6], "mrope_interleaved": True})
    tiny["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64,
        num_heads=2, out_hidden_size=64, num_position_embeddings=16)
    tiny.update(num_labels=1, classifier_dropout=0.0, id2label={0: "LABEL_0"}, label2id={"LABEL_0": 0})
    result = type(official).from_dict(tiny)
    result.num_labels = 1
    return result


def native_lora(module, shapes, c, *, adapter=None):
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    proxy = SimpleNamespace(model_config=SimpleNamespace(lora_rank=c["rank"], lora_alpha=c["alpha"],
        target_modules=sorted(shapes), target_parameters=None, exclude_modules=None,
        lora_adapter_path=str(adapter) if adapter else None, use_shm=False))
    return FSDPEngine._build_lora_module(proxy, module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, path = load_lora_config(ROOT)
    runtime, _ = load_runtime_config(ROOT)
    compat, _ = load_compat_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private / "runs" / c["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(sys.prefix).resolve() != private / "envs" / runtime["environment_name"]):
        raise ValueError("Fresh clean-source allocation-only CPU LoRA audit required")
    for name, version in {**runtime["versions"], "peft": "0.21.2"}.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned runtime changed")
    host = allocation_memory(job)
    if host["maximum_bytes"] - host["current_bytes"] < c["host_load_reserve_bytes"]:
        raise ValueError("Own job lacks the fixed20GiB CPU load margin")
    full, _ = load_optimizer_config(ROOT)
    observations, _, parent = verified_parents(private, full)
    checkpoint = private / "runs" / compat["sft_run_id"] / "checkpoint"
    for n, sha in compat["checkpoint_files_sha256"].items():
        if file_sha256(checkpoint/n) != sha:
            raise ValueError("Frozen format initializer changed")
    for n, sha in parent["native_source_sha256"].items():
        if file_sha256(Path(sys.prefix)/n) != sha:
            raise ValueError("Reviewed native source changed")
    began = time.monotonic()
    run.mkdir()
    (run / "manifest.json").write_text(artifact_json_dumps({"source_commit": commit,
        "config": c, "config_sha256": file_sha256(path), "code_sha256": lora_code_hashes(ROOT),
        "job_id": job, "gpu_requested": False, "training_ready": False}))
    try:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForTokenClassification
        from peft import get_peft_model_state_dict, PeftModel
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
        torch.set_num_threads(4)
        torch.manual_seed(c["seed"])
        official = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        official.num_labels = 1
        official.classifier_dropout = 0.0
        scopes, targets = {}, {}
        with torch.device("meta"):
            meta_actor = AutoModelForImageTextToText.from_config(official, dtype=torch.float32, attn_implementation="sdpa")
            meta_critic = AutoModelForTokenClassification.from_config(official, dtype=torch.float32, attn_implementation="sdpa")
        targets["actor"], targets["critic"] = linear_targets(meta_actor), linear_targets(meta_critic)
        if targets["actor"] != targets["critic"]:
            raise ValueError("Actor and critic must adapt the identical language linear scope")
        meta_actor = native_lora(meta_actor, targets["actor"], c)
        scopes["actor"] = audit_scope(meta_actor, targets["actor"], c, "actor")
        del meta_actor
        torch.manual_seed(c["seed"])
        meta_critic, scopes["critic"] = make_critic_adapter(meta_critic, targets["critic"], c,
            run / "critic-initial-adapter", materialize_meta=True)
        del meta_critic
        gc.collect()
        mini = miniature_config(official)
        probes, wrapper_control = [], None
        for role, factory in (("actor", AutoModelForImageTextToText), ("critic", AutoModelForTokenClassification)):
            torch.manual_seed(c["seed"])
            base = factory.from_config(deepcopy(mini), dtype=torch.float32, attn_implementation="sdpa")
            shapes = linear_targets(base)
            # Probe the ordinary fresh-CAUSAL_LM path too; do not silently
            # mistake its success on actors for native value-model support.
            if role == "critic":
                try:
                    control = native_lora(deepcopy(base), shapes, c)
                    output = control(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
                    wrapper_control = {"passed": tuple(output.logits.shape) == (1, 3, 1)}
                    del control, output
                except (AttributeError, TypeError, ValueError) as error:
                    wrapper_control = {"passed": False, "error_type": type(error).__name__, "error": str(error)}
                initialized, _ = make_critic_adapter(deepcopy(base), shapes, c, run / "tiny-critic-initial-adapter")
                del initialized
                model = native_lora(base, shapes, c, adapter=run / "tiny-critic-initial-adapter")
            else:
                model = native_lora(base, shapes, c)
            scope = audit_scope(model, shapes, c, role)
            model.eval()
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=c[f"{role}_lr"], foreach=False)
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            for i, obs in enumerate(observations):
                # Deliberately miniature synthetic fixtures, not token-truncated
                # Books examples or replacement production EOS/mask evidence.
                tokens = torch.tensor([[1, 2+i, 25+i, 50+i, 3]])
                output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
                logits = output.logits.float()
                if role == "actor":
                    loss = torch.nn.functional.cross_entropy(logits[:, -1], torch.tensor([70+i]))
                else:
                    if tuple(logits.shape) != (1, 5, 1):
                        raise ValueError("Generic native critic adapter changed scalar-value readout")
                    loss = (logits[:, -1, 0] - (1 if i % 2 == 0 else -1)).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite miniature LoRA loss")
                (loss/20).backward()
                probes.append({"role": role, "user_id": obs["user_id"], "fixture_tokens": tokens.tolist()[0],
                    "miniature_loss": float(loss.detach()), "full_sequence_replayed": False})
            trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in trainable):
                raise ValueError("Missing/nonfinite adapter or head gradient")
            if not any(bool(p.grad.abs().max() > 0) for n, p in trainable if ".lora_" in n):
                raise ValueError("Head-only training is not a LoRA critic pass")
            if role == "critic" and not any(bool(p.grad.abs().max() > 0) for n, p in trainable if ".score." in n):
                raise ValueError("Value head did not receive gradients")
            optimizer.step()
            if not any(not torch.equal(p, before[n]) for n, p in trainable if ".lora_" in n):
                raise ValueError("No adapter update occurred")
            if any(p.grad is not None or not torch.equal(p, frozen[n])
                   for n, p in model.named_parameters() if not p.requires_grad):
                raise ValueError("Frozen base/vision changed during adapter optimization")
            saved = run / f"tiny-{role}-updated-adapter"
            model.save_pretrained(saved, safe_serialization=True, save_embedding_layers=False)
            original_state = {n: t.detach().clone() for n, t in get_peft_model_state_dict(model).items()}
            # Native pretrained-adapter path on a fresh unmodified model.
            torch.manual_seed(c["seed"])
            fresh = factory.from_config(deepcopy(mini), dtype=torch.float32, attn_implementation="sdpa")
            restored = native_lora(fresh, shapes, c, adapter=saved)
            if audit_scope(restored, shapes, c, role)["trainable_shapes"] != scope["trainable_shapes"]:
                raise ValueError("Native adapter reload changed trainable scope")
            restored_state = get_peft_model_state_dict(restored)
            if set(original_state) != set(restored_state) or any(not torch.equal(t, restored_state[n]) for n, t in original_state.items()):
                raise ValueError("Adapter/head roundtrip changed tensor values")
            restored.eval()
            with torch.no_grad():
                for i, obs in enumerate(observations):
                    tokens = torch.tensor([[1, 2+i, 25+i, 50+i, 3]])
                    original_logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
                    restored_logits = restored(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
                    if not torch.equal(original_logits, restored_logits):
                        raise ValueError("Miniature adapter/head reload changed actual forward logits")
                    next(p for p in probes if p["role"] == role and p["user_id"] == obs["user_id"])["reload_max_logit_delta"] = 0.0
            del original_logits, restored_logits
            del model, base, restored, fresh, optimizer, frozen, before, original_state, restored_state, output, logits, loss, trainable
            gc.collect()
        if torch.cuda.is_initialized():
            raise ValueError("CPU audit unexpectedly initialized CUDA")
        (run / "probes.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in probes))
        report = {"status": "LORA_NATIVE_CPU20_PASS_FULL_GPU_AND_MULTITURN_GATES_REMAIN",
            "source_commit": commit, "config_sha256": file_sha256(path), "code_sha256": lora_code_hashes(ROOT),
            "samples_per_role": 20, "roles": scopes, "targets": targets,
            "critic_fresh_causal_wrapper_control": wrapper_control,
            "critic_production_route": "native_pretrained_generic_peft_adapter_with_score_modules_to_save",
            "native_source_sha256": {str(Path(inspect.getfile(k)).relative_to(Path(sys.prefix))):
                file_sha256(Path(inspect.getfile(k))) for k in (FSDPEngine, PeftModel, get_peft_model_state_dict)},
            "parent_generated_sha256": full["generated_sha256"], "allocation_memory": allocation_memory(job),
            "official_full_model_meta_only": True, "full_weights_loaded": False, "miniature_updates_per_role": 1,
            "physical_requests": 0, "books_outcomes_accessed": False, "ranking_metrics_computed": False,
            "gpu_requested": False, "keeper_stopped": False, "training_ready": False,
            "artifact_sha256": {str(p.relative_to(run)): file_sha256(p) for p in run.rglob("*") if p.is_file()},
            "elapsed_seconds": time.monotonic()-began}
        (run / "report.json").write_text(artifact_json_dumps(report))
        print(json.dumps({k: report[k] for k in ("status", "samples_per_role", "elapsed_seconds")}), flush=True)
    except BaseException as error:
        (run / "failure.json").write_text(artifact_json_dumps({"error_type": type(error).__name__, "error": str(error),
            "source_commit": commit, "elapsed_seconds": time.monotonic()-began, "gpu_requested": False}))
        raise


if __name__ == "__main__":
    main()
