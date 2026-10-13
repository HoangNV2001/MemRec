"""Approved generated-rollout qualification; imports never select CUDA."""

import json
import math
from pathlib import Path

from .ppo_parity import load_parity_config, server_options
from .provenance import file_sha256


def load_corrected_config(root):
    path = root / "configs/cmirank/ppo_corrected_rollout_v1.json"
    c = json.loads(path.read_text())
    expected = {"schema_version": 1, "samples": 20, "physical_request_cap": 20,
        "temperature": 0.9, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "max_new_tokens": 64, "seed": 42, "port": 18124,
        "logprobs_mode": "processed_logprobs", "rollout_is": "sequence",
        "rollout_is_threshold": 2.0, "rollout_is_batch_normalize": False,
        "rollout_rs": None, "bypass_mode": False, "loss_type": "ppo_clip",
        "native_engine_strategy": "fsdp2", "forward_only": True,
        "model_dtype": "bfloat16", "use_remove_padding": False,
        "use_fused_kernels": False, "use_torch_compile": False,
        "gpu_index": 1, "tensor_parallel_size": 1, "timeout_minutes": 25,
        "optimizer_updates": 0, "training_ready": False,
        "books_outcomes_accessed": False, "ranking_metrics_computed": False}
    if any(c.get(k) != v or type(c.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("Corrected rollout is a fixed20 no-update qualification, not tuning")
    extra = {"prepare_run_id", "run_id", "review_run_id", "researcher_decision",
        "parent_prepare_run_id", "parent_prepare_report_sha256", "parent_diagnostic_run_id",
        "parent_diagnostic_report_sha256", "tokens_sha256", "acceptance"}
    if set(c) != set(expected) | extra:
        raise ValueError("No hidden server/profile/sampler override permitted")
    native, _ = load_parity_config(root, 8)
    native.update(c)
    return native, path


def corrected_code_hashes(root):
    return {n: file_sha256(root / n) for n in (
        "src/cmirank/corrected_rollout.py", "scripts/cmirank/28_prepare_corrected_rollout_cpu.py",
        "scripts/cmirank/29_smoke_corrected_rollout_gpu.py", "scripts/cmirank/30_native_logprob_worker_gpu.py",
        "scripts/cmirank/31_review_corrected_rollout_cpu.py", "src/cmirank/ppo_parity.py",
        "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py",
        "src/cmirank/reserved_handoff.py", "src/cmirank/gpu_resources.py",
        "scripts/cmirank/run_real_policy_smoke.sh", "scripts/cmirank/run_real_policy_with_keeper.sh")}


def generated_request(row, c):
    return {"model": c["served_model_name"], "prompt": row["input_ids"][:row["prompt_tokens"]],
        "max_tokens": c["max_new_tokens"], "temperature": c["temperature"],
        "top_p": c["top_p"], "top_k": c["top_k"], "min_p": c["min_p"],
        "seed": c["seed"], "n": 1, "stream": False, "echo": False,
        "logprobs": 0, "return_token_ids": True, "add_special_tokens": False,
        "skip_special_tokens": False, "ignore_eos": False,
        "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def corrected_server_options(c):
    return [*server_options(c), "--logprobs-mode", c["logprobs_mode"]]


def generated_observation(response, row, c, eos_id):
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("One complete sampled response required; no retry")
    choice = choices[0]
    ids = choice.get("token_ids")
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs")
    if (not isinstance(ids, list) or not ids or len(ids) > c["max_new_tokens"]
            or any(type(i) is not int or i < 0 for i in ids)
            or not isinstance(logprobs, list) or len(logprobs) != len(ids)
            or any(type(x) not in (float, int) or not math.isfinite(x) or x > 1e-6 for x in logprobs)
            or not isinstance(choice.get("text"), str)):
        raise ValueError("Actual sampled IDs/logprobs must be complete and finite, including EOS")
    finish = choice.get("finish_reason")
    if eos_id in ids[:-1] or (finish == "stop" and ids[-1] != eos_id):
        raise ValueError("EOS accounting mismatch; never append/strip/repair EOS")
    if finish not in ("stop", "length") or (finish == "length" and len(ids) != c["max_new_tokens"]):
        raise ValueError("Aborted or inconsistent sampled response")
    prefix = row["input_ids"][:row["prompt_tokens"]]
    return {"user_id": row["user_id"], "input_ids": prefix + ids,
        "prompt_tokens": len(prefix), "response_token_ids": ids,
        "behavior_logprobs": logprobs, "finish_reason": finish,
        "eos_observed": ids[-1] == eos_id, "raw_text": choice["text"],
        "response_tokens_including_eos": len(ids)}


def verify_corrected_preparation(private, root, c, path, commit):
    p = private / "runs" / c["prepare_run_id"]
    report = json.loads((p / "report.json").read_text())
    if (report["status"] != "CORRECTED_ROLLOUT_CPU_PREPARATION_PASS_GENERATED_GPU_GATES_REMAIN"
            or report["source_commit"] != commit or report["config_sha256"] != file_sha256(path)
            or report["code_sha256"] != corrected_code_hashes(root)
            or report["gpu_requested"] or report["training_ready"]):
        raise ValueError("Exact-source corrected CPU receipt required")
    for n, sha in report["artifact_sha256"].items():
        if Path(n).name != n or file_sha256(p / n) != sha:
            raise ValueError("Corrected preparation bytes changed")
    rows = list(map(json.loads, (p / "tokens.jsonl").read_text().splitlines()))
    if len(rows) != 20 or len({r["user_id"] for r in rows}) != 20:
        raise ValueError("All20 frozen synthetic inputs required")
    return rows, report


def native_microbatch(ids, c, torch):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    nested = torch.nested.nested_tensor([ids], layout=torch.jagged)
    positions = torch.nested.nested_tensor([torch.arange(len(ids), device=ids.device)], layout=torch.jagged)
    data = TensorDict({"input_ids": nested, "position_ids": positions,
        "temperature": torch.tensor([c["temperature"]], device=ids.device)}, batch_size=[1])
    for k, v in {"use_remove_padding": False, "use_fused_kernels": False,
                 "calculate_entropy": False, "calculate_sum_pi_squared": False}.items():
        tu.assign_non_tensor(data, **{k: v})
    return data


def native_cpu_rehearsal(c):
    """20 native packing/unfused arithmetic proxies; native CE is CUDA-only."""
    import torch
    from types import SimpleNamespace
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.workers.config.actor import ActorConfig
    from verl.workers.config.rollout import RolloutConfig
    torch.set_num_threads(4)
    correction = RolloutCorrectionConfig()
    for k in ("rollout_is", "rollout_is_threshold", "rollout_is_batch_normalize", "rollout_rs", "bypass_mode", "loss_type"):
        if getattr(correction, k) != c[k]:
            raise ValueError("Native correction defaults changed")
    if RolloutConfig(name="vllm").logprobs_mode != c["logprobs_mode"]:
        raise ValueError("Native behavior-logprob mode changed")
    actor = ActorConfig(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=20,
                        ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False)
    proxy = SimpleNamespace(use_ulysses_sp=False, pad_to_length=False)
    for i in range(20):
        ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8 + i % 8])
        data = native_microbatch(ids, c, torch)
        inputs, args = FSDPEngineWithLMHead.prepare_model_inputs(proxy, data)
        logits = torch.arange(ids.numel() * 32, dtype=torch.float32).reshape(1, -1, 32) / 200
        # The pinned native output dispatcher chooses FA2 cross entropy when
        # the overlay is installed, even for CPU logits. Do not fake CUDA or
        # patch the dispatcher. Rehearse its documented arithmetic unfused on
        # CPU; the actual native output kernel remains mandatory in GPU30.
        expected = (logits / c["temperature"]).log_softmax(-1).gather(-1, torch.roll(ids, -1)[None, :, None]).reshape(-1)
        scaled = logits / args["temperature"].unsqueeze(-1).unsqueeze(-1)
        observed = scaled.log_softmax(-1).reshape(-1, 32).gather(
            -1, args["input_ids_rmpad_rolled"][:, None]).reshape(-1)
        if not torch.equal(inputs["input_ids"][0], ids):
            raise ValueError("Native input packing changed the exact token sequence")
        if not torch.allclose(observed, expected, atol=1e-5, rtol=1e-5):
            raise ValueError("Native next-token/temperature alignment failed")
        mask = torch.ones((1, 3))
        weights, _ = compute_rollout_correction_weights(torch.zeros_like(mask), mask,
            rollout_is=correction.rollout_is, rollout_is_threshold=correction.rollout_is_threshold)
        current = observed[-4:-1][None].detach().requires_grad_()
        loss, _ = compute_policy_loss_vanilla(current.detach(), current, mask, mask,
                                             config=actor, rollout_is_weights=weights)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(current.grad).all():
            raise ValueError("Native CPU correction/loss proxy failed")
    if torch.cuda.is_initialized():
        raise ValueError("CPU rehearsal initialized CUDA")
    return {"samples": 20, "native_input_packing_and_unfused_temperature_loss_proxy_pass": True,
            "native_gpu_output_kernel_executed": False,
            "scope": "CPU_PACKING_UNFUSED_ARITHMETIC_NOT_NATIVE_CUDA_CE_OR_FSDP_MODEL", "gpu_requested": False}
