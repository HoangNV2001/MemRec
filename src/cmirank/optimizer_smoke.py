"""Bounded native actor optimizer qualification, not a PPO research trainer.

No torch/CUDA or full weights at import time. Native packing/loss/AdamW only;
the one discarded update consumes all20 immutable synthetic sampled responses.
"""

import json
import math
from pathlib import Path

from .corrected_rollout import load_corrected_config
from .provenance import file_sha256

TEXT_PARAMETERS = 4205751296
VISION_PARAMETERS = 333514240


def load_optimizer_config(root, version=2):
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unknown native actor optimizer contract")
    path = root / "configs/cmirank/ppo_actor_optimizer_v1.json"
    c = json.loads(path.read_text())
    fixed = {"schema_version": 1, "samples": 20, "optimizer_updates": 1,
        "micro_batch_size": 1, "actor_lr": 1e-6, "model_dtype": "fp32",
        "forward_dtype": "bfloat16", "optimizer": "torch.optim.AdamW", "foreach": False,
        "freeze_vision": True, "gradient_checkpointing": True, "param_offload": False,
        "optimizer_offload": False, "gpu_index": 1, "tensor_parallel_size": 1,
        "timeout_minutes": 25, "worker_timeout_seconds": 1200,
        "workspace_reserve_bytes": 8 * 2**30, "host_load_reserve_bytes": 20 * 2**30,
        "advantages": "alternating_plus_minus_one_by_frozen_row_order", "physical_requests": 0,
        "discard_update": True, "training_ready": False, "books_outcomes_accessed": False,
        "ranking_metrics_computed": False, "co_residency_proven": False,
        "full_checkpoint_roundtrip_proven": False}
    provenance = {"prepare_run_id", "run_id", "parent_source_commit", "parent_prepare_run_id",
        "parent_prepare_report_sha256", "parent_run_id", "parent_report_sha256",
        "parent_review_run_id", "parent_review_report_sha256", "generated_sha256", "native_logprobs_sha256"}
    if (set(c) != set(fixed) | provenance
            or any(type(c.get(k)) is not type(v) or c[k] != v for k, v in fixed.items())):
        raise ValueError("Only the fixed20 one-update native actor compatibility profile is allowed")
    for k in ("prepare_run_id", "run_id", "parent_prepare_run_id", "parent_run_id", "parent_review_run_id"):
        if not isinstance(c[k], str) or Path(c[k]).name != c[k] or not c[k].startswith("cmirank-"):
            raise ValueError("Run IDs must be isolated basenames")
    for k in ("parent_source_commit", "parent_prepare_report_sha256", "parent_report_sha256",
              "parent_review_report_sha256", "generated_sha256", "native_logprobs_sha256"):
        if len(c[k]) != (40 if k == "parent_source_commit" else 64) or any(x not in "0123456789abcdef" for x in c[k]):
            raise ValueError("Immutable parent provenance required")
    if version == 2:
        path = root / "configs/cmirank/ppo_actor_optimizer_v2.json"
        retry = json.loads(path.read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "reason"}
                or retry["schema_version"] != 2
                or retry["inherits"] != "configs/cmirank/ppo_actor_optimizer_v1.json"
                or retry["prepare_run_id"] != "cmirank-actor-optimizer-data-v2-20261014-hnv"
                or retry["run_id"] != "cmirank-actor-optimizer-smoke-v2-20261014-hnv"):
            raise ValueError("Release-integration retry cannot change the scientific/resource profile")
        c.update(retry)
    return c, path


def optimizer_code_hashes(root):
    names = ("configs/cmirank/ppo_actor_optimizer_v1.json", "configs/cmirank/ppo_actor_optimizer_v2.json",
        "src/cmirank/optimizer_smoke.py",
        "scripts/cmirank/32_prepare_actor_optimizer_cpu.py", "scripts/cmirank/33_smoke_actor_optimizer_gpu.py",
        "scripts/cmirank/34_native_actor_optimizer_gpu.py", "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py",
        "src/cmirank/ppo_parity.py", "src/cmirank/reserved_handoff.py", "src/cmirank/gpu_resources.py",
        "scripts/cmirank/run_real_policy_smoke.sh", "scripts/cmirank/run_real_policy_with_keeper.sh")
    return {n: file_sha256(root / n) for n in names}


def memory_profile():
    # Native FP32 master parameters, FP32 gradients and two FP32 Adam moments.
    # Frozen visual branch has parameters only; activation/workspace is extra.
    return {"text_parameters": TEXT_PARAMETERS, "frozen_vision_parameters": VISION_PARAMETERS,
        "parameter_bytes": 4 * (TEXT_PARAMETERS + VISION_PARAMETERS),
        "gradient_bytes": 4 * TEXT_PARAMETERS, "adam_moment_bytes": 8 * TEXT_PARAMETERS,
        "persistent_train_bytes": 16 * TEXT_PARAMETERS + 4 * VISION_PARAMETERS,
        "roles_resident": ["actor"], "actor_critic_rollout_fit_proven": False}


def allocation_memory(job):
    """Read only this allocation's cgroup, never global or foreign memory."""
    line = next(x for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    p = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")
    while p.name != f"job_{job}":
        if p == p.parent or p == Path("/sys/fs/cgroup"):
            raise ValueError("Cannot identify the authorized allocation's memory cgroup")
        p = p.parent
    maximum = (p / "memory.max").read_text().strip()
    if maximum == "max":
        raise ValueError("An explicit allocation memory cap is required")
    return {"maximum_bytes": int(maximum), "current_bytes": int((p / "memory.current").read_text()),
        "scope": f"own_allocation_job_{job}"}


def validate_observations(observations, native, c):
    if (len(observations) != 20 or len(native) != 20
            or len({x["user_id"] for x in observations}) != 20
            or [x["user_id"] for x in observations] != [x["user_id"] for x in native]):
        raise ValueError("All20 frozen observations must remain in their original order")
    for obs, score in zip(observations, native):
        n, p = len(obs["response_token_ids"]), obs["prompt_tokens"]
        if (obs["user_id"] >= 0 or type(p) is not int or p < 1 or n < 1
                or len(obs["input_ids"]) != p + n or obs["input_ids"][p:] != obs["response_token_ids"]
                or obs["response_token_ids"][-1] != 248046 or not obs["eos_observed"]
                or obs["finish_reason"] != "stop" or not obs["format_valid"]
                or any(type(i) is not int or i < 0 for i in obs["input_ids"])):
            raise ValueError("Synthetic exact prefix/response/EOS scope changed; no filtering or repair")
        for values in (obs["behavior_logprobs"], score["old_logprobs"], score["recomputed_logprobs"]):
            if len(values) != n or any(not math.isfinite(x) or x > 1e-6 for x in values):
                raise ValueError("Every observed response token including EOS needs finite likelihoods")


def verified_parents(private, c):
    reports = {}
    for run_key, hash_key in (("parent_prepare_run_id", "parent_prepare_report_sha256"),
                             ("parent_run_id", "parent_report_sha256"),
                             ("parent_review_run_id", "parent_review_report_sha256")):
        path = private / "runs" / c[run_key] / "report.json"
        if file_sha256(path) != c[hash_key]:
            raise ValueError("Reviewed immutable parent report changed")
        reports[run_key] = json.loads(path.read_text())
        if reports[run_key]["source_commit"] != c["parent_source_commit"]:
            raise ValueError("Never relabel old artifacts as produced by the new optimizer source")
    review = reports["parent_review_run_id"]
    parent = private / "runs" / c["parent_run_id"]
    if review["status"] != "GENERATED_ROLLOUT_NATIVE_CORRECTION_REVIEW_PASS_OPTIMIZER_AND_MULTITURN_GATES_REMAIN":
        raise ValueError("Generated behavior/correction review must pass first")
    for n, sha in review["parent_artifact_sha256"].items():
        if Path(n).name != n or file_sha256(parent / n) != sha:
            raise ValueError("Reviewed parent output or cleanup changed")
    if (file_sha256(parent / "generated.jsonl") != c["generated_sha256"]
            or file_sha256(parent / "native-logprobs.jsonl") != c["native_logprobs_sha256"]):
        raise ValueError("Frozen generated responses/likelihoods changed")
    obs = list(map(json.loads, (parent / "generated.jsonl").read_text().splitlines()))
    native = list(map(json.loads, (parent / "native-logprobs.jsonl").read_text().splitlines()))
    validate_observations(obs, native, c)
    return obs, native, reports["parent_prepare_run_id"]


def native_actor_batch(observations, old_logprobs, c, torch, device="cpu"):
    """Adapt frozen samples to native VeRL TensorDict; no custom policy loss."""
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights
    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    correction = RolloutCorrectionConfig()
    if (correction.rollout_is != "sequence" or correction.rollout_is_threshold != 2.0
            or correction.rollout_rs is not None or correction.rollout_is_batch_normalize
            or correction.bypass_mode):
        raise ValueError("Approved native correction defaults changed")
    fields = {k: [] for k in ("input_ids", "position_ids", "prompts", "responses", "loss_mask",
        "response_mask", "old_log_probs", "advantages", "rollout_is_weights")}
    weights = []
    for i, (obs, old) in enumerate(zip(observations, old_logprobs)):
        ids = torch.tensor(obs["input_ids"], device=device)
        p, n = obs["prompt_tokens"], len(obs["response_token_ids"])
        if len(old) != n:
            raise ValueError("Native old-policy response/EOS shape differs")
        old = torch.tensor(old, dtype=torch.float32, device=device)
        behavior = torch.tensor(obs["behavior_logprobs"], dtype=torch.float32, device=device)
        weight, _ = compute_rollout_correction_weights((old-behavior)[None], torch.ones_like(old)[None],
            rollout_is=correction.rollout_is, rollout_is_threshold=correction.rollout_is_threshold,
            rollout_is_batch_normalize=False)
        if not torch.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("Nonfinite/zero native correction weight")
        weights.append(float(weight[0, 0]))
        values = {"input_ids": ids, "position_ids": torch.arange(p+n, device=device),
            "prompts": ids[:p], "responses": ids[p:],
            "loss_mask": torch.cat((torch.zeros(p, device=device), torch.ones(n, device=device))),
            "response_mask": torch.ones(n, device=device), "old_log_probs": old,
            "advantages": torch.full((n,), 1.0 if i % 2 == 0 else -1.0, device=device),
            "rollout_is_weights": weight[0]}
        for key, value in values.items():
            fields[key].append(value)
    if len(observations) != c["samples"] or len(old_logprobs) != c["samples"]:
        raise ValueError("One native update must include exactly20 samples")
    data = TensorDict({k: torch.nested.as_nested_tensor(v, layout=torch.jagged) for k, v in fields.items()},
        batch_size=[c["samples"]])
    data["temperature"] = torch.full((c["samples"],), 0.9, device=device)
    tu.assign_non_tensor(data, use_remove_padding=False, use_fused_kernels=False,
        calculate_entropy=False, calculate_sum_pi_squared=False, use_dynamic_bsz=False,
        micro_batch_size_per_gpu=c["micro_batch_size"], global_batch_size=c["samples"],
        max_response_len=max(len(x["response_token_ids"]) for x in observations), return_model_output=False)
    return data, weights


def verify_optimizer_preparation(private, root, c, path, commit):
    p = private / "runs" / c["prepare_run_id"]
    report = json.loads((p / "report.json").read_text())
    if (report["status"] != "NATIVE_ACTOR_OPTIMIZER_CPU_PREPARATION_PASS_GPU_GATE_REMAINS"
            or report["source_commit"] != commit or report["config_sha256"] != file_sha256(path)
            or report["code_sha256"] != optimizer_code_hashes(root)
            or report["samples"] != 20 or report["gpu_requested"] or report["optimizer_updates"]
            or report["full_weights_loaded"] or report["training_ready"]):
        raise ValueError("Exact-source native optimizer CPU receipt required")
    for n, sha in report["artifact_sha256"].items():
        if Path(n).name != n or file_sha256(p / n) != sha:
            raise ValueError("Prepared CPU optimizer probes changed")
    verified_parents(private, c)
    return report
