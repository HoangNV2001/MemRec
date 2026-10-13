#!/usr/bin/env python3
"""Native correction/loss review AFTER full model exit/release/keeper handback."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.corrected_rollout import load_corrected_config, verify_corrected_preparation
from src.cmirank.ppo_parity import compare_logprobs
from src.cmirank.provenance import artifact_json_dumps, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    require_allocation()
    c, path = load_corrected_config(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private / "runs" / c["review_run_id"] or run.exists()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Fresh exact-source CPU review required")
    rows, prepared = verify_corrected_preparation(private, ROOT, c, path, commit)
    parent = private / "runs" / c["run_id"]
    report = json.loads((parent / "report.json").read_text())
    cleanup = json.loads((parent / "cleanup.json").read_text())
    handback = json.loads((parent / "keeper-handback.json").read_text())
    if (report["status"] != "GENERATED_ROLLOUT_NATIVE_ENGINE_PASS_CPU_CORRECTION_REVIEW_REQUIRED"
            or report["source_commit"] != commit or report["config_sha256"] != file_sha256(path)
            or report["physical_requests"] != 20 or cleanup["process_exit_code"]
            or not cleanup["child_exited"] or not cleanup["gpu_released"] or not cleanup["allocation_still_running"]
            or handback["status"] != "KEEPER_GPU1_HANDBACK_VERIFIED"
            or handback["reserved_job_cancelled"] or handback["gpu0_signalled"]):
        raise ValueError("Successful bounded GPU result plus actual release/keeper handback required")
    observations = list(map(json.loads, (parent / "generated.jsonl").read_text().splitlines()))
    native = list(map(json.loads, (parent / "native-logprobs.jsonl").read_text().splitlines()))
    expected = [r["user_id"] for r in rows]
    if (len(observations) != 20 or len(native) != 20
            or [x["user_id"] for x in observations] != expected
            or [x["user_id"] for x in native] != expected):
        raise ValueError("Never omit/reorder malformed or truncated cases")
    import torch
    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights, SAFETY_BOUND
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.workers.config.actor import ActorConfig
    correction = RolloutCorrectionConfig()
    for k in ("rollout_is", "rollout_is_threshold", "rollout_is_batch_normalize", "rollout_rs", "bypass_mode", "loss_type"):
        if getattr(correction, k) != c[k]:
            raise ValueError("Native correction defaults changed")
    actor = ActorConfig(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=20,
        ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False)
    weights, probes = [], []
    for row, obs, score in zip(rows, observations, native):
        n = len(obs["response_token_ids"])
        if (obs["input_ids"] != row["input_ids"][:row["prompt_tokens"]] + obs["response_token_ids"]
                or any(len(score[k]) != n for k in ("old_logprobs", "recomputed_logprobs", "reference_logprobs"))
                or len(obs["behavior_logprobs"]) != n
                or obs["eos_observed"] != (obs["response_token_ids"][-1] == prepared["eos_token_id"])):
            raise ValueError("Exact generated prefix/response/EOS alignment required")
        repeat = compare_logprobs(score["old_logprobs"], score["recomputed_logprobs"], c)
        if not repeat["passed"]:
            raise ValueError("Same-engine pre-update repeat failed")
        old = torch.tensor([score["old_logprobs"]], dtype=torch.float64)
        behavior = torch.tensor([obs["behavior_logprobs"]], dtype=torch.float64)
        mask = torch.ones_like(old)
        w, _ = compute_rollout_correction_weights(old-behavior, mask,
            rollout_is=correction.rollout_is, rollout_is_threshold=correction.rollout_is_threshold,
            rollout_is_batch_normalize=correction.rollout_is_batch_normalize)
        raw_logweight = float((old-behavior).sum())
        expected_w = min(correction.rollout_is_threshold,
            math.exp(min(float(SAFETY_BOUND), max(-float(SAFETY_BOUND), raw_logweight))))
        # Native safety clamp is audited through the helper, not used to conceal drift.
        if not torch.isfinite(w).all() or (w <= 0).any():
            raise ValueError("Finite positive native IS weights required")
        if not torch.allclose(w, torch.full_like(w, expected_w), atol=1e-12, rtol=1e-12):
            raise ValueError("Independent formula differs from the pinned native correction")
        current = old.clone().requires_grad_()
        loss, _ = compute_policy_loss_vanilla(old, current, mask, mask, config=actor, rollout_is_weights=w)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(current.grad).all():
            raise ValueError("Native corrected loss wiring failed")
        weights.append(float(w[0, 0]))
        probes.append({"user_id": obs["user_id"], "response_tokens_including_eos": n,
            "native_sequence_is_weight": weights[-1], "raw_action_logratio": raw_logweight,
            "above_native_cap": raw_logweight > math.log(2.0), "native_loss_finite": True,
            "native_repeat": repeat, "format_valid": obs["format_valid"],
            "finish_reason": obs["finish_reason"], "formula_diagnostic_weight": expected_w})
    if torch.cuda.is_initialized():
        raise ValueError("CPU review initialized CUDA")
    ess = sum(weights)**2 / (len(weights)*sum(x*x for x in weights))
    run.mkdir()
    (run / "probes.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in probes))
    result = {"status": "GENERATED_ROLLOUT_NATIVE_CORRECTION_REVIEW_PASS_OPTIMIZER_AND_MULTITURN_GATES_REMAIN",
        "source_commit": commit, "config_sha256": file_sha256(path), "samples": 20,
        "weight_min": min(weights), "weight_max": max(weights), "ess_fraction": ess,
        "above_native_cap_samples": sum(p["above_native_cap"] for p in probes),
        "format_valid_samples": sum(p["format_valid"] for p in probes),
        "parent_artifact_sha256": {n: file_sha256(parent/n) for n in
            ("report.json", "generated.jsonl", "native-logprobs.jsonl", "cleanup.json", "keeper-handback.json")},
        "optimizer_updates": 0, "training_ready": False, "gpu_requested": False,
        "books_outcomes_accessed": False, "ranking_metrics_computed": False,
        "old_strict_cross_engine_gate_status": "FAIL_UNCHANGED",
        "reference_scope": "shared_initial_weights_no_separate_reference_worker_proof"}
    (run / "report.json").write_text(artifact_json_dumps(result))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
