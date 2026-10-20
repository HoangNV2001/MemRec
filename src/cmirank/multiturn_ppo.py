"""Episode-continuous MPSS/GAE adapter for the unmodified native VeRL losses.

Twenty synthetic episodes qualify integration, not Books research learning.
Prompts are observations: only response tokens enter GAE, across all nine turns.
No torch/CUDA/model loading at import time.
"""

import hashlib
import json
import math
from pathlib import Path
import subprocess

from .corrected_rollout import generated_observation, generated_request
from .format_sft import synthetic_request
from .labels import make_labels
from .lora_rollout import load_lora_rollout_config
from .parser import parse_action
from .ppo_runtime import response_reward_layout
from .prompts import render_step_prompt
from .provenance import file_sha256
from .request import RankRequest
from .rewards import mpss_rewards, ndcg_at_k


def load_multiturn_config(root):
    c, _ = load_lora_rollout_config(root)
    path = root / "configs/cmirank/ppo_lora_multiturn_v1.json"
    r = json.loads(path.read_text())
    fixed = {"schema_version": 1,
        "prepare_run_id": "cmirank-lora-multiturn-data-v1-20261020-hnv",
        "run_id": "cmirank-lora-multiturn-smoke-v1-20261020-hnv",
        "data_seed": "cmirank-t0-multiturn-v1-20261020", "episodes": 20,
        "turns_per_episode": 9, "physical_request_cap": 180, "ppo_epochs": 1,
        "global_minibatch": 36, "micro_batch_size": 1, "gamma": 1.0, "gae_lambda": 1.0,
        "kl_coef": 1e-4, "kl_loss_type": "low_var_kl", "updates_per_role": 5,
        "optimizer_updates": 10, "discard_update": True, "worker_timeout_seconds": 1200,
        "port": 18127, "parent_source_commit": "844284d547a5075f23ffd3d754b3afc5e68e9a7f",
        "parent_prepare_report_sha256": "64cb2b9b1a702d4607563937e3a883f438a59f9d65cc99cfcf350a317e4776bc",
        "parent_report_sha256": "1f732b51f436613db94a82145b2ad785043cfb5d9e5382377ca41d7e4b77f097",
        "parent_review_report_sha256": "500fe246b145ce9a5e7f09204d4e2353067e3980a98ae94276e4e5dfc75c8abf"}
    if r != fixed or any(type(r[k]) is not type(v) for k, v in fixed.items()):
        raise ValueError("Only fixed20 synthetic N-1 MPSS/GAE qualification; no tuning")
    c["rollout_parent_prepare_run_id"] = c["prepare_run_id"]
    c["rollout_parent_run_id"] = c["run_id"]
    c["rollout_parent_review_run_id"] = c["review_run_id"]
    c.update(r)
    c["parameterization"] = "lora_multiturn"
    c["worker_status"] = "NATIVE_LORA_MULTITURN_MPSS_GAE_PPO_SMOKE_PASS_REVIEW_REQUIRED"
    return c, path


def multiturn_code_hashes(root):
    names = ("configs/cmirank/ppo_lora_multiturn_v1.json", "src/cmirank/multiturn_ppo.py",
        "scripts/cmirank/41_prepare_multiturn_ppo_cpu.py", "scripts/cmirank/42_native_multiturn_ppo_gpu.py",
        "scripts/cmirank/29_smoke_corrected_rollout_gpu.py", "scripts/cmirank/run_real_policy_smoke.sh",
        "scripts/cmirank/run_real_policy_with_keeper.sh", "src/cmirank/rewards.py",
        "src/cmirank/ppo_runtime.py", "src/cmirank/prompts.py", "src/cmirank/parser.py",
        "src/cmirank/format_sft.py", "src/cmirank/lora_joint.py", "src/cmirank/lora_checkpoint.py",
        "src/cmirank/lora_rollout.py", "src/cmirank/reserved_handoff.py", "src/cmirank/gpu_resources.py")
    return {n: file_sha256(root/n) for n in names}


def reviewed_rollout_parent(private, root, c):
    folders = [private/"runs"/c[k] for k in ("rollout_parent_prepare_run_id",
        "rollout_parent_run_id", "rollout_parent_review_run_id")]
    reports = []
    for p, key in zip(folders, ("parent_prepare_report_sha256", "parent_report_sha256", "parent_review_report_sha256")):
        if file_sha256(p/"report.json") != c[key]:
            raise ValueError("Reviewed LoRA rollout dependency changed")
        row = json.loads((p/"report.json").read_text())
        if row["source_commit"] != c["parent_source_commit"] or row["training_ready"]:
            raise ValueError("Parent scope/source changed")
        reports.append(row)
    prep, report, review = reports
    if (review["status"] != "LORA_ROLLOUT_REFERENCE_NATIVE_CORRECTION_REVIEW_PASS_MULTITURN_REMAINS"
            or report["format_valid_samples"] != 20 or report["physical_requests"] != 60
            or report["optimizer_updates"] or not review["cold_process_restart_proven"]):
        raise ValueError("Completed20 cold-restore/reference/correction parent required")
    for name, sha in prep["code_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe producing source path")
        if hashlib.sha256(subprocess.check_output(["git", "show", c["parent_source_commit"]+":"+name], cwd=root)).hexdigest() != sha:
            raise ValueError("Parent source receipt changed")
    for directory, entries in ((folders[0], prep["artifact_sha256"]), (folders[1], review["parent_artifact_sha256"])):
        for name, sha in entries.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or file_sha256(directory/name) != sha:
                raise ValueError("Parent data/adapter/cleanup bytes changed")
    return prep


def prompt_row(request, active, tokenizer, c, turn):
    text = render_step_prompt(request, active)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    if not tokens or len(tokens)+c["max_new_tokens"] > c["max_model_len"]:
        raise ValueError("Complete prompt exceeds original cap; never truncate")
    return {"user_id": request.user_id, "turn": turn, "active_labels": list(active),
        "rank_request_sha256": request.sha256(), "input_ids": tokens, "prompt_tokens": len(tokens)}


def sample_trajectories(rows, tokenizer, c, call, emit):
    """No labels/reward access. Retain every malformed episode; no retry/repair."""
    observations, episodes = [], []
    labels = list(make_labels(range(10)))
    for row in rows:
        request = RankRequest.from_dict(row)
        if request.user_id >= 0:
            raise ValueError("This T0 gate accepts synthetic negative identities only")
        active, exclusions = labels.copy(), []
        for turn in range(9):
            prompt = prompt_row(request, active, tokenizer, c, turn)
            payload = {**generated_request(prompt, c), "model": c["served_adapter_name"]}
            raw = call(payload, request.user_id, turn)
            obs = generated_observation(raw, prompt, c, tokenizer.eos_token_id)
            parsed = parse_action(obs["raw_text"], active)
            valid = parsed.valid and obs["finish_reason"] == "stop" and obs["eos_observed"]
            obs.update(turn=turn, active_labels=active.copy(), rank_request_sha256=request.sha256(),
                format_valid=valid, format_failure=parsed.failure_reason)
            emit("generated", obs)
            observations.append(obs)
            if not valid:
                break
            exclusions.append(parsed.label)
            active.remove(parsed.label)
        episode = {"user_id": request.user_id, "rank_request_sha256": request.sha256(),
            "exclusions": exclusions, "survivor": active[0] if len(active) == 1 else None,
            "valid": len(exclusions) == 9 and len(active) == 1}
        emit("episodes", episode)
        episodes.append(episode)
    return observations, episodes


def validate_episodes(observations, episodes, requests):
    if len(requests) != 20 or len(episodes) != 20 or len(observations) != 180:
        raise ValueError("All20 complete9-turn episodes required; no filtering")
    if len({r["user_id"] for r in requests}) != 20:
        raise ValueError("Unique synthetic episodes required")
    labels = list(make_labels(range(10)))
    for index, (episode, raw) in enumerate(zip(episodes, requests)):
        request = RankRequest.from_dict(raw)
        if (request.user_id >= 0 or episode["user_id"] != request.user_id
                or episode["rank_request_sha256"] != request.sha256() or not episode["valid"]):
            raise ValueError("Synthetic identity/provenance/episode invalid")
        active, exclusions = labels.copy(), []
        for turn, obs in enumerate(observations[index*9:index*9+9]):
            if (obs["user_id"] != request.user_id or obs["turn"] != turn or obs["active_labels"] != active
                    or obs["rank_request_sha256"] != request.sha256() or not obs["format_valid"]
                    or obs["finish_reason"] != "stop" or not obs["eos_observed"]
                    or obs["response_token_ids"][-1] != 248046
                    or obs["input_ids"][obs["prompt_tokens"]:] != obs["response_token_ids"]):
                raise ValueError("Turn continuity/EOS/response bytes changed")
            parsed = parse_action(obs["raw_text"], active)
            if not parsed.valid:
                raise ValueError("Malformed action retained, never repaired")
            exclusions.append(parsed.label)
            active.remove(parsed.label)
        if episode["exclusions"] != exclusions or active != [episode["survivor"]]:
            raise ValueError("Episode replay differs")


def episode_gae(observations, episodes, old_values, c, torch, device="cpu"):
    """Native GAE over20 response-only episode rows; then split back into180 turns."""
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    if len(observations) != 180 or len(episodes) != 20 or len(old_values) != 180:
        raise ValueError("Need all20 nine-turn trajectories")
    labels = list(make_labels(range(10)))
    packed, lengths, audits = [], [], []
    for i, episode in enumerate(episodes):
        group = observations[i*9:i*9+9]
        ns = [len(r["response_token_ids"]) for r in group]
        # Predeclared exchangeable synthetic target, fixed by episode index;
        # never inferred from actions, visible context, or a Books outcome.
        target = labels[i % 10]
        rewards, rank = mpss_rewards(labels, episode["exclusions"], episode["survivor"], target)
        token_rewards, endpoints = response_reward_layout(ns, list(rewards))
        values = [x for n, vs in zip(ns, old_values[i*9:i*9+9]) for x in vs if len(vs) == n]
        if len(values) != sum(ns) or not all(math.isfinite(x) for x in values):
            raise ValueError("Every action token needs a finite native critic value")
        packed.append((token_rewards, values)); lengths.append(ns)
        audits.append({"user_id": episode["user_id"], "synthetic_target_label": target,
            "target_rank": rank, "mpss_rewards": list(rewards), "episode_return": sum(rewards),
            "expected_ndcg5": ndcg_at_k(rank), "response_final_indices": endpoints})
    width = max(len(reward) for reward, _ in packed)
    reward = torch.zeros((20, width), dtype=torch.float32, device=device)
    values = torch.zeros_like(reward); mask = torch.zeros_like(reward)
    expected_returns = torch.zeros_like(reward)
    for i, (rs, vs) in enumerate(packed):
        n = len(rs)
        reward[i, :n] = torch.tensor(rs, device=device); values[i, :n] = torch.tensor(vs, device=device)
        mask[i, :n] = 1
        suffix = 0.0
        for t in reversed(range(n)):
            suffix += rs[t]; expected_returns[i, t] = suffix
    advantages, returns = compute_gae_advantage_return(reward, values, mask, gamma=c["gamma"], lam=c["gae_lambda"])
    # gamma=lambda=1 => bootstrap terms telescope across turns; catches9
    # independent terminal episodes, prompt masking, padding/EOS mistakes.
    if not torch.allclose(returns*mask, expected_returns*mask, atol=2e-5, rtol=2e-5):
        raise ValueError("Native GAE return is not complete-episode suffix reward")
    if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
        raise ValueError("Nonfinite native episode GAE")
    turn_adv, turn_ret = [], []
    for i, ns in enumerate(lengths):
        start = 0
        for n in ns:
            turn_adv.append(advantages[i, start:start+n].detach())
            turn_ret.append(returns[i, start:start+n].detach())
            start += n
    return turn_adv, turn_ret, audits


def native_training_batch(observations, scores, advantages, returns, c, role, torch, device="cpu"):
    """The native TensorDict ABI; real GAE/MPSS, not alternating smoke advantages."""
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights
    if role not in ("actor", "critic") or not len(observations) == len(scores) == len(advantages) == len(returns):
        raise ValueError("Matching complete action batch and native role required")
    fields = {k: [] for k in ("input_ids", "position_ids", "prompts", "responses", "loss_mask", "response_mask")}
    fields.update({k: [] for k in (("old_log_probs", "ref_log_prob", "advantages", "rollout_is_weights")
        if role == "actor" else ("values", "returns"))})
    for obs, score, adv, ret in zip(observations, scores, advantages, returns):
        ids = torch.tensor(obs["input_ids"], device=device)
        p, n = obs["prompt_tokens"], len(obs["response_token_ids"])
        row = {"input_ids": ids, "position_ids": torch.arange(p+n, device=device), "prompts": ids[:p],
            "responses": ids[p:], "loss_mask": torch.cat((torch.zeros(p, device=device), torch.ones(n, device=device))),
            "response_mask": torch.ones(n, device=device)}
        if role == "actor":
            old = torch.tensor(score["old_logprobs"], dtype=torch.float32, device=device)
            behavior = torch.tensor(obs["behavior_logprobs"], dtype=torch.float32, device=device)
            weight, _ = compute_rollout_correction_weights((old-behavior)[None], torch.ones_like(old)[None],
                rollout_is="sequence", rollout_is_threshold=2.0, rollout_is_batch_normalize=False)
            row.update(old_log_probs=old, ref_log_prob=torch.tensor(score["reference_logprobs"], device=device),
                advantages=adv.to(device), rollout_is_weights=weight[0])
        else:
            row.update(values=torch.tensor(score["critic_values"], device=device), returns=ret.to(device))
        if any(v.ndim != 1 or not torch.isfinite(v).all() for v in row.values()):
            raise ValueError("Finite1D native PPO fields required")
        if role == "actor" and (row["rollout_is_weights"] <= 0).any():
            raise ValueError("Native sequence correction weights must stay positive")
        for key in set(row)-{"input_ids", "position_ids", "prompts", "loss_mask"}:
            if row[key].numel() != n:
                raise ValueError("Response/EOS lengths disagree")
        for key, value in row.items():
            fields[key].append(value)
    size = len(observations)
    data = TensorDict({k: torch.nested.as_nested_tensor(v, layout=torch.jagged) for k, v in fields.items()}, batch_size=[size])
    data["temperature"] = torch.full((size,), c["temperature"], device=device)
    tu.assign_non_tensor(data, use_remove_padding=False, use_fused_kernels=False,
        calculate_entropy=False, calculate_sum_pi_squared=False, use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1, global_batch_size=size,
        max_response_len=max(len(x["response_token_ids"]) for x in observations), return_model_output=False)
    return data


def verify_multiturn_preparation(private, root, c, path, commit):
    directory = private/"runs"/c["prepare_run_id"]
    p = json.loads((directory/"report.json").read_text())
    if (p["status"] != "MULTITURN_PPO_CPU20_PASS_GPU_GATE_REMAINS" or p["source_commit"] != commit
            or p["config_sha256"] != file_sha256(path) or p["code_sha256"] != multiturn_code_hashes(root)
            or p["episodes"] != 20 or p["gpu_requested"] or p["optimizer_updates"] or p["training_ready"]):
        raise ValueError("Exact-source CPU20 multi-turn rehearsal required")
    for name, sha in p["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(directory/name) != sha:
            raise ValueError("Fresh CPU multi-turn artifacts changed")
    rows = list(map(json.loads, (directory/"requests.jsonl").read_text().splitlines()))
    if rows != [synthetic_request(c["data_seed"], "holdout", i).to_dict() for i in range(20)]:
        raise ValueError("Immutable outcome-free request recipe changed")
    # Old parent artifacts are checked against their own producing source,
    # not relabelled as generated by this gate.
    if file_sha256(private/"runs"/c["rollout_parent_prepare_run_id"]/"report.json") != c["parent_prepare_report_sha256"]:
        raise ValueError("Parent preparation changed")
    from .lora_rollout import checkpoint_directory
    for name, expected in p["native_checkpoint_stat"].items():
        stat = (checkpoint_directory(private, c)/name).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino} != expected:
            raise ValueError("Native role checkpoint changed after CPU hashing")
    parent = json.loads((private/"runs"/c["rollout_parent_prepare_run_id"]/"report.json").read_text())
    for name, sha in parent["artifact_sha256"].items():
        if name.startswith("actor-adapter/") and file_sha256(private/"runs"/c["rollout_parent_prepare_run_id"]/name) != sha:
            raise ValueError("Static sampler adapter changed after CPU qualification")
    return rows, p
