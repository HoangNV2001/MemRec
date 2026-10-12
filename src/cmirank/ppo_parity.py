"""Outcome-free protocol/mask guards for an initial vLLM/HF numerical probe.

No CUDA, model loading or HTTP at import time. Does not implement PPO training.
"""

import json
import math
from pathlib import Path

from .provenance import file_sha256


def load_parity_config(root):
    path = root / "configs/cmirank/ppo_parity_v1.json"
    config = json.loads(path.read_text())
    if (config["samples"] != 20 or config["physical_request_cap"] != 20
            or config["generated_tokens_per_request"] != 1 or config["tensor_parallel_size"] != 1
            or config["gpu_index"] != 0 or config["resource_policy"] != "gpu0_only_gpu1_all_workloads_read_only"
            or config["optimizer_updates"] or config["books_outcomes_accessed"]
            or config["ranking_metrics_computed"] or config["training_ready"] or config["ppo_worker_proven"]
            or config["temperature"] != 0 or config["prompt_logprob_temperature"] != 1):
        raise ValueError("Not the twenty-synthetic-request no-update parity gate")
    return config, path


def parity_code_hashes(root):
    return {name: file_sha256(root / name) for name in (
        "src/cmirank/ppo_parity.py", "scripts/cmirank/22_prepare_ppo_parity_cpu.py",
        "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py", "scripts/cmirank/24_hf_parity_worker_gpu.py",
        "src/cmirank/ppo_compat.py", "src/cmirank/reserved_handoff.py", "src/cmirank/gpu_resources.py",
        "scripts/cmirank/run_real_policy_smoke.sh", "scripts/cmirank/run_real_policy_with_keeper.sh")}


def server_options(config):
    return ["--served-model-name", config["served_model_name"], "--host", "127.0.0.1", "--port", str(config["port"]),
        "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--generation-config", "vllm",
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--max-model-len", str(config["max_model_len"]), "--max-num-seqs", "1",
        "--max-num-batched-tokens", str(config["max_model_len"]), "--seed", str(config["seed"]),
        "--no-enable-prefix-caching", "--enforce-eager"]


def completion_request(row, config):
    if (row["user_id"] >= 0 or row["snapshot_id"] != "synthetic-format-holdout-v1"
            or not 1 <= row["prompt_tokens"] < len(row["input_ids"])
            or len(row["input_ids"]) + config["generated_tokens_per_request"] > config["max_model_len"]):
        raise ValueError("Parity requires the frozen synthetic prompt and complete response/EOS")
    return {"model": config["served_model_name"], "prompt": row["input_ids"],
            "max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "seed": config["seed"],
            "stream": False, "add_special_tokens": False, "prompt_logprobs": 0,
            "return_token_ids": True}


def response_prompt_logprobs(response, row):
    if len(response.get("choices", [])) != 1:
        raise ValueError("Parity needs exactly one physical completion response")
    choice = response["choices"][0]
    ids = choice.get("prompt_token_ids")
    probabilities = choice.get("prompt_logprobs")
    usage = response.get("usage", {})
    if (ids != row["input_ids"] or not isinstance(probabilities, list) or len(probabilities) != len(ids)
            or choice.get("finish_reason") not in ("stop", "length")
            or usage.get("prompt_tokens") != len(ids) or usage.get("completion_tokens") != 1):
        raise ValueError("vLLM changed tokens, inserted/truncated a prompt, or violated the one-token budget")
    result = []
    for index in range(row["prompt_tokens"], len(ids)):
        entry = probabilities[index]
        value = entry.get(str(ids[index]), entry.get(ids[index])) if isinstance(entry, dict) else None
        logprob = value.get("logprob") if isinstance(value, dict) else None
        if not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
            raise ValueError("Missing/nonfinite exact response-token logprob, including EOS")
        result.append(float(logprob))
    return result


def compare_logprobs(expected, observed, config):
    if (not expected or len(expected) != len(observed)
            or not all(math.isfinite(x) for x in expected + observed)):
        raise ValueError("Numerical parity needs matching finite response/EOS vectors")
    differences = [o - e for e, o in zip(expected, observed)]
    maximum = max(abs(x) for x in differences)
    mean = sum(abs(x) for x in differences) / len(differences)
    logratio = sum(differences)
    passed = (maximum <= config["max_token_absolute_logprob_delta"]
              and mean <= config["max_mean_absolute_logprob_delta"]
              and abs(logratio) <= config["max_action_absolute_logratio"])
    return {"passed": passed, "response_tokens_including_EOS": len(expected),
            "max_token_absolute_logprob_delta": maximum, "mean_absolute_logprob_delta": mean,
            "action_log_likelihood_ratio": logratio}


def verify_parity_preparation(directory, config, *, commit, config_sha, code_sha):
    report = json.loads((directory / "report.json").read_text())
    if (report["status"] != "PPO_PARITY_CPU_PREPARATION_PASS_GPU_NUMERICS_REMAIN"
            or report["source_commit"] != commit or report["config_sha256"] != config_sha
            or report["code_sha256"] != code_sha or report["gpu_requested"]
            or report["optimizer_updates"] or report["samples"] != 20):
        raise ValueError("Parity GPU task requires its exact-source completed CPU receipt")
    for name, sha in report["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(directory / name) != sha:
            raise ValueError("Prepared parity artifact changed")
    rows = [json.loads(line) for line in (directory / "tokens.jsonl").read_text().splitlines()]
    if len(rows) != 20 or len({row["user_id"] for row in rows}) != 20:
        raise ValueError("Twenty unique frozen synthetic inputs required")
    for row in rows:
        completion_request(row, config)
    return rows, report
