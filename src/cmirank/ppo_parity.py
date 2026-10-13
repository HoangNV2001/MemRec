"""Outcome-free protocol/mask guards for an initial vLLM/HF numerical probe.

No CUDA, model loading or HTTP at import time. Does not implement PPO training.
"""

import json
import math
import os
import time
from pathlib import Path

from .provenance import file_sha256


def load_parity_config(root, version=None):
    version = int(os.environ.get("MEMREC_PARITY_CONTRACT_VERSION", "8")) if version is None else version
    if version not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        raise ValueError("Unknown parity contract")
    config = json.loads((root / "configs/cmirank/ppo_parity_v1.json").read_text())
    path = root / f"configs/cmirank/ppo_parity_v{version}.json"
    if version >= 2:
        retry = json.loads((root / "configs/cmirank/ppo_parity_v2.json").read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "language_model_only", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_parity_v1.json"
                or retry["language_model_only"] is not True):
            raise ValueError("Text-only serving retry cannot alter weights, tokens, or numerical thresholds")
        config.update(retry)
    if version >= 3:
        retry = json.loads((root / "configs/cmirank/ppo_parity_v3.json").read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "gpu_index",
                          "resource_policy", "gdn_prefill_backend", "server_environment", "server_kernel_config", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_parity_v2.json"
                or retry["gpu_index"] != 1 or retry["resource_policy"] != "gpu1_only_gpu0_all_workloads_read_only"
                or retry["gdn_prefill_backend"] != "triton"
                or retry["server_kernel_config"] != {"enable_flashinfer_autotune": False}
                or retry["server_environment"] != {"VLLM_USE_DEEP_GEMM": "0", "VLLM_DEEP_GEMM_WARMUP": "skip"}):
            raise ValueError("GPU1/native backend retry cannot alter the numerical/research contract")
        config.update(retry)
    if version == 4:
        retry = json.loads(path.read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "physical_request_cap",
                          "timeout_minutes", "hf_precision_diagnostic", "diagnostic_variants",
                          "diagnostic_parent_run_id", "diagnostic_parent_sha256", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_parity_v3.json"
                or retry["hf_precision_diagnostic"] is not True or retry["physical_request_cap"] != 0
                or retry["timeout_minutes"] != 10
                or retry["diagnostic_variants"] != ["baseline_bf16_fa2", "readout_bf16_response_only",
                    "readout_fp32_response_only", "bf16_sdpa", "fp32_sdpa"]):
            raise ValueError("Precision diagnosis cannot choose a subset, alter thresholds or promote a profile")
        config.update(retry)
    if version >= 5:
        retry = json.loads((root / "configs/cmirank/ppo_parity_v5.json").read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "hf_kernel_overlay",
                          "fla_prepare_run_id", "fla_prepare_report_sha256", "native_gdn_gradient_samples",
                          "kernel_probe_lengths", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_parity_v3.json"
                or retry["hf_kernel_overlay"] != "cmirank-fla-052-torch211-cu129-v1-hnv"
                or retry["native_gdn_gradient_samples"] != 20
                or retry["kernel_probe_lengths"] != [33, 64, 65, 128, 529]):
            raise ValueError("Native GDN candidate cannot change checkpoint/dtype/masks/bounds/research recipe")
        config.update(retry)
    if version >= 6:
        retry = json.loads((root / "configs/cmirank/ppo_parity_v6.json").read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "hf_cuda_toolchain",
                          "toolchain_prepare_run_id", "hf_worker_environment", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_parity_v5.json"
                or retry["hf_cuda_toolchain"] != "cuda-12.9.1-nvcc-12.9.86-hnv"
                or retry["toolchain_prepare_run_id"] != "cmirank-cuda129-toolchain-cpu-v1-20261013-hnv"
                or retry["hf_worker_environment"] != {"FLA_TILELANG": "1"}):
            raise ValueError("Private compiler retry cannot change the numerical or scientific recipe")
        config.update(retry)
    if version >= 7:
        retry = json.loads((root / "configs/cmirank/ppo_parity_v7.json").read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "hf_causal_overlay",
                          "native_causal_gradient_samples", "toolchain_prepare_source_commit",
                          "toolchain_prepare_report_sha256", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_parity_v6.json"
                or retry["hf_causal_overlay"] != "cmirank-causalconv17-torch211-cu129-v1-hnv"
                or retry["native_causal_gradient_samples"] != 20
                or retry["toolchain_prepare_source_commit"] != "78a2c07793827c9470e02d24fffc25e362cb6b45"
                or retry["toolchain_prepare_report_sha256"] != "84a542cc7b36a68d84e73554bf4d94306c8a966ba39f89ff4bbc92075eb55c50"):
            raise ValueError("Full native fast-path retry cannot change the research or parity contract")
        config.update(retry)
    if version >= 8:
        retry = json.loads((root / "configs/cmirank/ppo_parity_v8.json").read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "causal_prepare_source_commit",
                          "causal_prepare_report_sha256", "hf_release_wait_seconds", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_parity_v7.json"
                or retry["causal_prepare_source_commit"] != "132e4148c1bfa0bb21be2a6bb7fda75ab416f86d"
                or retry["causal_prepare_report_sha256"] != "440b838398d5b11ba035e83987e5c0b75caffcaad5a3bb804d89baff56fe9550"
                or retry["hf_release_wait_seconds"] != 15):
            raise ValueError("Release-drain retry cannot weaken GPU or numerical guards")
        config.update(retry)
    if version == 9:
        retry = json.loads(path.read_text())
        if (set(retry) != {"schema_version", "inherits", "prepare_run_id", "run_id", "repeatability_diagnostic",
                          "server_boots", "replays_per_boot", "physical_request_cap", "cache_policy",
                          "hf_repeat_reference_sha256", "reason"}
                or retry["inherits"] != "configs/cmirank/ppo_parity_v8.json"
                or retry["repeatability_diagnostic"] is not True or retry["server_boots"] != 2
                or retry["replays_per_boot"] != 2 or retry["physical_request_cap"] != 80
                or retry["cache_policy"] != "separate_cold_private_namespace_per_boot_warm_within_boot"
                or retry["hf_repeat_reference_sha256"] != "2f88c7e4c042ac1dc8e3a7caf789deb655a9013015aa7587b3c399e679126626"):
            raise ValueError("Repeatability is a fixed2x2x20 diagnosis, not a profile/threshold search")
        config.update(retry)
    expected_gpu = 1 if version >= 3 else 0
    expected_policy = f"gpu{expected_gpu}_only_gpu{1-expected_gpu}_all_workloads_read_only"
    if (config["samples"] != 20 or config["physical_request_cap"] != (0 if version == 4 else 80 if version == 9 else 20)
            or config["generated_tokens_per_request"] != 1 or config["tensor_parallel_size"] != 1
            or config["gpu_index"] != expected_gpu or config["resource_policy"] != expected_policy
            or config["optimizer_updates"] or config["books_outcomes_accessed"]
            or config["ranking_metrics_computed"] or config["training_ready"] or config["ppo_worker_proven"]
            or config["temperature"] != 0 or config["prompt_logprob_temperature"] != 1):
        raise ValueError("Not the twenty-synthetic-request no-update parity gate")
    return config, path


def summarize_repeatability(observations, rows, config):
    expected = {(b,r,u["user_id"]) for b in range(2) for r in range(2) for u in rows}
    seen = {(x["boot"],x["replay"],x["user_id"]) for x in observations}
    if (not config.get("repeatability_diagnostic") or len(observations) != 80 or seen != expected
            or len(rows) != 20 or len({u["user_id"] for u in rows}) != 20):
        raise ValueError("Repeatability requires every fixed2x2x20 observation, no duplicate/omission")
    vectors = {(x["boot"],x["replay"],x["user_id"]):x["logprobs"] for x in observations}
    lengths = {u["user_id"]:len(u["input_ids"])-u["prompt_tokens"] for u in rows}
    if any(len(x["logprobs"]) != lengths[x["user_id"]] for x in observations):
        raise ValueError("Repeatability vectors must retain every original response token including EOS")
    pairs = []
    for u in rows:
        for b in range(2):
            a,z = vectors[b,0,u["user_id"]],vectors[b,1,u["user_id"]]
            pairs.append({"kind":"within_boot","boot":b,"user_id":u["user_id"],
                          "exact_equal_json_vectors":a==z,**compare_logprobs(a,z,config)})
        for r in range(2):
            a,z = vectors[0,r,u["user_id"]],vectors[1,r,u["user_id"]]
            pairs.append({"kind":"across_boot","replay":r,"user_id":u["user_id"],
                          "exact_equal_json_vectors":a==z,**compare_logprobs(a,z,config)})
    summary = {}
    for kind in ("within_boot","across_boot"):
        subset = [p for p in pairs if p["kind"]==kind]
        summary[kind] = {"pairs":len(subset),"exact_equal_pairs":sum(p["exact_equal_json_vectors"] for p in subset),
            "original_bound_pass_pairs":sum(p["passed"] for p in subset),
            "max_token_absolute_logprob_delta":max(p["max_token_absolute_logprob_delta"] for p in subset),
            "max_action_absolute_logratio":max(abs(p["action_log_likelihood_ratio"]) for p in subset)}
    return summary,pairs


def parity_code_hashes(root):
    return {name: file_sha256(root / name) for name in (
        "src/cmirank/ppo_parity.py", "scripts/cmirank/22_prepare_ppo_parity_cpu.py",
        "scripts/cmirank/23_smoke_vllm_hf_parity_gpu.py", "scripts/cmirank/24_hf_parity_worker_gpu.py",
        "src/cmirank/ppo_compat.py", "src/cmirank/reserved_handoff.py", "src/cmirank/t0_authority.py", "src/cmirank/gpu_resources.py",
        "src/cmirank/cuda_toolchain.py", "scripts/cmirank/26_prepare_cuda_toolchain_cpu.py",
        "src/cmirank/causal_conv.py", "scripts/cmirank/27_build_causal_conv_cpu.py",
        "scripts/cmirank/run_real_policy_smoke.sh", "scripts/cmirank/run_real_policy_with_keeper.sh")}


def server_options(config):
    options = ["--served-model-name", config["served_model_name"], "--host", "127.0.0.1", "--port", str(config["port"]),
        "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--generation-config", "vllm",
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--max-model-len", str(config["max_model_len"]), "--max-num-seqs", "1",
        "--max-num-batched-tokens", str(config["max_model_len"]), "--seed", str(config["seed"]),
        "--no-enable-prefix-caching", "--enforce-eager"]
    if config.get("language_model_only"):
        options.append("--language-model-only")
    if config.get("gdn_prefill_backend"):
        options.extend(["--gdn-prefill-backend", config["gdn_prefill_backend"]])
    if config.get("server_kernel_config"):
        options.extend(["--kernel-config", json.dumps(config["server_kernel_config"], sort_keys=True)])
    return options


def wait_owned_hf_release(snapshot, config, expected_uuid, *, child_exited, emit, child_kind="hf",
                          clock=time.monotonic, sleep=time.sleep):
    """Bounded observation only: original idle thresholds, no PID/step signals.

    A returned child exit code does not guarantee an instantaneous NVML idle
    sample. Do not load the next model until the same physical card is clear.
    """
    from .gpu_resources import select_reserved_gpu
    if not child_exited or child_kind not in ("hf", "vllm") or config.get("hf_release_wait_seconds") != 15:
        raise ValueError("Only confirmed owned-child exit permits the fixed15s release drain")
    started = clock()
    attempts = 0
    while True:
        cards, apps, raw, raw_apps = snapshot()
        selected = [card for card in cards if card.index == config["gpu_index"]]
        if len(selected) != 1 or selected[0].uuid != expected_uuid:
            raise ValueError("Physical GPU changed during owned-child release; no fallback")
        attempts += 1
        emit({"attempt":attempts, "elapsed_seconds":clock()-started,
              "gpu_snapshot":raw, "compute_snapshot":raw_apps,
              "gpu_uuid":expected_uuid, "child_exited":True, "owned_child_kind":child_kind})
        try:
            card = select_reserved_gpu(cards, set(apps), gpu_index=config["gpu_index"])
        except ValueError:
            if clock()-started >= config["hf_release_wait_seconds"]:
                raise ValueError(f"Owned {child_kind} release did not meet original idle guards within15s")
            sleep(1)
            continue
        return {"child_exited":True, "gpu_released":True, "gpu_uuid":card.uuid,
                "attempts":attempts, "elapsed_seconds":clock()-started}


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
    if config.get("language_model_only") and report.get("native_text_only_registry_inputs_supported") is not False:
        raise ValueError("Native registry must bypass multimodal budget before GPU text-only retry")
    if config.get("gdn_prefill_backend") and (
            report.get("native_gdn_prefill_backend") != "triton"
            or report.get("native_server_environment_checked") != config["server_environment"]
            or report.get("native_kernel_warmup_skip_no_model_cpu_probe") is not True):
        raise ValueError("Native backend/environment receipt required before the kernel retry")
    if config.get("hf_kernel_overlay") and report.get("native_fla_overlay_audited") is not True:
        raise ValueError("Native FLA overlay CPU audit receipt is required before GPU candidate qualification")
    if config.get("hf_cuda_toolchain") and report.get("private_cuda_toolchain_cpu_replay_passed") is not True:
        raise ValueError("Native TileLang compiler regression must pass CPU replay before GPU handoff")
    if config.get("hf_causal_overlay") and report.get("native_causal_overlay_cpu_audited") is not True:
        raise ValueError("Native convolution CPU build/import must precede GPU full fast-path qualification")
    if config.get("hf_precision_diagnostic"):
        rehearsal = report.get("native_precision_control_cpu_rehearsal") or {}
        if (rehearsal.get("samples_per_variant") != 20 or rehearsal.get("variants") != config["diagnostic_variants"]
                or rehearsal.get("scope") != "MINIATURE_SDPA_CPU_CONTROL_FLOW_ONLY_BASELINE_FA2_NOT_EXECUTED"
                or rehearsal.get("official_weights_loaded") is not False or rehearsal.get("gpu_requested") is not False
                or "precision-cpu-logprobs.jsonl" not in report["artifact_sha256"]):
            raise ValueError("Twenty miniature CPU controls must precede full GPU precision diagnosis")
    for name, sha in report["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(directory / name) != sha:
            raise ValueError("Prepared parity artifact changed")
    rows = [json.loads(line) for line in (directory / "tokens.jsonl").read_text().splitlines()]
    if len(rows) != 20 or len({row["user_id"] for row in rows}) != 20:
        raise ValueError("Twenty unique frozen synthetic inputs required")
    for row in rows:
        completion_request(row, config)
    return rows, report


def verify_fla_overlay(private, config):
    directory = private / "runs" / config["fla_prepare_run_id"]
    if file_sha256(directory / "report.json") != config["fla_prepare_report_sha256"]:
        raise ValueError("Wrong native FLA preparation receipt")
    report = json.loads((directory / "report.json").read_text())
    probe = report["probe"]
    if (report["status"] != "FRESH_NATIVE_FLA_CPU_IMPORT_PASS_GPU_KERNEL_AND_PARITY_GATES_REMAIN"
            or report["overlay_name"] != config["hf_kernel_overlay"] or report["gpu_requested"]
            or report["full_weights_loaded"] or report["training_ready"] or not report["base_environment_unchanged"]
            or probe["samples"] != 20 or probe["cuda_initialized"] or probe["full_weights_loaded"]
            or not probe["hf_cuda_only_availability_guard_respected"] or probe["hf_cuda_dispatch_proven"]
            or any(probe["versions"][n] != "0.5.2" for n in ("flash-linear-attention", "fla-core"))):
        raise ValueError("Native GDN candidate requires honest completed CPU-only pinned-package inspection")
    overlay = private / "overlays" / config["hf_kernel_overlay"]
    for name, digest in report["overlay_sha256"].items():
        path = overlay / name
        if not path.resolve().is_relative_to(overlay) or file_sha256(path) != digest:
            raise ValueError("Native FLA overlay bytes changed after CPU audit")
    return report


def diagnostic_parent(directory, config, rows):
    """Source/hash-bound frozen probabilities, not another HTTP/model request."""
    if directory.name != config["diagnostic_parent_run_id"]:
        raise ValueError("Wrong numerical diagnostic predecessor")
    for name, digest in config["diagnostic_parent_sha256"].items():
        if Path(name).name != name or file_sha256(directory / name) != digest:
            raise ValueError("Numerical predecessor artifact changed")
    report = json.loads((directory / "report.json").read_text())
    cleanup = json.loads((directory / "cleanup.json").read_text())
    handback = json.loads((directory / "keeper-handback.json").read_text())
    if (report["status"] != "VLLM_HF_RESPONSE_PARITY_FAIL_NUMERICAL_DIAGNOSIS_REQUIRED"
            or report["samples"] != 20 or report["physical_requests"] != 20 or report["optimizer_updates"]
            or report["books_outcomes_accessed"] or not cleanup["gpu_released"] or not cleanup["child_exited"]
            or not cleanup["allocation_still_running"] or handback["status"] != "KEEPER_GPU1_HANDBACK_VERIFIED"
            or handback["reserved_job_cancelled"] or handback["gpu0_signalled"]):
        raise ValueError("Predecessor must be the reviewed20-sample numerical failure with clean handback")
    physical = [json.loads(s) for s in (directory / "physical-requests.jsonl").read_text().splitlines()]
    hf = [json.loads(s) for s in (directory / "hf-logprobs.jsonl").read_text().splitlines()]
    if len(physical) != 40 or len(hf) != 20:
        raise ValueError("Exact20 request/response pairs and HF vectors required")
    result = []
    original = {**config, "physical_request_cap": 20}
    for i, row in enumerate(rows):
        request, response = physical[2*i:2*i+2]
        if (request["event"] != "request" or response["event"] != "response"
                or request["user_id"] != row["user_id"] or response["user_id"] != row["user_id"]
                or hf[i]["user_id"] != row["user_id"] or request["attempt"] != i+1 or response["attempt"] != i+1
                or request["payload"] != completion_request(row, original)):
            raise ValueError("Diagnostic predecessor changed input/order/payload/budget")
        result.append({"user_id": row["user_id"], "hf": hf[i]["logprobs"],
                       "vllm": response_prompt_logprobs(response["response"], row)})
    return result
