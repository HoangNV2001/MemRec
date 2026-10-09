"""Twenty-request functional N−1/direct smoke, never Books scoring/training."""

from dataclasses import asdict
import json
from pathlib import Path

from .iterative_env import rank_iteratively
from .labels import label_to_item
from .parser import parse_direct_ranking
from .prompts import render_direct_prompt
from .provenance import file_sha256
from .request import RankRequest


def load_policy_smoke_contract(root: Path, version: int) -> tuple[dict, Path]:
    """V2 is a resource-only GPU0 amendment; freeze every scientific V1 field."""
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unknown functional smoke contract")
    path = root / f"configs/cmirank/real_policy_smoke_v{version}.json"
    config = json.loads(path.read_text())
    if version == 2:
        base_path = root / "configs/cmirank/real_policy_smoke_v1.json"
        base = json.loads(base_path.read_text())
        added = {"cpu_run_id", "gpu_index", "resource_delta", "inherited_contract_sha256"}
        if (config.get("inherited_contract_sha256") != file_sha256(base_path)
                or set(config) != set(base) | added or config["gpu_index"] != 0
                or type(config["gpu_index"]) is not int
                or config["resource_delta"] != "researcher_authorized_gpu0_only_restore_omni_gen_0_protect_gpu1"
                or config["cpu_run_id"] != "cmirank-qwen35-real-policy-cpu-v2-gpu0-20261009-hnv"
                or config["run_id"] != "cmirank-qwen35-real-nminus1-direct-smoke-v2-gpu0-20261009-hnv"
                or config["schema_version"] != 2
                or any(config[key] != value for key, value in base.items() if key not in ("schema_version", "run_id"))):
            raise ValueError("GPU0 amendment must not change the frozen scientific contract")
    return config, path


def load_secondary_inputs(run: Path, review: Path, *, expected_review_sha: str) -> tuple[list[RankRequest], dict]:
    if file_sha256(review) != expected_review_sha:
        raise ValueError("Functional smoke requires the independently reviewed v7 control")
    receipt = json.loads(review.read_text())
    if (receipt["source_run_id"] != run.name
            or receipt["status"] != "SECONDARY_CONTROL_REVIEW_PASS_LITERAL_VALIDITY_NOT_SEMANTIC_GROUNDING"
            or not receipt["pseudo_memory_read_only_verified"] or receipt["primary_provider_replaced"]
            or receipt["unknown_context_occurrences"] or receipt["training_ready"]):
        raise ValueError("Not reviewed read-only secondary memory")
    source = run / "policy-inputs.jsonl"
    if file_sha256(source) != receipt["source_artifact_sha256"][source.name]:
        raise ValueError("Frozen policy inputs changed")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    requests = [RankRequest.from_dict(row["rank_request"]) for row in rows]
    users = [request.user_id for request in requests]
    if len(rows) != 20 or len(set(users)) != 20 or users != sorted(users):
        raise ValueError("Exactly twenty ordered frozen requests required")
    for row, request in zip(rows, requests):
        if (set(row) != {"episode_id", "rank_request", "memory_state_sha256", "warmup_scope_user_ids"}
                or row["memory_state_sha256"] != receipt["replayed_prefix_memory_sha256"]
                or row["warmup_scope_user_ids"] != users or len(request.candidates) != 10):
            raise ValueError("Frozen input scope/state/label separation differs")
    return requests, {"review_sha256": expected_review_sha, "policy_inputs_sha256": file_sha256(source),
        "rank_request_hashes": [request.sha256() for request in requests], "user_ids": users,
        "memory_state_sha256": receipt["replayed_prefix_memory_sha256"],
        "comparison_role": "secondary_functional_diagnostic_not_primary_replacement", "training_ready": False}


class IncompleteGeneration(ValueError):
    pass


def verify_cpu_token_audit(cpu: Path, *, commit: str, config_sha: str, prompt_sha: str,
                           parser_sha: str, marker_sha: str, provenance: dict, max_input: int) -> list[dict]:
    """A fresh GPU process consumes the receipt, never a CPU CUDA-driver cache."""
    report = json.loads((cpu / "report.json").read_text())
    manifest = json.loads((cpu / "manifest.json").read_text())
    audit_path = cpu / "cpu-token-audit.json"
    rows = json.loads(audit_path.read_text())
    expected_manifest = {"source_commit": commit, "config_sha256": config_sha,
        "prompt_code_sha256": prompt_sha, "parser_code_sha256": parser_sha,
        "checkpoint_marker_sha256": marker_sha, "cpu_audit_only": True}
    expected_pairs = [(uid, mode) for uid in provenance["user_ids"] for mode in ("iterative", "direct")]
    if (report["status"] != "CPU_REAL_POLICY_INPUT_TOKEN_AUDIT_PASS_NO_GPU"
            or report["source_commit"] != commit or report["config_sha256"] != config_sha
            or report["gpu_requested"] or report["model_weights_loaded"] or report["training_ready"]
            or report["token_audit_sha256"] != file_sha256(audit_path)
            or any(manifest.get(key) != value for key, value in expected_manifest.items())
            or any(report.get(key) != value for key, value in provenance.items())
            or [(row["user_id"], row["mode"]) for row in rows] != expected_pairs
            or any(type(row["tokens"]) is not int or not 0 < row["tokens"] <= max_input for row in rows)):
        raise ValueError("CPU real-input/tokenizer receipt differs from the exact GPU source/state")
    return rows


def run_functional_smoke(requests: list[RankRequest], *, generate, emit) -> dict:
    """Invalid model outputs remain failures; never peek at a target/reward."""
    if len(requests) != 20 or len({request.user_id for request in requests}) != 20:
        raise ValueError("Twenty unique users required")
    counters = {"iterative_valid": 0, "direct_valid": 0, "generation_calls": 0, "iterative_steps": 0}
    for sample, request in enumerate(requests):
        mapping = label_to_item([row["id"] for row in request.candidates])

        def observed(prompt, mode):
            result = generate(prompt)
            counters["generation_calls"] += 1
            counters["iterative_steps"] += int(mode == "iterative")
            emit("generation", {"sample": sample, "mode": mode, "user_id": request.user_id,
                "rank_request_sha256": request.sha256(), **result})
            if not result["complete"]:
                raise IncompleteGeneration("response_not_terminated")
            return result["raw_output"]

        try:
            iterative = rank_iteratively(request, lambda prompt: observed(prompt, "iterative"))
            iterative_valid, reason = iterative.valid, iterative.failure_reason
            ranking = list(iterative.ranked_candidate_ids) if iterative.valid else []
            traces = [asdict(row) for row in iterative.trace]
        except IncompleteGeneration as error:
            iterative_valid, reason, ranking, traces = False, str(error), [], []
        counters["iterative_valid"] += int(iterative_valid)
        emit("episode", {"sample": sample, "mode": "iterative", "user_id": request.user_id,
            "rank_request_sha256": request.sha256(), "valid": iterative_valid, "failure_reason": reason,
            "ranked_candidate_ids": ranking, "trace": traces})
        try:
            raw = observed(render_direct_prompt(request), "direct")
            parsed = parse_direct_ranking(raw, list(mapping))
            direct_valid, reason = parsed.valid, parsed.failure_reason
            ranking = [mapping[label] for label in parsed.labels] if parsed.valid else []
        except IncompleteGeneration as error:
            direct_valid, reason, ranking = False, str(error), []
        counters["direct_valid"] += int(direct_valid)
        emit("episode", {"sample": sample, "mode": "direct", "user_id": request.user_id,
            "rank_request_sha256": request.sha256(), "valid": direct_valid,
            "failure_reason": reason, "ranked_candidate_ids": ranking})
        emit("progress", {"completed_users": sample + 1, **counters})
    passed = counters["iterative_valid"] == counters["direct_valid"] == 20
    if passed and (counters["generation_calls"] != 200 or counters["iterative_steps"] != 180):
        raise ValueError("Passed N−1/direct smoke must have exactly 180+20 calls")
    return {"status": "FUNCTIONAL_SMOKE_PASS" if passed else "FUNCTIONAL_SMOKE_FAILED_NO_PROMOTION",
        "users": 20, **counters, "model_updates": 0, "output_repair": False,
        "ranking_metrics_computed": False, "primary_provider_replaced": False,
        "training_ready": False, "ppo_proven": False, "semantic_memory_grounding_proven": False}
