#!/usr/bin/env python3
"""Review the failed real-memory v3 artifacts offline; no model or promotion.

Recheck captured requests, raw JSON, candidate roles and durable accounting.
Do not repair the failed run, rewrite its report, or compute ranking metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cmirank.memory_smoke import (
    inspect_retrieval_grounding, load_memory_contract, object_sha256, validate_retrieval, validate_scores,
)
from src.cmirank.policy_data import NEUTRAL_PSEUDO_INSTRUCTION
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.request import RankRequest
from src.models.llm_client import validate_json_shape
from src.models.reranker_llm import LLMReranker


def review(run: Path, candidate_run: Path) -> dict:
    config, config_sha = load_memory_contract(ROOT, 3)
    manifest = json.loads((run / "manifest.json").read_text())
    cleanup = json.loads((run / "cleanup.json").read_text())
    if (run.name != config["run_id"] or manifest["config_sha256"] != config_sha
            or manifest["source_commit"] != "f212c87b69c67e453ff7aac9a4b2d05bac662de0"
            or manifest["config"] != config or cleanup["process_exit_code"] != 1
            or not cleanup["child_exited"] or not cleanup["owned_server_exited"]
            or not cleanup["gpu_released"] or not cleanup["allocation_still_running"]):
        raise ValueError("Not the completed/released failed v3 smoke")
    candidate_file = candidate_run / "full-hnv/candidate-rows.jsonl"
    if (candidate_run.name != config["candidate_run_id"]
            or file_sha256(candidate_file) != config["full_candidates_sha256"]):
        raise ValueError("Unreviewed candidate data")
    candidates = {r["policy_input"]["episode_id"]: r for r in
                  map(json.loads, candidate_file.read_text().splitlines())}
    physical = list(map(json.loads, (run / "physical-requests.jsonl").read_text().splitlines()))
    traces = list(map(json.loads, (run / "retrieval-trace.jsonl").read_text().splitlines()))
    requests, responses = [x for x in physical if x["event"] == "request"], [x for x in physical if x["event"] == "response"]
    if len(requests) != len(responses) or len(requests) != 2 or len(traces) != 1:
        raise ValueError("Unexpected v3 failure scope")
    total_input = total_output = 0
    for request, response in zip(requests, responses):
        if (request["request_sha256"] != object_sha256(request["kwargs"])
                or request["request_sha256"] != response["request_sha256"]
                or request["physical_attempt"] != response["physical_attempt"]
                or response["finish_reason"] != "stop"):
            raise ValueError("Physical journal is inconsistent or truncated")
        validate_json_shape(json.loads(response["content"]), request["kwargs"]["response_format"]["json_schema"]["schema"])
        total_input += response["usage"]["prompt_tokens"]
        total_output += response["usage"]["completion_tokens"]
    with sqlite3.connect(f"file:{run / 'request-budget.sqlite'}?mode=ro", uri=True) as connection:
        cap, used = connection.execute("SELECT lim, used FROM budget").fetchone()
    if cap != 110 or used != len(requests):
        raise ValueError("Durable physical budget differs")
    trace = traces[0]
    episode = trace["episode_id"]
    details = trace["details"]
    source = candidates[episode]
    request = RankRequest.from_dict(details["rank_request"])
    ids = source["policy_input"]["candidate_ids"]
    if (episode != "books-policy-v1-user-731-warmup" or request.instruction != NEUTRAL_PSEUDO_INSTRUCTION
            or request.snapshot_id != manifest["graph_snapshot_sha256"]
            or [row["id"] for row in request.candidates] != ids
            or request.baseline_prompt_kwargs()["instruction"] != NEUTRAL_PSEUDO_INSTRUCTION
            or requests[-1]["kwargs"]["messages"] != LLMReranker(None).build_rerank_prompt(**request.baseline_prompt_kwargs())
            or json.loads(responses[0]["content"]) != details["retrieval_bundle"]
            or json.loads(responses[1]["content"])["scores"] != details["rerank_scores"]):
        raise ValueError("Raw responses/captured request differ from actual baseline API inputs")
    validate_scores(details["rerank_scores"], ids)
    grounding = inspect_retrieval_grounding(details, request.user_id)
    try:
        validate_retrieval(details, request.user_id)
    except ValueError:
        strict_passed = False
    else:
        strict_passed = True
    stages = [set(row["kwargs"]["response_format"]["json_schema"]["schema"]["properties"]) for row in requests]
    if stages != [{"facets", "support_edges"}, {"scores"}] or (run / "stage-w-journal.jsonl").exists():
        raise ValueError("Unexpected Stage-W call or mutation")
    return {"status": "FAILED_SMOKE_REVIEW_COMPLETE_NOT_PROMOTED", "source_run_id": run.name,
        "physical_requests": used, "physical_responses": len(responses), "physical_retries": 0,
        "input_tokens": total_input, "output_tokens": total_output, "total_tokens": total_input + total_output,
        "stage_r_calls": 1, "stage_rerank_calls": 1, "stage_w_calls": 0, "completed_smoke_users": 0,
        "schema_and_ten_candidate_ids_valid": True, "exact_rankrequest_api_prompt_parity": True,
        "strict_collaborative_only_gate_passed": strict_passed, "grounding": grounding,
        "warmup_positive_audit_only": source["reward_audit"]["positive_item_id"],
        "elapsed_seconds": json.loads((run / "failure.json").read_text())["elapsed_seconds"],
        "gpu_released": True, "semantic_memory_grounding_proven": False,
        "full_memory_cache_promoted": False, "training_ready": False,
        "source_artifact_sha256": {name: file_sha256(run / name) for name in
            ("manifest.json", "physical-requests.jsonl", "retrieval-trace.jsonl", "cleanup.json", "request-budget.sqlite")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite a review")
    result = review(args.run_dir, args.candidate_run_dir)
    result["review_code_sha256"] = file_sha256(Path(__file__))
    result["grounding_code_sha256"] = file_sha256(ROOT / "src/cmirank/memory_smoke.py")
    result["review_source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "review.json").write_text(artifact_json_dumps(result))
    print(artifact_json_dumps({key: value for key, value in result.items() if key not in ("grounding", "source_artifact_sha256")}), end="")


if __name__ == "__main__":
    main()
