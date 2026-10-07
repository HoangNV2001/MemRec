#!/usr/bin/env python3
"""Review failed real-memory v3/v6 artifacts offline; no model or promotion.

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
from types import SimpleNamespace

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


def review_v6(run: Path, candidate_run: Path) -> dict:
    from src.cmirank.candidate_artifacts import smoke_user_ids
    from src.cmirank.gpu_resources import compute_gpu_processes, gpu_release_verified, parse_gpu_snapshot
    from src.cmirank.memory_review import read_rows, review_partial_records
    from src.data.dataset_base import RecDataset
    from src.memory.storage import MemoryStorage

    config, config_sha = load_memory_contract(ROOT, 6)
    manifest = json.loads((run / "manifest.json").read_text())
    cleanup = json.loads((run / "cleanup.json").read_text())
    if (run.name != config["run_id"] or manifest["config_sha256"] != config_sha
            or manifest["source_commit"] != "19b9779abb2288ce5b2468b2506a1a263c31615a"
            or manifest["config"] != config or cleanup["process_exit_code"] != 1
            or not all(cleanup[key] for key in
                       ("child_exited", "owned_server_exited", "gpu_released", "allocation_still_running"))
            or manifest["original_instruction_accessed"] or manifest["original_test_candidates_accessed"]
            or manifest["original_suffix_item_ids_parsed_or_used"]):
        raise ValueError("Not the completed/released failed v6 diagnostic")
    uuid = cleanup["gpu_uuid"]
    before = next(card for card in parse_gpu_snapshot((run / "gpus-load-time.csv").read_text()) if card.uuid == uuid)
    after = next(card for card in parse_gpu_snapshot((run / "gpus-after.csv").read_text()) if card.uuid == uuid)
    before_apps = compute_gpu_processes((run / "apps-load-time.csv").read_text()).get(uuid, set())
    after_apps = compute_gpu_processes((run / "apps-after.csv").read_text()).get(uuid, set())
    if (before.index != 1 or after.index != 1 or manifest["gpu_uuid"] != uuid
            or after.used_mib != cleanup["gpu_used_mib_after"]
            or not gpu_release_verified(after, before, before_apps, after_apps)
            or " RUNNING " not in (run / "allocation-after.txt").read_text()):
        raise ValueError("Recorded GPU release/allocation evidence differs")
    candidate_file = candidate_run / "full-hnv/candidate-rows.jsonl"
    if (candidate_run.name != config["candidate_run_id"]
            or file_sha256(candidate_file) != config["full_candidates_sha256"]):
        raise ValueError("Unreviewed candidate data")
    rows = read_rows(candidate_file)
    candidates = {row["policy_input"]["episode_id"]: row for row in rows}
    episode_config = json.loads((ROOT / "configs/cmirank/episode_candidates_v1.json").read_text())
    eligible = sorted({row["provenance"]["user_id"] for row in rows})
    if len(candidates) != len(rows) or manifest["query_user_ids"] != smoke_user_ids(eligible, episode_config):
        raise ValueError("Episode identities or fixed smoke cohort differs")
    metadata_path = ROOT / "data/processed/instructrec-books/instructrec-books.meta"
    candidate_config = json.loads((ROOT / "configs/cmirank/candidate_sampler_v1.json").read_text())
    if file_sha256(metadata_path) != manifest["metadata_sha256"] or manifest["metadata_sha256"] != candidate_config["metadata_sha256"]:
        raise ValueError("Static metadata differs from the recorded initial memory")
    holder = SimpleNamespace(data_path=metadata_path.with_suffix(".inter"), item_metadata=None)
    RecDataset.load_item_metadata(holder)  # Static metadata only, never RecDataset.__init__ or suffix outcomes.
    storage = MemoryStorage()
    storage.initialize_item_descriptions(holder.item_metadata)
    result = review_partial_records(run, candidates=candidates, config=config, manifest=manifest,
                                    storage=storage, metadata=holder.item_metadata)
    if (result["physical_requests"], result["completed_warmup_users"], result["stage_r_calls"]) != (35, 11, 12):
        raise ValueError("Unexpected v6 terminal scope")
    result.update({"source_run_id": run.name, "source_commit": manifest["source_commit"],
        "config_sha256": config_sha, "gpu_released": True,
        "elapsed_seconds": json.loads((run / "failure.json").read_text())["elapsed_seconds"],
        "source_artifact_sha256": {path.name: file_sha256(path) for path in sorted(run.iterdir()) if path.is_file()}})
    return result


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
    parser.add_argument("--contract-version", type=int, choices=(3, 6), default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite a review")
    result = (review if args.contract_version == 3 else review_v6)(args.run_dir, args.candidate_run_dir)
    result["review_code_sha256"] = file_sha256(Path(__file__))
    result["grounding_code_sha256"] = file_sha256(ROOT / "src/cmirank/memory_smoke.py")
    if args.contract_version == 6:
        result["partial_review_code_sha256"] = file_sha256(ROOT / "src/cmirank/memory_review.py")
    result["review_source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "review.json").write_text(artifact_json_dumps(result))
    print(artifact_json_dumps({key: value for key, value in result.items()
                              if key not in ("grounding", "failure", "source_artifact_sha256")}), end="")


if __name__ == "__main__":
    main()
