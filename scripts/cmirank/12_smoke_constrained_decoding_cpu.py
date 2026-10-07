#!/usr/bin/env python3
"""Compile input-only ID schemas on CPU before the secondary real GPU smoke.

Use the reviewed v6 physical inputs, not labels or outcome scores. Check exact
input-role domains and compile 35 schemas plus the empty propagation domain.
No model weights, generation, API requests, CUDA selection or generator handoff.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.constrained_decoding import constrain_id_properties, validate_constrained_output
from src.cmirank.memory_review import read_rows, response_schema
from src.cmirank.memory_smoke import inspect_retrieval_grounding, load_memory_contract, object_sha256
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.request import RankRequest
from src.memory.manager import MemRecManager

RUN_ID = "cmirank-constrained-decoding-cpu-v7-20261007-hnv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    private_root = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 2
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Exact clean source, CPU reservation and empty CUDA mask required")
    out = args.output_dir.resolve()
    if out != private_root / "runs" / RUN_ID or out.exists():
        raise ValueError("Fresh, scoped CPU smoke output required")
    config, config_sha = load_memory_contract(ROOT, 7)
    review_path = private_root / "runs/cmirank-real-memory-review-v6-20261007-hnv/review.json"
    if file_sha256(review_path) != config["constrained_decoding_delta"]["offline_v6_review_sha256"]:
        raise ValueError("Reviewed terminal v6 receipt required")
    review = json.loads(review_path.read_text())
    run = private_root / "runs" / review["source_run_id"]
    if any(file_sha256(run / name) != digest for name, digest in review["source_artifact_sha256"].items()):
        raise ValueError("Reviewed source artifacts changed")
    versions = {name: importlib.metadata.version(name) for name in ("vllm", "xgrammar", "transformers")}
    if versions != {"vllm": "0.10.2", "xgrammar": "0.1.23", "transformers": "4.55.4"}:
        raise ValueError("Teacher decoder environment changed")
    model_path = private_root / "models/Qwen3-30B-A3B-Instruct-2507-FP8-hnv"
    marker = json.loads((model_path / "download-complete-hnv.json").read_text())
    if marker["model"] != config["model_id"] or marker["revision"] != config["model_revision"]:
        raise ValueError("Wrong teacher tokenizer checkpoint")
    if any(file_sha256(model_path / name) != digest for name, digest in marker["small_file_sha256"].items()):
        raise ValueError("Teacher tokenizer/static config changed")
    # No model constructor or CUDA call; the tokenizer/grammar compiler run on CPU.
    import xgrammar as xgr
    from transformers import AutoTokenizer
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer), max_threads=2, cache_enabled=False)
    physical = read_rows(run / "physical-requests.jsonl")
    traces = {row["episode_id"]: row["details"] for row in read_rows(run / "retrieval-trace.jsonl")}
    compiled, rejected, counts = [], [], {"stage_r": 0, "rerank": 0, "stage_w": 0}
    for index in range(0, len(physical), 2):
        request, response = physical[index:index + 2]
        kwargs = request["kwargs"]
        props, audit = constrain_id_properties(kwargs["messages"], kwargs["response_format"]["json_schema"]["schema"]["properties"])
        details = traces[request["episode_id"]]
        captured = RankRequest.from_dict(details["rank_request"])
        stage = audit["stage"]
        if stage == "stage_r":
            grounding = inspect_retrieval_grounding(details, captured.user_id)
            expected = sorted(set(grounding["visible_neighbor_ids"]) | {f"User-{captured.user_id}"}
                              | {f"Item-{item}" for item in grounding["visible_candidate_ids"]})
            matches = (audit["allowed_source_ids"] == expected
                       and audit["allowed_target_ids"] == [f"User-{captured.user_id}"])
        elif stage == "rerank":
            matches = audit["allowed_candidate_ids"] == sorted(row["id"] for row in captured.candidates)
        else:
            matches = audit["allowed_neighbor_ids"] == sorted({f"{nb['type'].capitalize()}-{nb['id']}"
                                                             for nb in details["pruned_subgraph"]["neighbors"]})
        if not matches:
            raise ValueError("Parsed decoding domain differs from the original structured input")
        schema = response_schema(props)["json_schema"]["schema"]
        compiler.compile_json_schema(schema)
        counts[stage] += 1
        try:
            validate_constrained_output(json.loads(response["content"]), schema)
        except ValueError:
            rejected.append({"attempt": request["physical_attempt"], "stage": stage})
        compiled.append({"episode_id": request["episode_id"], "physical_attempt": request["physical_attempt"],
            "messages_sha256": object_sha256(kwargs["messages"]), "schema_sha256": object_sha256(schema), **audit})
    if counts != {"stage_r": 12, "rerank": 12, "stage_w": 11} or rejected != [{"attempt": 34, "stage": "stage_r"}]:
        raise ValueError("Counterfactual schema acceptance differs from the reviewed v6 failure")
    manager = MemRecManager(None)
    empty_prompt = manager.build_stage_w_prompt(0, {"action": "CLICK", "item_id": 1}, [], [], {}, [])
    empty_props, _ = constrain_id_properties(empty_prompt, manager.get_stage_w_schema())
    compiler.compile_json_schema(response_schema(empty_props)["json_schema"]["schema"])
    out.mkdir()
    schema_path = out / "compiled-input-domains.jsonl"
    schema_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in compiled))
    report = {"status": "CPU_INPUT_ID_GRAMMAR_SMOKE_PASS_GPU_SMOKE_AND_REVIEW_STILL_REQUIRED",
        "source_commit": commit, "config_sha256": config_sha, "slurm_job_id": job,
        "schemas_compiled": 36, "historical_inputs": 35, "stage_counts": counts,
        "expected_historical_response_rejections": rejected, "exact_structured_input_domain_parity": True,
        "new_llm_requests": 0, "gpu_requested": False, "model_weights_loaded": False,
        "training_ready": False, "full_memory_cache_promoted": False,
        "primary_provider_replaced": False, "comparison_role": "secondary_control_not_primary_replacement",
        "elapsed_seconds": time.monotonic() - started, "versions": versions,
        "source_review_sha256": file_sha256(review_path), "decoder_code_sha256": file_sha256(ROOT / "src/cmirank/constrained_decoding.py"),
        "tokenizer_config_sha256": file_sha256(model_path / "tokenizer_config.json"),
        "schema_domains_sha256": file_sha256(schema_path)}
    (out / "report.json").write_text(artifact_json_dumps(report))
    print(artifact_json_dumps(report), end="")


if __name__ == "__main__":
    main()
