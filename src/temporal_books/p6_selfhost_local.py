"""Smoke-first self-hosted local baseline for the fresh P6 test cohort."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence

from src.models.selfhost_transformers import TransformersJSONClient
from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.p2_oracle import Journal, call_phase, rerank_prompt, stage_r_prompt, validate_ranking
from src.temporal_books.p3_oracle import event_key


SCHEMA_VERSION = 1


def resource_snapshot() -> Dict[str, Any]:
    expected = os.environ.get("MEMREC_EXPECTED_SLURM_JOB_ID")
    actual = os.environ.get("SLURM_JOB_ID")
    if not expected or actual != expected:
        raise RuntimeError(f"expected Slurm job {expected!r}, running in {actual!r}")
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1:
        raise RuntimeError("P6 local baseline requires exactly one visible GPU")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        "slurm_job_id": actual,
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "cuda_visible_devices": visible,
        "all_gpu_snapshot": [line.strip() for line in output.splitlines() if line.strip()],
    }


def model_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    self_host = config["self_host"]
    keys = (
        "model",
        "revision",
        "dtype",
        "tensor_parallel_size",
        "hard_memory_fraction",
        "max_model_len",
        "seed",
        "temperature",
        "structured_decoding",
        "stage_r_max_tokens",
        "rerank_max_tokens",
    )
    return {key: self_host[key] for key in keys}


def smoke_events(prepared: Mapping[str, Any], count: int) -> list[Dict[str, Any]]:
    if not 20 <= count <= 30:
        raise ValueError("P6 LLM smoke must contain 20-30 events")
    return sorted(
        prepared["test_events"],
        key=lambda event: stable_order(f"p6-llm-smoke\0{event_key(event)}"),
    )[:count]


def stage_jobs(events: Sequence[Mapping[str, Any]]) -> list[Any]:
    jobs = []
    for event in events:
        messages, schema = stage_r_prompt(event)
        jobs.append((f"stage_r:{event_key(event)}", "stage_r", messages, schema, lambda response: response))
    return jobs


def rerank_jobs(
    events: Sequence[Mapping[str, Any]],
    item_info: Mapping[str, Mapping[str, str]],
    stages: Mapping[str, Mapping[str, Any]],
) -> list[Any]:
    jobs = []
    for event in events:
        key = event_key(event)
        facets = stages[f"stage_r:{key}"].get("facets", [])
        memories = {item_id: item_info[item_id]["base_memory"] for item_id in event["candidate_item_ids"]}
        messages, schema = rerank_prompt(event, facets, memories)
        jobs.append(
            (
                f"rerank:local:{key}",
                "rerank",
                messages,
                schema,
                lambda response, candidates=event["candidate_item_ids"]: validate_ranking(response, candidates),
            )
        )
    return jobs


def run_events(
    events: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    client: TransformersJSONClient,
    journal: Journal,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    self_host = config["self_host"]
    workers = int(self_host["workers"])
    if workers != 1:
        raise ValueError("direct Transformers P6 runner is locked to one worker")
    stages = call_phase(
        jobs=stage_jobs(events),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(self_host["stage_r_max_tokens"]),
    )
    reranks = call_phase(
        jobs=rerank_jobs(events, prepared["item_info"], stages),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(self_host["rerank_max_tokens"]),
    )
    return stages, reranks


def required_keys(events: Sequence[Mapping[str, Any]]) -> list[str]:
    return [key for event in events for key in (f"stage_r:{event_key(event)}", f"rerank:local:{event_key(event)}")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P6 self-hosted fresh-test local ranker")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p6_adaptive_graph.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-only", action="store_true")
    modes.add_argument("--request", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    p6, self_host = config["p6"], config["self_host"]
    prepared_path = project_path(p6["prepared"])
    if not prepared_path.exists():
        raise FileNotFoundError(prepared_path)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if len(prepared["test_events"]) != int(p6["test_events"]):
        raise RuntimeError("P6 prepared test count does not match config")
    if int(self_host["tensor_parallel_size"]) != 1:
        raise ValueError("P6 permits exactly one visible GPU")
    primary = len(prepared["test_events"]) * 2
    if primary != int(self_host["primary_generations"]):
        raise ValueError("P6 configured primary generation count is inconsistent")
    if primary + int(self_host["retry_reserve"]) > int(self_host["hard_generation_cap"]):
        raise ValueError("P6 LLM generation budget exceeds hard cap")
    manifest_path = project_path(self_host["manifest"])
    if args.request and manifest_path.exists():
        raise RuntimeError("P6 local completion manifest exists; do not overwrite")
    journal = Journal.load(
        project_path(self_host["attempts"]),
        project_path(self_host["calls"]),
        maximum=int(self_host["hard_generation_cap"]),
        primary_limit=primary,
        retry_limit=int(self_host["retry_reserve"]),
    )
    smoke = smoke_events(prepared, int(self_host["smoke_events"]))
    smoke_path = project_path(self_host["smoke_manifest"])
    if args.request:
        if not smoke_path.exists():
            raise RuntimeError("P6 full local run blocked: smoke manifest missing")
        recorded = json.loads(smoke_path.read_text(encoding="utf-8"))
        expected = {
            "decision": "pass",
            "config_sha256": sha256_file(config_path),
            "prepared_sha256": sha256_file(prepared_path),
            "model_contract": model_contract(config),
            "event_keys": [event_key(event) for event in smoke],
        }
        for key, value in expected.items():
            if recorded.get(key) != value:
                raise RuntimeError(f"P6 local smoke mismatch: {key}")
        missing = [key for key in required_keys(smoke) if key not in (journal.completed or {})]
        if missing:
            raise RuntimeError(f"P6 local smoke cache incomplete: {len(missing)} keys")
    snapshot = resource_snapshot()
    client = TransformersJSONClient(
        model_name=str(self_host["model"]),
        revision=str(self_host["revision"]),
        dtype=str(self_host["dtype"]),
        seed=int(self_host["seed"]),
        hard_memory_fraction=float(self_host["hard_memory_fraction"]),
        max_model_len=int(self_host["max_model_len"]),
    )
    selected = smoke if args.smoke_only else list(prepared["test_events"])
    run_events(selected, prepared, config, client, journal)
    stats = client.get_token_stats()
    missing = [key for key in required_keys(selected) if key not in (journal.completed or {})]
    if missing:
        raise RuntimeError(f"P6 local run incomplete: {len(missing)} keys")
    if int(stats["repaired_responses"]) != 0:
        raise RuntimeError("P6 structured decoding required parser repair")
    if float(stats["peak_vram_gib"]) > 20.0:
        raise RuntimeError(f"P6 local run exceeded locked 20 GiB safety envelope: {stats['peak_vram_gib']:.3f}")
    if args.smoke_only:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": p6["run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "pass",
            "config_sha256": sha256_file(config_path),
            "prepared_sha256": sha256_file(prepared_path),
            "model_contract": model_contract(config),
            "resource_before": snapshot,
            "event_keys": [event_key(event) for event in smoke],
            "attempts_after_smoke": journal.attempts,
            "runtime_stats": stats,
        }
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"smoke": "pass", "events": len(smoke), "attempts": journal.attempts, "peak_vram_gib": stats["peak_vram_gib"]}, ensure_ascii=False))
        return
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p6["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "complete",
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": sha256_file(prepared_path),
        "model_contract": model_contract(config),
        "resource_before": snapshot,
        "smoke_manifest_sha256": sha256_file(smoke_path),
        "calls_sha256": sha256_file(project_path(self_host["calls"])),
        "attempts_sha256": sha256_file(project_path(self_host["attempts"])),
        "successful_keys": len(journal.completed or {}),
        "journal": {"attempts": journal.attempts, "primary_attempts": journal.primary_attempts, "retry_attempts": journal.retry_attempts},
        "invocation_stats": stats,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"decision": "complete", "successful_keys": payload["successful_keys"], "journal": payload["journal"], "invocation_stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
