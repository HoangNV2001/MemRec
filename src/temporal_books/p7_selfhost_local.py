"""Smoke-first self-hosted local baseline for the fresh P7 cohort."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence

from src.models.selfhost_transformers import TransformersJSONClient
from src.temporal_books.common import load_yaml, project_path, sha256_file
from src.temporal_books.current_support import (
    LETTERS,
    Journal,
    call_phase,
    event_key,
    model_contract,
    read_jsonl,
    required_keys,
    rerank_prompt,
    resource_snapshot,
    stage_jobs,
    smoke_events,
)


SCHEMA_VERSION = 1


def complete_permutation(response: Mapping[str, Any], candidate_ids: Sequence[str]) -> list[str]:
    """Keep first label occurrences, then append missing labels in A-J order."""
    if len(candidate_ids) != len(LETTERS) or len(set(candidate_ids)) != len(LETTERS):
        raise ValueError("P7-v2 requires ten distinct candidates")
    labels = response.get("ranking")
    if not isinstance(labels, list) or len(labels) != len(LETTERS) or any(label not in LETTERS for label in labels):
        raise ValueError("P7-v2 ranking must contain ten labels from A-J")
    unique: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    unique.extend(label for label in LETTERS if label not in seen)
    by_label = dict(zip(LETTERS, candidate_ids))
    return [by_label[label] for label in unique]


def strict_rerank_jobs(
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
        schema["ranking"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(LETTERS)},
            "minItems": len(LETTERS),
            "maxItems": len(LETTERS),
        }
        jobs.append(
            (
                f"rerank:local:{key}",
                "rerank",
                messages,
                schema,
                lambda response, candidates=event["candidate_item_ids"]: complete_permutation(response, candidates),
            )
        )
    return jobs


def run_events_v2(
    events: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    client: TransformersJSONClient,
    journal: Journal,
) -> None:
    self_host = config["self_host"]
    workers = int(self_host["workers"])
    if workers != 1:
        raise ValueError("P7-v2 direct Transformers runner is locked to one worker")
    stages = call_phase(
        jobs=stage_jobs(events),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(self_host["stage_r_max_tokens"]),
    )
    call_phase(
        jobs=strict_rerank_jobs(events, prepared["item_info"], stages),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(self_host["rerank_max_tokens"]),
    )


def configured_smoke_events(prepared: Mapping[str, Any], self_host: Mapping[str, Any]) -> list[Dict[str, Any]]:
    events = smoke_events(prepared, int(self_host["smoke_events"]))
    regression_key = self_host.get("regression_event_key")
    if regression_key and all(event_key(event) != regression_key for event in events):
        match = next((event for event in prepared["test_events"] if event_key(event) == regression_key), None)
        if match is None:
            raise ValueError(f"P7-v2 regression event not found: {regression_key}")
        events.append(match)
    if not 20 <= len(events) <= 30:
        raise ValueError("P7-v2 smoke must contain 20-30 events")
    return events


def permutation_repairs(calls_path: str) -> list[str]:
    repaired: list[str] = []
    for row in read_jsonl(project_path(calls_path)):
        if row.get("status") != "success" or row.get("kind") != "rerank":
            continue
        labels = row.get("response", {}).get("ranking")
        if isinstance(labels, list) and (len(set(labels)) != len(LETTERS) or set(labels) != set(LETTERS)):
            repaired.append(str(row["key"]))
    return repaired


def ranking_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    self_host = config["self_host"]
    return {
        "name": self_host.get("ranking_contract", "strict_permutation_v1"),
        "regression_event_key": self_host.get("regression_event_key"),
        "max_permutation_repairs": int(self_host.get("max_permutation_repairs", 0)),
        "repair": "keep first A-J occurrence, append missing labels in A-J order",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7 self-hosted fresh-test local ranker")
    parser.add_argument(
        "--config", default="configs/temporal_amazon_books_2014/transition_ppr_replication_200.yaml"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-only", action="store_true")
    modes.add_argument("--request", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    p7, self_host = config["p7"], config["self_host"]
    prepared_path = project_path(p7["prepared"])
    selection_path = project_path(p7["locked_selection"])
    for path in (prepared_path, selection_path):
        if not path.exists():
            raise FileNotFoundError(path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["calibration"]["test_admission"]["decision"] != "admit":
        raise RuntimeError("P7 LLM run forbidden: calibration did not admit fresh test")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if len(prepared["test_events"]) != int(p7["test_events"]):
        raise RuntimeError("P7 prepared test count mismatch")
    if int(self_host["tensor_parallel_size"]) != 1:
        raise ValueError("P7 permits exactly one visible GPU")
    primary = len(prepared["test_events"]) * 2
    if primary != int(self_host["primary_generations"]):
        raise ValueError("P7 primary generation count mismatch")
    if primary + int(self_host["retry_reserve"]) > int(self_host["hard_generation_cap"]):
        raise ValueError("P7 LLM generation budget exceeds cap")
    manifest_path = project_path(self_host["manifest"])
    if args.request and manifest_path.exists():
        raise RuntimeError("P7 local completion manifest exists; do not overwrite")
    journal = Journal.load(
        project_path(self_host["attempts"]),
        project_path(self_host["calls"]),
        maximum=int(self_host["hard_generation_cap"]),
        primary_limit=primary,
        retry_limit=int(self_host["retry_reserve"]),
    )
    smoke = configured_smoke_events(prepared, self_host)
    smoke_path = project_path(self_host["smoke_manifest"])
    if args.request:
        if not smoke_path.exists():
            raise RuntimeError("P7 full local run blocked: smoke manifest missing")
        recorded = json.loads(smoke_path.read_text(encoding="utf-8"))
        expected = {
            "decision": "pass",
            "config_sha256": sha256_file(config_path),
            "prepared_sha256": sha256_file(prepared_path),
            "locked_selection_sha256": sha256_file(selection_path),
            "model_contract": model_contract(config),
            "ranking_contract": ranking_contract(config),
            "event_keys": [event_key(event) for event in smoke],
        }
        for key, value in expected.items():
            if recorded.get(key) != value:
                raise RuntimeError(f"P7 local smoke mismatch: {key}")
        missing = [key for key in required_keys(smoke) if key not in (journal.completed or {})]
        if missing:
            raise RuntimeError(f"P7 local smoke cache incomplete: {len(missing)} keys")
    snapshot = resource_snapshot()
    client = TransformersJSONClient(
        model_name=str(self_host["model"]),
        revision=str(self_host["revision"]),
        dtype=str(self_host["dtype"]),
        seed=int(self_host["seed"]),
        hard_memory_fraction=float(self_host["hard_memory_fraction"]),
        max_model_len=int(self_host["max_model_len"]),
    )
    selected_events = smoke if args.smoke_only else list(prepared["test_events"])
    if self_host.get("ranking_contract") != "permutation_completion_v2":
        raise RuntimeError("only the sealed P7-v2 ranking contract is supported")
    run_events_v2(selected_events, prepared, config, client, journal)
    stats = client.get_token_stats()
    missing = [key for key in required_keys(selected_events) if key not in (journal.completed or {})]
    if missing:
        raise RuntimeError(f"P7 local run incomplete: {len(missing)} keys")
    if int(stats["repaired_responses"]) != 0:
        raise RuntimeError("P7 structured decoding required parser repair")
    if float(stats["peak_vram_gib"]) > 20.0:
        raise RuntimeError(f"P7 exceeded locked 20 GiB envelope: {stats['peak_vram_gib']:.3f}")
    repaired_keys = permutation_repairs(str(self_host["calls"]))
    if len(repaired_keys) > int(self_host.get("max_permutation_repairs", 0)):
        raise RuntimeError(f"P7 permutation repair cap exceeded: {len(repaired_keys)}")
    common = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p7["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": sha256_file(prepared_path),
        "locked_selection_sha256": sha256_file(selection_path),
        "model_contract": model_contract(config),
        "ranking_contract": ranking_contract(config),
        "resource_before": snapshot,
        "permutation_repairs": {"count": len(repaired_keys), "keys": repaired_keys},
    }
    if args.smoke_only:
        payload = {
            **common,
            "decision": "pass",
            "event_keys": [event_key(event) for event in smoke],
            "attempts_after_smoke": journal.attempts,
            "runtime_stats": stats,
        }
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"smoke": "pass", "events": len(smoke), "attempts": journal.attempts, "peak_vram_gib": stats["peak_vram_gib"]}, ensure_ascii=False))
        return
    payload = {
        **common,
        "decision": "complete",
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
