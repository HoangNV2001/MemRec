"""Smoke-first self-hosted MovieLens local ranker with frozen model contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence

from src.models.selfhost_transformers import TransformersJSONClient
from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.current_support import (
    LETTERS,
    Journal,
    call_phase,
    event_key,
    model_contract,
    required_keys,
    resource_snapshot,
)
from src.temporal_books.p7_selfhost_local import (
    complete_permutation,
    permutation_repairs,
    ranking_contract,
)


SCHEMA_VERSION = 1


def clean_source_commit() -> str:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("MovieLens GPU run requires a clean tracked worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def stage_r_prompt(event: Mapping[str, Any]) -> tuple[list[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    history = "\n".join(
        f"- {row['title']} | Genres: {row['genres'].replace('|', ', ')} | Rating: {float(row['rating']):.1f}/5"
        for row in event["history"]
    ) or "(No usable prior rating events.)"
    prompt = f"""You are the frozen retrieval stage of a movie recommender. Infer the user's current movie preferences only from rating events strictly before the target time. Rating timestamps are interaction times, not proof of watch order. Do not use or guess the next movie.

Prior rating history:
{history}

Return at most five preference facets supported by this history. Each confidence must be from 0 to 1."""
    schema = {
        "facets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"facet": {"type": "string"}, "confidence": {"type": "number"}},
                "required": ["facet", "confidence"],
                "additionalProperties": False,
            },
        }
    }
    return [{"role": "user", "content": prompt}], schema


def rerank_prompt(
    event: Mapping[str, Any], facets: Sequence[Mapping[str, Any]], item_info: Mapping[str, Mapping[str, str]]
) -> tuple[list[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    facet_text = "\n".join(
        f"- {row['facet']} (confidence {float(row['confidence']):.2f})" for row in facets
    ) or "(No facets recovered.)"
    candidates = "\n".join(
        f"{label}. {item_info[str(movie_id)]['base_memory']}"
        for label, movie_id in zip(LETTERS, event["candidate_item_ids"])
    )
    prompt = f"""You are a frozen listwise movie recommender. Rank all ten candidates by how likely each movie is to be the user's next positive rating, using only the fixed preference facets and native title/genre memory. Do not infer relevance from title length, candidate order, or label.

Preference facets:
{facet_text}

Candidates:
{candidates}

Return JSON with `ranking` only: every label A through J exactly once, best to worst. Do not include a rationale or any other field."""
    return [{"role": "user", "content": prompt}], {
        "ranking": {
            "type": "array",
            "items": {"type": "string", "enum": list(LETTERS)},
            "minItems": len(LETTERS),
            "maxItems": len(LETTERS),
        }
    }


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
        messages, schema = rerank_prompt(event, facets, item_info)
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


def run_events(
    events: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    client: TransformersJSONClient,
    journal: Journal,
) -> None:
    self_host = config["self_host"]
    workers = int(self_host["workers"])
    if workers != 1:
        raise ValueError("MovieLens direct Transformers runner is locked to one worker")
    stages = call_phase(
        jobs=stage_jobs(events),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(self_host["stage_r_max_tokens"]),
    )
    call_phase(
        jobs=rerank_jobs(events, prepared["item_info"], stages),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(self_host["rerank_max_tokens"]),
    )


def configured_smoke_events(prepared: Mapping[str, Any], count: int) -> list[Dict[str, Any]]:
    if not 20 <= count <= 30:
        raise ValueError("MovieLens LLM smoke must contain 20-30 events")
    return sorted(
        prepared["primary_events"], key=lambda event: stable_order(f"ml32m-llm-smoke\0{event_key(event)}")
    )[:count]


def load_locked(config: Mapping[str, Any], config_path: Any) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    study = config["study"]
    prepared_path = project_path(study["prepared"])
    lock_path = project_path(study["method_lock"])
    manifest_path = project_path(study["prepare_manifest"])
    for path in (prepared_path, lock_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"]["sha256"] != sha256_file(config_path):
        raise RuntimeError("config changed after cohort lock")
    if manifest["prepared"]["sha256"] != sha256_file(prepared_path):
        raise RuntimeError("prepared cohort changed after lock")
    if manifest["method_lock"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("method contract changed after lock")
    if lock["primary_labels_used_for_lock"] is not False:
        raise RuntimeError("invalid method lock")
    return prepared, lock, manifest


def dry_contract(config: Mapping[str, Any], config_path: Any) -> Dict[str, Any]:
    prepared, _, manifest = load_locked(config, config_path)
    events = list(prepared.get("development_events", []))
    if not events:
        # Final confirmatory studies have no development outcomes. Prompt/schema
        # validation is outcome-blind, so use a bounded primary prefix without
        # reading gold_item_id.
        events = list(prepared["primary_events"][:20])
    stages = {f"stage_r:{event_key(event)}": {"facets": []} for event in events}
    jobs = [*stage_jobs(events), *rerank_jobs(events, prepared["item_info"], stages)]
    serialized = json.dumps(
        [{"key": key, "kind": kind, "messages": messages, "schema": schema} for key, kind, messages, schema, _ in jobs],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    lowered = serialized.lower()
    if "book recommender" in lowered or "reader history" in lowered or "prior review" in lowered:
        raise RuntimeError("MovieLens prompt leaked the Amazon Books rendering")
    return {
        "decision": "pass",
        "development_events": len(events),
        "jobs": len(jobs),
        "prompt_contract_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "prepared_sha256": manifest["prepared"]["sha256"],
        "artifact_written": False,
        "llm_requests": 0,
        "gpu_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen MovieLens local ranker")
    parser.add_argument("--config", default="configs/temporal_movielens32m/m1_frozen_transfer.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke-only", action="store_true")
    modes.add_argument("--request", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.dry_run:
        print(json.dumps(dry_contract(config, config_path), ensure_ascii=False, indent=2, sort_keys=True))
        return
    study, self_host = config["study"], config["self_host"]
    prepared, _, prepare_manifest = load_locked(config, config_path)
    events = list(prepared["primary_events"])
    if len(events) != int(study["primary_events"]):
        raise RuntimeError("MovieLens primary cohort size mismatch")
    if int(self_host["tensor_parallel_size"]) != 1:
        raise ValueError("MovieLens permits exactly one visible GPU")
    primary = len(events) * 2
    if primary != int(self_host["primary_generations"]):
        raise ValueError("MovieLens primary generation count mismatch")
    if primary + int(self_host["retry_reserve"]) > int(self_host["hard_generation_cap"]):
        raise ValueError("MovieLens generation budget exceeds cap")
    source_commit = clean_source_commit()
    manifest_path = project_path(self_host["manifest"])
    if args.request and manifest_path.exists():
        raise RuntimeError("MovieLens local completion manifest exists; do not overwrite")
    journal = Journal.load(
        project_path(self_host["attempts"]),
        project_path(self_host["calls"]),
        maximum=int(self_host["hard_generation_cap"]),
        primary_limit=primary,
        retry_limit=int(self_host["retry_reserve"]),
    )
    smoke = configured_smoke_events(prepared, int(self_host["smoke_events"]))
    smoke_path = project_path(self_host["smoke_manifest"])
    if args.smoke_only and smoke_path.exists():
        raise RuntimeError("MovieLens smoke manifest exists; do not overwrite")
    if args.request:
        if not smoke_path.exists():
            raise RuntimeError("MovieLens full run blocked: smoke manifest missing")
        recorded = json.loads(smoke_path.read_text(encoding="utf-8"))
        expected = {
            "decision": "pass",
            "config_sha256": sha256_file(config_path),
            "source_commit": source_commit,
            "prepared_sha256": prepare_manifest["prepared"]["sha256"],
            "method_lock_sha256": prepare_manifest["method_lock"]["sha256"],
            "model_contract": model_contract(config),
            "ranking_contract": ranking_contract(config),
            "event_keys": [event_key(event) for event in smoke],
        }
        for key, value in expected.items():
            if recorded.get(key) != value:
                raise RuntimeError(f"MovieLens smoke mismatch: {key}")
        missing = [key for key in required_keys(smoke) if key not in (journal.completed or {})]
        if missing:
            raise RuntimeError(f"MovieLens smoke cache incomplete: {len(missing)} keys")
    snapshot = resource_snapshot()
    client = TransformersJSONClient(
        model_name=str(self_host["model"]),
        revision=str(self_host["revision"]),
        dtype=str(self_host["dtype"]),
        seed=int(self_host["seed"]),
        hard_memory_fraction=float(self_host["hard_memory_fraction"]),
        max_model_len=int(self_host["max_model_len"]),
    )
    selected = smoke if args.smoke_only else events
    if self_host.get("ranking_contract") != "permutation_completion_v2":
        raise RuntimeError("only the sealed permutation-completion contract is supported")
    run_events(selected, prepared, config, client, journal)
    stats = client.get_token_stats()
    missing = [key for key in required_keys(selected) if key not in (journal.completed or {})]
    if missing:
        raise RuntimeError(f"MovieLens local run incomplete: {len(missing)} keys")
    if int(stats["repaired_responses"]) != 0:
        raise RuntimeError("MovieLens structured decoding required parser repair")
    if float(stats["peak_vram_gib"]) > 20.0:
        raise RuntimeError(f"MovieLens exceeded locked 20 GiB envelope: {stats['peak_vram_gib']:.3f}")
    repaired_keys = permutation_repairs(str(self_host["calls"]))
    if len(repaired_keys) > int(self_host["max_permutation_repairs"]):
        raise RuntimeError("MovieLens permutation repair cap exceeded")
    common = {
        "schema_version": SCHEMA_VERSION,
        "run_id": study["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "source_commit": source_commit,
        "prepared_sha256": prepare_manifest["prepared"]["sha256"],
        "method_lock_sha256": prepare_manifest["method_lock"]["sha256"],
        "model_contract": model_contract(config),
        "ranking_contract": ranking_contract(config),
        "resource_before": snapshot,
        "permutation_repairs": {"count": len(repaired_keys), "keys": repaired_keys},
    }
    if args.smoke_only:
        if journal.retry_attempts != 0 or repaired_keys:
            raise RuntimeError("MovieLens smoke used retry or permutation repair")
        payload = {
            **common,
            "decision": "pass",
            "event_keys": [event_key(event) for event in smoke],
            "attempts_after_smoke": journal.attempts,
            "runtime_stats": stats,
        }
        smoke_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"smoke": "pass", "events": len(smoke), "attempts": journal.attempts, "peak_vram_gib": stats["peak_vram_gib"]}))
        return
    payload = {
        **common,
        "decision": "complete",
        "smoke_manifest_sha256": sha256_file(smoke_path),
        "calls_sha256": sha256_file(project_path(self_host["calls"])),
        "attempts_sha256": sha256_file(project_path(self_host["attempts"])),
        "successful_keys": len(journal.completed or {}),
        "journal": {
            "attempts": journal.attempts,
            "primary_attempts": journal.primary_attempts,
            "retry_attempts": journal.retry_attempts,
        },
        "invocation_stats": stats,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"decision": "complete", "successful_keys": payload["successful_keys"], "journal": payload["journal"], "invocation_stats": stats}, indent=2))


if __name__ == "__main__":
    main()
