"""MH2 — bounded-oracle headroom evaluation on the frozen validation cohort.

The runner is deliberately resumable.  Every paid completion is appended to a
JSONL cache immediately; rerunning the same ``run_id`` only schedules missing
keys.  It has three strictly separated phases:

1. generate Stage-R outputs for one-hop, naive q=4 and the 12 oracle bundles;
2. use a *selection pass* to choose the best oracle bundle per user; and
3. make two independent reranker calls for one-hop, naive and the chosen oracle.

The report only uses phase 3.  Selection values are never reported as quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.models.llm_client import LLMClient
from src.models.reranker_llm import LLMReranker
from src.rl.policy import parse_facets, to_chat

from .mh0 import canonical_hash, project_path, safe_relative, sha256


RUN_SCHEMA_VERSION = 1
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "ndcg_at_3", "ndcg_at_5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MH2 bounded-oracle headroom evaluation")
    parser.add_argument("--config", default="configs/multihop/mh0_books.yaml")
    parser.add_argument("--run-id", default=None, help="resumable directory name under results/multihop")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="run only first N locked users (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="show the exact call plan without calling Azure")
    parser.add_argument("--repair-cache", action="store_true", help="deduplicate a stopped run's JSONL cache without API calls")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return substitute_env(config)


def substitute_env(value: Any) -> Any:
    """Resolve the repository's ``${ENV:NAME}`` syntax without importing torch."""
    if isinstance(value, dict):
        return {key: substitute_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute_env(item) for item in value]
    if isinstance(value, str) and value.startswith("${ENV:") and value.endswith("}"):
        return os.getenv(value[6:-1])
    return value


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()


def read_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    for record in iter_jsonl(path):
        key = record.get("key")
        if not key:
            raise ValueError(f"cache record without key: {path}")
        if key in records:
            raise ValueError(f"duplicate cache key {key!r}: {path}")
        records[key] = record
    return records


def repair_cache(path: Path) -> Dict[str, Any]:
    """Keep the first completed record per key after an interrupted duplicate run.

    This does not call the API or alter a record's content.  It writes a sibling
    temporary file and replaces the cache atomically only after the full scan.
    The function must be run only after all MH2 workers have stopped.
    """
    if not path.exists():
        return {"records_before": 0, "records_after": 0, "duplicates_removed": 0}
    seen: set[str] = set()
    kept: List[Dict[str, Any]] = []
    removed = 0
    for record in iter_jsonl(path):
        key = record.get("key")
        if not key:
            raise ValueError(f"cache record without key: {path}")
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(record)
    if removed:
        temporary = path.with_suffix(".repair.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        temporary.replace(path)
    return {"records_before": len(kept) + removed, "records_after": len(kept), "duplicates_removed": removed}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def schema_for(properties: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "name": "response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": dict(properties),
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def metrics_for_ranking(ranking: Sequence[int], gold_item_id: int) -> Dict[str, float]:
    try:
        rank = list(ranking).index(int(gold_item_id)) + 1
    except ValueError:
        rank = len(ranking) + 1

    def hit(k: int) -> float:
        return float(rank <= k)

    def ndcg(k: int) -> float:
        return 1.0 / math.log2(rank + 1) if rank <= k else 0.0

    return {
        "gold_rank": rank,
        "hit_at_1": hit(1),
        "hit_at_3": hit(3),
        "hit_at_5": hit(5),
        "ndcg_at_3": ndcg(3),
        "ndcg_at_5": ndcg(5),
    }


def validate_score_payload(payload: Mapping[str, Any], candidates: Sequence[int]) -> Dict[int, float]:
    scores = payload.get("scores")
    if not isinstance(scores, list):
        raise ValueError("reranker response lacks scores array")
    expected = [int(candidate) for candidate in candidates]
    expected_set = set(expected)
    found: Dict[int, float] = {}
    for row in scores:
        if not isinstance(row, Mapping) or "item_id" not in row or "score" not in row:
            raise ValueError("reranker score entry lacks item_id/score")
        item_id = int(row["item_id"])
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite score for item {item_id}")
        if item_id in found:
            raise ValueError(f"duplicate reranker item_id {item_id}")
        found[item_id] = score
    if set(found) != expected_set:
        missing = sorted(expected_set - set(found))
        extra = sorted(set(found) - expected_set)
        raise ValueError(f"reranker candidate mismatch; missing={missing}, extra={extra}")
    return found


def rank_from_scores(score_map: Mapping[int, float], candidates: Sequence[int]) -> List[int]:
    order = {int(candidate): index for index, candidate in enumerate(candidates)}
    return sorted((int(candidate) for candidate in candidates), key=lambda item: (-score_map[item], order[item]))


def bootstrap_ci(values: Sequence[float], *, samples: int, seed: int) -> List[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def stage_key(user_id: int, stage_id: str) -> str:
    return f"stage_r:{user_id}:{stage_id}"


def selection_key(user_id: int, stage_id: str) -> str:
    return f"selection_rerank:{user_id}:{stage_id}"


def report_key(user_id: int, arm: str, repeat: int) -> str:
    return f"report_rerank:{user_id}:{arm}:r{repeat}"


def stage_id_for_oracle(bundle: Mapping[str, Any]) -> str:
    return f"oracle_q{int(bundle['quota_requested'])}_v{int(bundle['variant'])}"


def eligible_inputs(
    controls_path: Path, bundles_path: Path, *, naive_quota: int, limit: int | None
) -> List[Dict[str, Any]]:
    controls = {int(row["user_id"]): row for row in iter_jsonl(controls_path)}
    users = []
    for bundle_record in iter_jsonl(bundles_path):
        if not bundle_record["eligibility"]["eligible_all_arms"]:
            continue
        user_id = int(bundle_record["user_id"])
        control = controls.get(user_id)
        if control is None:
            raise ValueError(f"MH1 bundle user {user_id} absent from MH0 val control")
        if bundle_record["evaluation_contract"]["candidate_order_sha256"] != control["ranking_context"]["candidate_order_sha256"]:
            raise ValueError(f"candidate order hash mismatch for user {user_id}")
        naive = bundle_record["naive_two_hop"].get(str(naive_quota))
        if naive is None:
            raise ValueError(f"user {user_id} lacks naive quota {naive_quota}")
        oracle = list(bundle_record["oracle_two_hop"])
        if len(oracle) != 12:
            raise ValueError(f"user {user_id}: expected 12 oracle bundles, got {len(oracle)}")
        users.append({"user_id": user_id, "control": control, "bundle": bundle_record, "naive": naive, "oracle": oracle})
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        users = users[:limit]
    if not users:
        raise ValueError("no eligible MH2 users")
    return users


def stage_definitions(user: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    control = user["control"]
    stages = {
        "one_hop": {
            "stage_r": control["stage_r_context"],
            "arm": "one_hop",
            "remote_actual": 0,
            "remote_shortfall": 0,
        },
        f"naive_q{int(user['naive']['quota_requested'])}": {
            "stage_r": user["naive"]["stage_r"],
            "arm": "naive_two_hop",
            "remote_actual": int(user["naive"]["remote_actual"]),
            "remote_shortfall": int(user["naive"]["remote_shortfall"]),
        },
    }
    for bundle in user["oracle"]:
        identifier = stage_id_for_oracle(bundle)
        if identifier in stages:
            raise ValueError(f"duplicate stage identifier: {identifier}")
        stages[identifier] = {
            "stage_r": bundle["stage_r"],
            "arm": "oracle_two_hop",
            "remote_actual": int(bundle["remote_actual"]),
            "remote_shortfall": int(bundle["remote_shortfall"]),
            "quota_requested": int(bundle["quota_requested"]),
            "variant": int(bundle["variant"]),
        }
    if len(stages) != 14:
        raise AssertionError(f"expected one-hop + naive + 12 oracle stages, got {len(stages)}")
    return stages


def stage_r_properties() -> Dict[str, Any]:
    return {
        "facets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "facet": {"type": "string"},
                    "confidence": {"type": "number"},
                    "supporting_neighbors": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["facet", "confidence", "supporting_neighbors"],
                "additionalProperties": False,
            },
        },
        "support_edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "w": {"type": "number"},
                },
                "required": ["from", "to", "w"],
                "additionalProperties": False,
            },
        },
    }


def call_stage_r(client: LLMClient, user_id: int, stage_id: str, definition: Mapping[str, Any], *, n_facets: int, max_tokens: int) -> Dict[str, Any]:
    context = definition["stage_r"]
    prompt = str(context["prompt"])
    if canonical_hash(prompt) != context["prompt_sha256"]:
        raise ValueError(f"Stage-R prompt hash mismatch for user {user_id}/{stage_id}")
    started = time.monotonic()
    raw = client.generate(
        messages=to_chat(prompt),
        temperature=0.0,
        max_tokens=max_tokens,
        json_schema=schema_for(stage_r_properties()),
    )
    parsed = parse_facets(
        raw,
        valid_node_ids=list(context["selected_node_ids"]),
        max_facets=n_facets,
    )
    return {
        "kind": "stage_r",
        "key": stage_key(user_id, stage_id),
        "user_id": user_id,
        "stage_id": stage_id,
        "arm": definition["arm"],
        "prompt_sha256": context["prompt_sha256"],
        "remote_actual": definition["remote_actual"],
        "remote_shortfall": definition["remote_shortfall"],
        "raw": raw,
        "facets": parsed.facets,
        "parse": {"is_valid": parsed.is_valid, "error": parsed.error, "n_dropped": parsed.n_dropped, "truncated": parsed.truncated},
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def rerank_messages(reranker: LLMReranker, control: Mapping[str, Any], facets: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, str]], List[int]]:
    ranking = control["ranking_context"]
    candidates = [
        {"id": int(item_id), "title": ranking["candidate_titles"].get(str(item_id), f"Item-{item_id}"), "tags": []}
        for item_id in ranking["candidates"]
    ]
    item_mems = {int(item_id): memory for item_id, memory in ranking.get("candidate_memories", {}).items()}
    messages = reranker.build_rerank_prompt(
        user_id=int(control["user_id"]),
        facets=list(facets),
        candidates=candidates,
        item_mems=item_mems,
        instruction=ranking.get("instruction"),
        vanilla_mode=False,
    )
    return messages, [int(item_id) for item_id in ranking["candidates"]]


def call_reranker(
    client: LLMClient,
    reranker: LLMReranker,
    *,
    key: str,
    kind: str,
    user_id: int,
    stage_id: str,
    control: Mapping[str, Any],
    facets: Sequence[Mapping[str, Any]],
    max_tokens: int,
    repeat: int | None = None,
) -> Dict[str, Any]:
    messages, candidates = rerank_messages(reranker, control, facets)
    started = time.monotonic()
    raw = client.generate(
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        json_schema=schema_for(reranker.get_rerank_schema()),
    )
    payload = json.loads(raw)
    score_map = validate_score_payload(payload, candidates)
    ranking = rank_from_scores(score_map, candidates)
    result = {
        "kind": kind,
        "key": key,
        "user_id": user_id,
        "stage_id": stage_id,
        "rerank_prompt_sha256": canonical_hash(messages),
        "raw": raw,
        "scores": {str(item_id): score_map[item_id] for item_id in candidates},
        "ranking": ranking,
        "metrics": metrics_for_ranking(ranking, int(control["ranking_context"]["gold_item_id"])),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if repeat is not None:
        result["repeat"] = repeat
    return result


def error_record(*, kind: str, key: str, user_id: int, stage_id: str, exc: Exception) -> Dict[str, Any]:
    return {
        "kind": kind,
        "key": key,
        "user_id": user_id,
        "stage_id": stage_id,
        "error": f"{type(exc).__name__}: {exc}",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def execute_missing(jobs: Sequence[Any], fn, cache_path: Path, cache: MutableMapping[str, Dict[str, Any]], workers: int, label: str) -> int:
    if not jobs:
        print(f"  {label}: cache complete")
        return 0
    print(f"  {label}: {len(jobs)} remote calls with {workers} workers")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            append_jsonl(cache_path, record)
            cache[record["key"]] = record
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"    {label}: {done}/{len(jobs)}")
    return done


def is_success(record: Mapping[str, Any] | None) -> bool:
    return record is not None and not record.get("error")


def choose_oracle(user_id: int, oracle_ids: Sequence[str], cache: Mapping[str, Mapping[str, Any]]) -> str | None:
    candidates = []
    for identifier in oracle_ids:
        record = cache.get(selection_key(user_id, identifier))
        if not is_success(record):
            return None
        metrics = record["metrics"]
        # NDCG is the pre-registered oracle objective. Ties are deterministic and
        # deliberately do not inspect score margins or report-pass outcomes.
        candidates.append((float(metrics["ndcg_at_5"]), identifier, record))
    return sorted(candidates, key=lambda value: (-value[0], value[1]))[0][1]


def report_metrics(users: Sequence[Mapping[str, Any]], cache: Mapping[str, Mapping[str, Any]], *, report_repeats: int, bootstrap_samples: int, seed: int) -> Dict[str, Any]:
    analysed = []
    dropped: Dict[str, str] = {}
    for user in users:
        user_id = int(user["user_id"])
        stages = stage_definitions(user)
        oracle_ids = [stage_id_for_oracle(bundle) for bundle in user["oracle"]]
        chosen = choose_oracle(user_id, oracle_ids, cache)
        if chosen is None:
            dropped[str(user_id)] = "missing/failed oracle selection"
            continue
        arm_stages = {"one_hop": "one_hop", "naive_two_hop": f"naive_q{int(user['naive']['quota_requested'])}", "oracle_two_hop": chosen}
        per_arm: Dict[str, Dict[str, float]] = {}
        valid = True
        for arm, stage_id in arm_stages.items():
            repeats = [cache.get(report_key(user_id, arm, repeat)) for repeat in range(1, report_repeats + 1)]
            if not all(is_success(record) for record in repeats):
                dropped[str(user_id)] = f"missing/failed report for {arm}"
                valid = False
                break
            per_arm[arm] = {
                metric: sum(float(record["metrics"][metric]) for record in repeats) / report_repeats
                for metric in _METRICS
            }
        if not valid:
            continue
        chosen_def = stages[chosen]
        analysed.append(
            {
                "user_id": user_id,
                "chosen_oracle_stage": chosen,
                "chosen_oracle_quota": int(chosen_def["quota_requested"]),
                "chosen_oracle_variant": int(chosen_def["variant"]),
                "oracle_remote_actual": int(chosen_def["remote_actual"]),
                "oracle_remote_shortfall": int(chosen_def["remote_shortfall"]),
                "naive_remote_actual": int(user["naive"]["remote_actual"]),
                "naive_remote_shortfall": int(user["naive"]["remote_shortfall"]),
                "metrics": per_arm,
            }
        )

    arms = ("one_hop", "naive_two_hop", "oracle_two_hop")
    aggregates: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        aggregates[arm] = {metric: (sum(row["metrics"][arm][metric] for row in analysed) / len(analysed) if analysed else float("nan")) for metric in _METRICS}
    comparisons = {}
    for arm in ("naive_two_hop", "oracle_two_hop"):
        deltas = [row["metrics"][arm]["ndcg_at_5"] - row["metrics"]["one_hop"]["ndcg_at_5"] for row in analysed]
        comparisons[arm] = {
            "delta_ndcg_at_5": sum(deltas) / len(deltas) if deltas else float("nan"),
            "bootstrap_95_ci": bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed + (1 if arm == "naive_two_hop" else 2)),
        }
    quota_distribution: Dict[str, int] = {}
    for row in analysed:
        quota = str(row["chosen_oracle_quota"])
        quota_distribution[quota] = quota_distribution.get(quota, 0) + 1
    return {
        "initial_locked_users": len(users),
        "analysed_users": len(analysed),
        "dropped_users": dropped,
        "arms": aggregates,
        "comparisons": comparisons,
        "chosen_oracle_quota_distribution": quota_distribution,
        "mean_naive_remote_actual": sum(row["naive_remote_actual"] for row in analysed) / len(analysed) if analysed else float("nan"),
        "mean_oracle_remote_actual": sum(row["oracle_remote_actual"] for row in analysed) / len(analysed) if analysed else float("nan"),
        "per_user": analysed,
    }


def protocol_descriptor(config_path: Path, manifest_path: Path, controls_path: Path, bundles_path: Path, *, naive_quota: int, report_repeats: int) -> Dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "config": {"path": safe_relative(config_path), "sha256": sha256(config_path)},
        "mh1_manifest": {"path": safe_relative(manifest_path), "sha256": sha256(manifest_path)},
        "mh0_control_val": {"path": safe_relative(controls_path), "sha256": sha256(controls_path)},
        "mh1_bundles_val": {"path": safe_relative(bundles_path), "sha256": sha256(bundles_path)},
        "naive_report_quota": naive_quota,
        "report_repeats": report_repeats,
    }


def write_or_verify_run_manifest(path: Path, descriptor: Mapping[str, Any], run_id: str) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol") != descriptor:
            raise ValueError("existing run manifest has a different frozen protocol; choose a new --run-id")
        return
    payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "protocol": descriptor,
        "note": "Resumable MH2 cache. Selection outcomes are not report outcomes.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    mh_cfg = config.get("multihop", {})
    mh2_cfg = mh_cfg.get("mh2", {})
    run_id = args.run_id or str(mh2_cfg["run_id"])
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("--run-id may contain only letters, digits, _ and -")
    workers = int(args.workers or mh2_cfg["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    naive_quota = int(mh2_cfg["naive_report_quota"])
    report_repeats = int(mh2_cfg["report_repeats"])
    if report_repeats < 2:
        raise ValueError("MH2 requires at least two independent report repeats")
    stage_r_max_tokens = int(mh2_cfg["stage_r_max_tokens"])
    rerank_max_tokens = int(mh2_cfg["rerank_max_tokens"])
    bootstrap_samples = int(mh2_cfg["bootstrap_samples"])
    seed = int(config.get("seed", 42))
    n_facets = int(config.get("memrec", {}).get("n_facets", 7))
    output_root = project_path(mh2_cfg.get("output_root", "results/multihop")) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "raw_completions.jsonl"
    result_path = output_root / "metrics.json"
    run_manifest_path = output_root / "run_manifest.json"
    repair_path = output_root / "cache_repair.json"
    mh1_manifest_path = project_path(mh_cfg.get("mh1_manifest", "data/multihop/mh1_manifest.json"))
    controls_path = project_path(mh_cfg.get("mh0_control_val", "data/multihop/mh0_control_val.jsonl"))
    bundles_path = project_path(mh_cfg.get("mh1_bundles_val", "data/multihop/mh1_bundles_val.jsonl"))
    required = [config_path, mh1_manifest_path, controls_path, bundles_path]
    missing = [safe_relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("MH2 required input(s) missing: " + ", ".join(missing))
    descriptor = protocol_descriptor(
        config_path, mh1_manifest_path, controls_path, bundles_path,
        naive_quota=naive_quota, report_repeats=report_repeats,
    )
    write_or_verify_run_manifest(run_manifest_path, descriptor, run_id)
    if args.repair_cache:
        repair = repair_cache(raw_path)
        repair.update({"repaired_at_utc": datetime.now(timezone.utc).isoformat(), "raw_completions": safe_relative(raw_path)})
        repair_path.write_text(json.dumps(repair, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"MH2 cache repair: {repair['records_before']} -> {repair['records_after']} records; removed {repair['duplicates_removed']} duplicate keys")
        return
    users = eligible_inputs(controls_path, bundles_path, naive_quota=naive_quota, limit=args.limit)
    stages_by_user = {int(user["user_id"]): stage_definitions(user) for user in users}
    expected_stage = len(users) * 14
    expected_selection = len(users) * 12
    expected_report = len(users) * 3 * report_repeats
    print(f"MH2 run: {run_id}")
    print(f"  locked users in this invocation: {len(users)}")
    print(f"  planned calls: Stage-R={expected_stage}, selection={expected_selection}, report={expected_report}, total={expected_stage + expected_selection + expected_report}")
    if args.dry_run:
        return

    provider = config.get("provider", {})
    client_kwargs = {
        "api_endpoint": provider.get("endpoint"),
        "api_key": provider.get("api_key"),
        "api_version": provider.get("api_version"),
        "model": provider.get("model", config.get("llm_model")),
        "provider_name": provider.get("name", "azure_openai"),
    }
    stage_client = LLMClient(**client_kwargs)
    rerank_client = LLMClient(**client_kwargs)
    reranker = LLMReranker(rerank_client)
    cache = read_cache(raw_path)
    t0 = time.monotonic()

    stage_jobs = []
    for user in users:
        user_id = int(user["user_id"])
        for stage_id, definition in stages_by_user[user_id].items():
            key = stage_key(user_id, stage_id)
            if key not in cache:
                stage_jobs.append((user_id, stage_id, definition))

    def run_stage(job):
        user_id, stage_id, definition = job
        key = stage_key(user_id, stage_id)
        try:
            return call_stage_r(stage_client, user_id, stage_id, definition, n_facets=n_facets, max_tokens=stage_r_max_tokens)
        except Exception as exc:  # noqa: BLE001
            return error_record(kind="stage_r", key=key, user_id=user_id, stage_id=stage_id, exc=exc)

    execute_missing(stage_jobs, run_stage, raw_path, cache, workers, "Stage-R")

    selection_jobs = []
    users_by_id = {int(user["user_id"]): user for user in users}
    for user in users:
        user_id = int(user["user_id"])
        for bundle in user["oracle"]:
            stage_id = stage_id_for_oracle(bundle)
            key = selection_key(user_id, stage_id)
            stage = cache.get(stage_key(user_id, stage_id))
            if key not in cache and is_success(stage):
                selection_jobs.append((user_id, stage_id))

    def run_selection(job):
        user_id, stage_id = job
        key = selection_key(user_id, stage_id)
        try:
            stage = cache[stage_key(user_id, stage_id)]
            return call_reranker(
                rerank_client, reranker, key=key, kind="selection_rerank", user_id=user_id,
                stage_id=stage_id, control=users_by_id[user_id]["control"], facets=stage["facets"], max_tokens=rerank_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            return error_record(kind="selection_rerank", key=key, user_id=user_id, stage_id=stage_id, exc=exc)

    execute_missing(selection_jobs, run_selection, raw_path, cache, workers, "oracle selection")

    report_jobs = []
    for user in users:
        user_id = int(user["user_id"])
        oracle_ids = [stage_id_for_oracle(bundle) for bundle in user["oracle"]]
        selected = choose_oracle(user_id, oracle_ids, cache)
        if selected is None:
            continue
        report_arms = {"one_hop": "one_hop", "naive_two_hop": f"naive_q{naive_quota}", "oracle_two_hop": selected}
        for arm, stage_id in report_arms.items():
            stage = cache.get(stage_key(user_id, stage_id))
            if not is_success(stage):
                continue
            for repeat in range(1, report_repeats + 1):
                key = report_key(user_id, arm, repeat)
                if key not in cache:
                    report_jobs.append((user_id, arm, stage_id, repeat))

    def run_report(job):
        user_id, arm, stage_id, repeat = job
        key = report_key(user_id, arm, repeat)
        try:
            stage = cache[stage_key(user_id, stage_id)]
            return call_reranker(
                rerank_client, reranker, key=key, kind="report_rerank", user_id=user_id,
                stage_id=stage_id, control=users_by_id[user_id]["control"], facets=stage["facets"], max_tokens=rerank_max_tokens, repeat=repeat,
            )
        except Exception as exc:  # noqa: BLE001
            return error_record(kind="report_rerank", key=key, user_id=user_id, stage_id=stage_id, exc=exc)

    execute_missing(report_jobs, run_report, raw_path, cache, workers, "independent report")
    summary = report_metrics(users, cache, report_repeats=report_repeats, bootstrap_samples=bootstrap_samples, seed=seed)
    wall = time.monotonic() - t0
    summary["run"] = {
        "run_id": run_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "protocol": descriptor,
        "raw_completions": {"path": safe_relative(raw_path), "sha256": sha256(raw_path), "records": len(cache)},
        "wall_seconds_this_invocation": wall,
        "model": stage_client.model,
        "azure_deployment": stage_client.request_model,
        "api_version": stage_client.api_version,
        "token_stats": {"stage_r": stage_client.get_token_stats(), "reranker": rerank_client.get_token_stats()},
        "api_cost_usd": None,
        "api_cost_note": "Azure deployment pricing was not supplied; token counts are recorded instead.",
    }
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"MH2 complete for this invocation: {summary['analysed_users']}/{summary['initial_locked_users']} users analysed")
    for arm, stats in summary["arms"].items():
        print(f"  {arm}: NDCG@5={stats['ndcg_at_5']:.4f}")
    for arm, comparison in summary["comparisons"].items():
        low, high = comparison["bootstrap_95_ci"]
        print(f"  delta {arm} vs one_hop: {comparison['delta_ndcg_at_5']:+.4f} [{low:+.4f}, {high:+.4f}]")
    print(f"  wrote {safe_relative(result_path)}")


if __name__ == "__main__":
    main()
