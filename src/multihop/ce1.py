"""CE1 — independent reranking pilot for candidate-conditioned 1-hop evidence.

CE1 deliberately reuses the frozen one-hop Stage-R completion from MH2.  The
only experimental difference is optional ranking-time evidence in the reranker
prompt:

* ``baseline_one_hop``: existing Stage-R facets only;
* ``request_one_hop``: the same facets plus one shared, request-matched 1-hop
  evidence bundle; and
* ``candidate_one_hop``: the same facets plus one equal-budget evidence slot
  for each candidate.

The CE0 selector is allowed to read the current candidate text, but never gold
or outcomes.  This runner reports only two independent rerank repeats; it does
not select an oracle or tune a setting after results are seen.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from .mh0 import PROJECT_ROOT, canonical_hash, git_sha, project_path, safe_relative, sha256
from . import mh2

from src.models.llm_client import LLMClient
from src.models.reranker_llm import LLMReranker


RUN_SCHEMA_VERSION = 1
ARMS = ("baseline_one_hop", "request_one_hop", "candidate_one_hop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CE1 candidate-conditioned evidence pilot")
    parser.add_argument("--config", default="configs/multihop/mh0_books.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="recompute metrics from a completed cache without API calls")
    parser.add_argument("--repair-cache", action="store_true", help="deduplicate a stopped run without API calls")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="after a stopped run, archive failed records and retry only their keys",
    )
    return parser.parse_args()


def report_key(user_id: int, arm: str, repeat: int) -> str:
    return f"ce1_report:{user_id}:{arm}:r{repeat}"


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def archive_error_records(raw_path: Path, archive_path: Path) -> Dict[str, int]:
    """Make failed keys schedulable again while retaining an append-only audit.

    This must only be called when no CE1 worker is running. Successful records
    are copied into an atomic replacement cache; failed records are written to
    a retry ledger before removal, so retrying cannot create duplicate keys.
    """
    if not raw_path.exists():
        return {"records_before": 0, "successful_kept": 0, "errors_archived": 0}
    records = list(mh2.iter_jsonl(raw_path))
    seen = set()
    for record in records:
        key = record.get("key")
        if not key or key in seen:
            raise ValueError("CE1 cache must have unique non-empty keys before retrying errors")
        seen.add(key)
    failed = [record for record in records if record.get("error")]
    kept = [record for record in records if not record.get("error")]
    if not failed:
        return {"records_before": len(records), "successful_kept": len(kept), "errors_archived": 0}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("a", encoding="utf-8") as archive:
        for record in failed:
            archived = dict(record)
            archived["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
            archive.write(json.dumps(archived, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary = raw_path.with_suffix(".retry.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(raw_path)
    return {"records_before": len(records), "successful_kept": len(kept), "errors_archived": len(failed)}


def read_inputs(
    controls_path: Path, evidence_path: Path, source_mh2_raw_path: Path
) -> List[Dict[str, Any]]:
    controls = {int(row["user_id"]): row for row in mh2.iter_jsonl(controls_path)}
    source_cache = mh2.read_cache(source_mh2_raw_path)
    users: List[Dict[str, Any]] = []
    for evidence in mh2.iter_jsonl(evidence_path):
        user_id = int(evidence["user_id"])
        control = controls.get(user_id)
        if control is None:
            raise ValueError(f"CE0 user {user_id} missing MH0 control")
        contract = evidence["evaluation_contract"]
        if contract["candidate_order_sha256"] != control["ranking_context"]["candidate_order_sha256"]:
            raise ValueError(f"CE0 candidate order mismatch for user {user_id}")
        if contract.get("gold_item_id_used_by_selector") is not False:
            raise ValueError(f"CE0 evidence contract is not gold-blind for user {user_id}")
        stage = source_cache.get(mh2.stage_key(user_id, "one_hop"))
        if not mh2.is_success(stage):
            raise ValueError(f"MH2 one-hop Stage-R output missing/failed for CE0 user {user_id}")
        if stage["prompt_sha256"] != control["stage_r_context"]["prompt_sha256"]:
            raise ValueError(f"MH2 Stage-R prompt mismatch for user {user_id}")
        if evidence["source_one_hop"]["node_ids"] != control["stage_r_context"]["selected_node_ids"]:
            raise ValueError(f"CE0 source node mismatch for user {user_id}")
        candidates = [int(candidate) for candidate in control["ranking_context"]["candidates"]]
        candidate_rows = evidence["candidate_evidence"]["by_candidate"]
        if set(candidate_rows) != {str(candidate) for candidate in candidates}:
            raise ValueError(f"CE0 candidate evidence keys mismatch for user {user_id}")
        users.append({"user_id": user_id, "control": control, "evidence": evidence, "stage": stage})
    if not users:
        raise ValueError("CE1 has no materialized evidence users")
    return sorted(users, key=lambda row: row["user_id"])


def build_messages(reranker: LLMReranker, user: Mapping[str, Any], arm: str) -> tuple[List[Dict[str, str]], List[int]]:
    control = user["control"]
    ranking = control["ranking_context"]
    candidates = [int(candidate) for candidate in ranking["candidates"]]
    candidate_rows = [
        {"id": candidate, "title": ranking["candidate_titles"].get(str(candidate), f"Item-{candidate}"), "tags": []}
        for candidate in candidates
    ]
    memories = {int(item_id): memory for item_id, memory in ranking.get("candidate_memories", {}).items()}
    evidence = user["evidence"]
    kwargs: Dict[str, Any] = {}
    if arm == "request_one_hop":
        kwargs["shared_evidence"] = evidence["request_evidence"]["rows"]
    elif arm == "candidate_one_hop":
        kwargs["candidate_evidence"] = {
            int(candidate): rows
            for candidate, rows in evidence["candidate_evidence"]["by_candidate"].items()
        }
    elif arm != "baseline_one_hop":
        raise ValueError(f"unknown CE1 arm: {arm}")
    messages = reranker.build_rerank_prompt(
        user_id=int(control["user_id"]),
        facets=user["stage"]["facets"],
        candidates=candidate_rows,
        item_mems=memories,
        instruction=ranking.get("instruction"),
        vanilla_mode=False,
        **kwargs,
    )
    return messages, candidates


def call_reranker(
    client: LLMClient,
    reranker: LLMReranker,
    *,
    user: Mapping[str, Any],
    arm: str,
    repeat: int,
    max_tokens: int,
) -> Dict[str, Any]:
    messages, candidates = build_messages(reranker, user, arm)
    started = time.monotonic()
    raw = client.generate(
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        json_schema=mh2.schema_for(reranker.get_rerank_schema()),
    )
    payload = json.loads(raw)
    score_map = mh2.validate_score_payload(payload, candidates)
    ranking = mh2.rank_from_scores(score_map, candidates)
    evidence = user["evidence"]
    evidence_tokens = 0
    if arm == "request_one_hop":
        evidence_tokens = int(evidence["request_evidence"]["estimated_tokens"])
    elif arm == "candidate_one_hop":
        evidence_tokens = int(evidence["candidate_evidence"]["estimated_tokens"])
    return {
        "kind": "ce1_report_rerank",
        "key": report_key(int(user["user_id"]), arm, repeat),
        "user_id": int(user["user_id"]),
        "arm": arm,
        "repeat": repeat,
        "rerank_prompt_sha256": canonical_hash(messages),
        "evidence_estimated_tokens": evidence_tokens,
        "raw": raw,
        "scores": {str(candidate): score_map[candidate] for candidate in candidates},
        "ranking": ranking,
        "metrics": mh2.metrics_for_ranking(ranking, int(user["control"]["ranking_context"]["gold_item_id"])),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def error_record(*, user_id: int, arm: str, repeat: int, exc: Exception) -> Dict[str, Any]:
    return {
        "kind": "ce1_report_rerank",
        "key": report_key(user_id, arm, repeat),
        "user_id": user_id,
        "arm": arm,
        "repeat": repeat,
        "error": f"{type(exc).__name__}: {exc}",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def execute_missing(
    jobs: Sequence[tuple[Mapping[str, Any], str, int]],
    fn,
    cache_path: Path,
    cache: MutableMapping[str, Dict[str, Any]],
    workers: int,
) -> int:
    if not jobs:
        print("  report rerank: cache complete")
        return 0
    print(f"  report rerank: {len(jobs)} remote calls with {workers} workers")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            mh2.append_jsonl(cache_path, record)
            cache[record["key"]] = record
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                print(f"    report rerank: {completed}/{len(jobs)}")
    return completed


def report_metrics(
    users: Sequence[Mapping[str, Any]], cache: Mapping[str, Mapping[str, Any]], *, repeats: int, bootstrap_samples: int, seed: int
) -> Dict[str, Any]:
    analysed: List[Dict[str, Any]] = []
    dropped: Dict[str, str] = {}
    for user in users:
        user_id = int(user["user_id"])
        per_arm: Dict[str, Dict[str, float]] = {}
        for arm in ARMS:
            records = [cache.get(report_key(user_id, arm, repeat)) for repeat in range(1, repeats + 1)]
            if not all(mh2.is_success(record) for record in records):
                dropped[str(user_id)] = f"missing/failed {arm} report"
                break
            per_arm[arm] = {
                metric: sum(float(record["metrics"][metric]) for record in records) / repeats
                for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "ndcg_at_3", "ndcg_at_5")
            }
        else:
            analysed.append({"user_id": user_id, "metrics": per_arm})
    aggregates = {
        arm: {
            metric: sum(row["metrics"][arm][metric] for row in analysed) / len(analysed) if analysed else float("nan")
            for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "ndcg_at_3", "ndcg_at_5")
        }
        for arm in ARMS
    }
    comparisons = {}
    for offset, arm in enumerate(ARMS[1:], start=1):
        deltas = [row["metrics"][arm]["ndcg_at_5"] - row["metrics"]["baseline_one_hop"]["ndcg_at_5"] for row in analysed]
        comparisons[arm] = {
            "delta_ndcg_at_5": sum(deltas) / len(deltas) if deltas else float("nan"),
            "bootstrap_95_ci": mh2.bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed + offset),
        }
    candidate_vs_request = [
        row["metrics"]["candidate_one_hop"]["ndcg_at_5"] - row["metrics"]["request_one_hop"]["ndcg_at_5"]
        for row in analysed
    ]
    comparisons["candidate_vs_request_one_hop"] = {
        "delta_ndcg_at_5": sum(candidate_vs_request) / len(candidate_vs_request) if candidate_vs_request else float("nan"),
        "bootstrap_95_ci": mh2.bootstrap_ci(candidate_vs_request, samples=bootstrap_samples, seed=seed + 3),
    }
    return {
        "initial_users": len(users),
        "analysed_users": len(analysed),
        "dropped_users": dropped,
        "arms": aggregates,
        "comparisons": comparisons,
        "per_user": analysed,
    }


def manifest_descriptor(
    config_path: Path, controls_path: Path, evidence_path: Path, source_mh2_raw_path: Path, *, repeats: int
) -> Dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "config": {"path": safe_relative(config_path), "sha256": sha256(config_path)},
        "mh0_control_val": {"path": safe_relative(controls_path), "sha256": sha256(controls_path)},
        "ce0_evidence": {"path": safe_relative(evidence_path), "sha256": sha256(evidence_path)},
        "source_mh2_raw": {"path": safe_relative(source_mh2_raw_path), "sha256": sha256(source_mh2_raw_path)},
        "report_repeats": repeats,
        "arms": list(ARMS),
        "report_only": "two independent reranks per arm; no oracle selection",
    }


def write_or_verify_manifest(path: Path, descriptor: Mapping[str, Any], run_id: str) -> None:
    if path.exists():
        existing = read_json(path)
        if existing.get("protocol") != descriptor:
            raise ValueError("existing CE1 run manifest has a different frozen protocol; choose a new --run-id")
        return
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_sha": git_sha(),
                "protocol": descriptor,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = mh2.load_yaml(config_path)
    section = config.get("candidate_evidence", {})
    ce1 = section.get("ce1", {})
    run_id = args.run_id or str(ce1["run_id"])
    if not mh2._RUN_ID_RE.fullmatch(run_id):
        raise ValueError("--run-id may contain only letters, digits, _ and -")
    workers = int(args.workers or ce1["workers"])
    repeats = int(ce1["report_repeats"])
    max_tokens = int(ce1["rerank_max_tokens"])
    bootstrap_samples = int(ce1["bootstrap_samples"])
    if workers < 1 or repeats < 2:
        raise ValueError("CE1 needs positive workers and at least two independent report repeats")
    controls_path = project_path(section["mh0_control_val"])
    evidence_path = project_path(section["artifact"])
    source_mh2_raw_path = project_path(section["source_mh2_raw"])
    required = [config_path, controls_path, evidence_path, source_mh2_raw_path]
    missing = [safe_relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("CE1 required input(s) missing: " + ", ".join(missing))
    users = read_inputs(controls_path, evidence_path, source_mh2_raw_path)
    descriptor = manifest_descriptor(config_path, controls_path, evidence_path, source_mh2_raw_path, repeats=repeats)
    output_root = project_path(config.get("multihop", {}).get("mh2", {}).get("output_root", "results/multihop")) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    raw_path = output_root / "raw_completions.jsonl"
    metrics_path = output_root / "metrics.json"
    repair_path = output_root / "cache_repair.json"
    retry_archive_path = output_root / "retry_errors.jsonl"
    write_or_verify_manifest(manifest_path, descriptor, run_id)
    if args.repair_cache:
        repair = mh2.repair_cache(raw_path)
        repair.update({"repaired_at_utc": datetime.now(timezone.utc).isoformat(), "raw_completions": safe_relative(raw_path)})
        repair_path.write_text(json.dumps(repair, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"CE1 cache repair: {repair['records_before']} -> {repair['records_after']} records; removed {repair['duplicates_removed']} duplicate keys")
        return
    if args.retry_errors:
        if args.dry_run:
            raise ValueError("--retry-errors cannot be combined with --dry-run")
        retry = archive_error_records(raw_path, retry_archive_path)
        print(
            "CE1 error retry preparation: "
            f"{retry['records_before']} -> {retry['successful_kept']} canonical records; "
            f"archived {retry['errors_archived']} failures"
        )
    if args.summarize_only and (args.dry_run or args.retry_errors):
        raise ValueError("--summarize-only cannot be combined with --dry-run or --retry-errors")
    expected = len(users) * len(ARMS) * repeats
    print(f"CE1 run: {run_id}")
    print(f"  frozen pilot users: {len(users)}; planned independent reranks: {expected}")
    if args.dry_run:
        return
    cache = mh2.read_cache(raw_path)
    if args.summarize_only:
        summary = report_metrics(users, cache, repeats=repeats, bootstrap_samples=bootstrap_samples, seed=int(config.get("seed", 42)))
        if metrics_path.exists():
            summary["run"] = read_json(metrics_path).get("run", {})
        summary["run"]["resummarized_at_utc"] = datetime.now(timezone.utc).isoformat()
        metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"CE1 metrics resummarized without API calls: {summary['analysed_users']}/{summary['initial_users']} users analysed")
        return
    provider = config.get("provider", {})
    client = LLMClient(
        api_endpoint=provider.get("endpoint"),
        api_key=provider.get("api_key"),
        api_version=provider.get("api_version"),
        model=provider.get("model", config.get("llm_model")),
        provider_name=provider.get("name", "azure_openai"),
    )
    reranker = LLMReranker(client)
    jobs = []
    for user in users:
        for arm in ARMS:
            for repeat in range(1, repeats + 1):
                if report_key(int(user["user_id"]), arm, repeat) not in cache:
                    jobs.append((user, arm, repeat))
    started = time.monotonic()

    def run(job: tuple[Mapping[str, Any], str, int]) -> Dict[str, Any]:
        user, arm, repeat = job
        try:
            return call_reranker(client, reranker, user=user, arm=arm, repeat=repeat, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            return error_record(user_id=int(user["user_id"]), arm=arm, repeat=repeat, exc=exc)

    execute_missing(jobs, run, raw_path, cache, workers)
    summary = report_metrics(users, cache, repeats=repeats, bootstrap_samples=bootstrap_samples, seed=int(config.get("seed", 42)))
    summary["run"] = {
        "run_id": run_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "protocol": descriptor,
        "raw_completions": {"path": safe_relative(raw_path), "sha256": sha256(raw_path), "records": len(cache)},
        "wall_seconds_this_invocation": time.monotonic() - started,
        "model": client.model,
        "azure_deployment": client.request_model,
        "api_version": client.api_version,
        "token_stats": client.get_token_stats(),
        "api_cost_usd": None,
        "api_cost_note": "Azure deployment pricing was not supplied; token counts are recorded instead.",
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"CE1 complete: {summary['analysed_users']}/{summary['initial_users']} users analysed")
    for arm, values in summary["arms"].items():
        print(f"  {arm}: NDCG@5={values['ndcg_at_5']:.4f}")
    for arm, comparison in summary["comparisons"].items():
        low, high = comparison["bootstrap_95_ci"]
        print(f"  delta {arm} vs baseline: {comparison['delta_ndcg_at_5']:+.4f} [{low:+.4f}, {high:+.4f}]")
    print(f"  wrote {safe_relative(metrics_path)}")


if __name__ == "__main__":
    main()
