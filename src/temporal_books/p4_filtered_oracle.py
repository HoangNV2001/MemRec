"""Smoke-first filtered 3-hop rerank using frozen P3 packets and Stage-R."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.models.selfhost_transformers import TransformersJSONClient
from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.p2_analysis import paired_bootstrap_ci
from src.temporal_books.p2_oracle import Journal, call_phase, event_hit_at_5, event_ndcg_at_5, rerank_prompt, text, validate_ranking
from src.temporal_books.p3_oracle import event_key, resource_snapshot


SCHEMA_VERSION = 1
FILTERED_ARM = "filtered_three_hop"


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def successful_calls(path: Path) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") != "success":
            continue
        key = str(row["key"])
        if key in values:
            raise ValueError(f"duplicate successful frozen P3 call: {key}")
        values[key] = row
    return values


def load_contract(config: Mapping[str, Any], config_path: Path) -> tuple[Dict[str, Any], Dict[str, list[str]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    p4 = config["p4"]
    paths = {
        "prepared": project_path(p4["p3_prepared"]),
        "calls": project_path(p4["p3_calls"]),
        "p3_manifest": project_path(p4["p3_manifest"]),
        "preflight": project_path(p4["preflight_manifest"]),
        "ledger": project_path(p4["ledger"]),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing P4 dependency {name}: {path}")
    for name, config_key in (("prepared", "p3_prepared_sha256"), ("calls", "p3_calls_sha256"), ("p3_manifest", "p3_manifest_sha256")):
        actual = sha256_file(paths[name])
        if actual != str(p4[config_key]):
            raise RuntimeError(f"frozen P3 {name} hash mismatch: {actual}")
    preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    if preflight["inputs"]["config"]["sha256"] != sha256_file(config_path):
        raise RuntimeError("filtered preflight config hash does not match")
    if preflight["admission"]["decision"] != "pass":
        raise RuntimeError("filtered structural coverage did not pass admission")
    if preflight["output"]["sha256"] != sha256_file(paths["ledger"]):
        raise RuntimeError("filtered ledger hash does not match preflight")
    threshold = int(p4["support_threshold"])
    origins = {
        str(row["endpoint_item_id"]): [str(value) for value in row["origin_users_by_quality"]]
        for row in read_jsonl(paths["ledger"])
        if int(row["n_independent_origins"]) >= threshold
    }
    prepared = json.loads(paths["prepared"].read_text(encoding="utf-8"))
    frozen = successful_calls(paths["calls"])
    required = []
    for packet in prepared["source_packets"]:
        required.append(f"packet:{packet['source_user_id']}")
    for event in prepared["events"]:
        key = event_key(event)
        required.extend((f"stage_r:{key}", f"rerank:local:{key}", f"rerank:oracle_three_hop:{key}"))
    missing = [key for key in required if key not in frozen]
    if missing:
        raise RuntimeError(f"frozen P3 cache is incomplete: {len(missing)} key(s) missing")
    audit = {
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": sha256_file(paths["prepared"]),
        "p3_calls_sha256": sha256_file(paths["calls"]),
        "p3_manifest_sha256": sha256_file(paths["p3_manifest"]),
        "preflight_sha256": sha256_file(paths["preflight"]),
        "ledger_sha256": sha256_file(paths["ledger"]),
    }
    return prepared, origins, frozen, audit


def overlay_filtered_memories(
    event: Mapping[str, Any],
    prepared: Mapping[str, Any],
    origins: Mapping[str, Sequence[str]],
    frozen_calls: Mapping[str, Mapping[str, Any]],
    *,
    packet_cap: int,
) -> Dict[str, str]:
    memories = {item_id: prepared["item_info"][item_id]["base_memory"] for item_id in event["candidate_item_ids"]}
    gold = str(event["gold_item_id"])
    packet_text = []
    for source in origins.get(gold, ())[:packet_cap]:
        response = frozen_calls[f"packet:{source}"]["value"]
        packet_text.append(text(str(response["memory"]), 180))
    if packet_text:
        memories[gold] = text(memories[gold] + "\nFiltered three-hop community packets: " + " | ".join(packet_text), 600)
    return memories


def build_jobs(
    prepared: Mapping[str, Any],
    origins: Mapping[str, Sequence[str]],
    frozen_calls: Mapping[str, Mapping[str, Any]],
    *,
    packet_cap: int,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> list[Any]:
    jobs = []
    for event in events or prepared["events"]:
        key = event_key(event)
        facets = frozen_calls[f"stage_r:{key}"]["value"].get("facets", [])
        memories = overlay_filtered_memories(event, prepared, origins, frozen_calls, packet_cap=packet_cap)
        messages, schema = rerank_prompt(event, facets, memories)
        jobs.append(
            (
                f"rerank:{FILTERED_ARM}:{key}",
                "rerank",
                messages,
                schema,
                lambda response, candidates=event["candidate_item_ids"]: validate_ranking(response, candidates),
            )
        )
    return jobs


def smoke_events(prepared: Mapping[str, Any], count: int) -> list[Dict[str, Any]]:
    return sorted(prepared["events"], key=lambda event: stable_order(f"p4-smoke-event\0{event_key(event)}"))[:count]


def model_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    self_host = config["self_host"]
    return {key: self_host[key] for key in ("model", "revision", "dtype", "tensor_parallel_size", "hard_memory_fraction", "max_model_len", "seed", "temperature", "structured_decoding", "max_output_tokens")}


def compare(delta: Sequence[float], *, resamples: int, seed: int) -> Dict[str, Any]:
    lower, upper = paired_bootstrap_ci(delta, resamples=resamples, seed=seed)
    return {
        "delta_ndcg_at_5": sum(delta) / len(delta),
        "paired_bootstrap_95_ci": [lower, upper],
        "improved_events": sum(value > 0 for value in delta),
        "worsened_events": sum(value < 0 for value in delta),
        "unchanged_events": sum(value == 0 for value in delta),
    }


def summarize(
    prepared: Mapping[str, Any],
    frozen_calls: Mapping[str, Mapping[str, Any]],
    completed: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    client: TransformersJSONClient,
    journal: Journal,
) -> Dict[str, Any]:
    rankings: Dict[str, list[Sequence[str]]] = {"local": [], "oracle_three_hop": [], FILTERED_ARM: []}
    for event in prepared["events"]:
        key = event_key(event)
        rankings["local"].append(frozen_calls[f"rerank:local:{key}"]["value"])
        rankings["oracle_three_hop"].append(frozen_calls[f"rerank:oracle_three_hop:{key}"]["value"])
        rankings[FILTERED_ARM].append(completed[f"rerank:{FILTERED_ARM}:{key}"]["value"])
    arms: Dict[str, Any] = {}
    ndcgs: Dict[str, list[float]] = {}
    for arm, rows in rankings.items():
        arm_ndcg, hits = [], []
        for event, ranking in zip(prepared["events"], rows):
            gold = str(event["gold_item_id"])
            arm_ndcg.append(event_ndcg_at_5(ranking, gold))
            hits.append(event_hit_at_5(ranking, gold))
        ndcgs[arm] = arm_ndcg
        arms[arm] = {"n_events": len(rows), "ndcg_at_5": sum(arm_ndcg) / len(rows), "hit_at_5": sum(hits) / len(hits), "per_event_ndcg_at_5": arm_ndcg}
    p4 = config["p4"]
    resamples, seed = int(p4["bootstrap_resamples"]), int(p4["bootstrap_seed"])
    vs_local = compare([value - base for value, base in zip(ndcgs[FILTERED_ARM], ndcgs["local"])], resamples=resamples, seed=seed)
    vs_raw = compare([value - base for value, base in zip(ndcgs[FILTERED_ARM], ndcgs["oracle_three_hop"])], resamples=resamples, seed=seed)
    gate = (
        vs_local["delta_ndcg_at_5"] >= float(p4["filtered_vs_local_min_delta"])
        and vs_local["paired_bootstrap_95_ci"][0] > 0
        and vs_raw["delta_ndcg_at_5"] >= float(p4["filtered_vs_raw_three_hop_min_delta"])
        and vs_raw["paired_bootstrap_95_ci"][0] > 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": p4["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
        "comparisons": {"filtered_vs_local": vs_local, "filtered_vs_raw_three_hop": vs_raw},
        "gate": {
            "criteria": {"filtered_vs_local_min_delta": float(p4["filtered_vs_local_min_delta"]), "filtered_vs_raw_three_hop_min_delta": float(p4["filtered_vs_raw_three_hop_min_delta"]), "both_ci_lower_gt_zero": True},
            "decision": "pass" if gate else "hard_stop",
        },
        "request_journal": {"attempts": journal.attempts, "primary_attempts": journal.primary_attempts, "retry_attempts": journal.retry_attempts},
        "invocation_token_stats": client.get_token_stats(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filtered 3-hop self-host rerank")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p4_filtered_three_hop.yaml")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--request", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_only and args.request:
        raise SystemExit("choose smoke-only or request")
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    p4, self_host = config["p4"], config["self_host"]
    prepared, origins, frozen, audit = load_contract(config, config_path)
    selected_smoke = smoke_events(prepared, int(p4["smoke_events"]))
    if not args.smoke_only and not args.request:
        print(json.dumps({"events": len(prepared["events"]), "filtered_support2_endpoints": len(origins), "smoke_events": len(selected_smoke), "new_generations": 0}))
        return
    primary, retry, maximum = len(prepared["events"]), int(p4["retry_reserve"]), int(p4["hard_generation_cap"])
    if primary + retry > maximum:
        raise ValueError("filtered P4 request budget exceeds hard cap")
    manifest_path = project_path(p4["manifest"])
    if args.request and manifest_path.exists():
        raise RuntimeError("filtered P4 completion manifest exists; do not overwrite")
    journal = Journal.load(project_path(p4["attempts"]), project_path(p4["calls"]), maximum=maximum, primary_limit=primary, retry_limit=retry)
    smoke_path = project_path(p4["llm_smoke_manifest"])
    smoke_keys = [f"rerank:{FILTERED_ARM}:{event_key(event)}" for event in selected_smoke]
    if args.request:
        if not smoke_path.exists():
            raise RuntimeError("filtered full run blocked: LLM smoke manifest missing")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        expected = {"decision": "pass", "input_audit": audit, "model_contract": model_contract(config), "event_keys": [event_key(event) for event in selected_smoke]}
        for key, value in expected.items():
            if smoke.get(key) != value:
                raise RuntimeError(f"filtered full run blocked: smoke {key} mismatch")
        if any(key not in (journal.completed or {}) for key in smoke_keys):
            raise RuntimeError("filtered full run blocked: smoke cache incomplete")
    before = resource_snapshot()
    client = TransformersJSONClient(
        model_name=str(self_host["model"]),
        revision=str(self_host["revision"]),
        dtype=str(self_host["dtype"]),
        seed=int(self_host["seed"]),
        hard_memory_fraction=float(self_host["hard_memory_fraction"]),
        max_model_len=int(self_host["max_model_len"]),
    )
    jobs = build_jobs(prepared, origins, frozen, packet_cap=int(p4["overlay_packet_cap"]), events=selected_smoke if args.smoke_only else None)
    call_phase(jobs=jobs, client=client, journal=journal, workers=1, max_tokens=int(self_host["max_output_tokens"]["rerank"]))
    if args.smoke_only:
        stats = client.get_token_stats()
        if stats["repaired_responses"] or float(stats["peak_vram_gib"]) > 20.0:
            raise RuntimeError("filtered LLM smoke failed repair/VRAM gate")
        if any(key not in (journal.completed or {}) for key in smoke_keys):
            raise RuntimeError("filtered LLM smoke incomplete")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": p4["run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "pass",
            "input_audit": audit,
            "model_contract": model_contract(config),
            "event_keys": [event_key(event) for event in selected_smoke],
            "attempts_after_smoke": journal.attempts,
            "runtime_stats": stats,
            "resource_before": before,
        }
        smoke_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"smoke": "pass", "events": len(selected_smoke), "attempts": journal.attempts, "peak_vram_gib": stats["peak_vram_gib"]}))
        return
    metrics = summarize(prepared, frozen, journal.completed or {}, config, client, journal)
    metrics_path = project_path(p4["metrics"])
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    after = resource_snapshot()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p4["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_audit": audit,
        "smoke": {"path": str(smoke_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(smoke_path)},
        "metrics": {"path": str(metrics_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(metrics_path)},
        "attempts": {"path": str(project_path(p4["attempts"]).relative_to(PROJECT_ROOT)), "records": journal.attempts},
        "calls": {"path": str(project_path(p4["calls"]).relative_to(PROJECT_ROOT))},
        "model_contract": model_contract(config),
        "resource_before": before,
        "resource_after": after,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    printable = {arm: {key: value for key, value in row.items() if key != "per_event_ndcg_at_5"} for arm, row in metrics["arms"].items()}
    print(json.dumps({"attempts": journal.attempts, "arms": printable, "comparisons": metrics["comparisons"], "gate": metrics["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
