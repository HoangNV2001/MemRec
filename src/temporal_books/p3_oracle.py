"""Smoke-first self-hosted comparison of local, 2-hop oracle and 3-hop oracle."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.models.selfhost_transformers import TransformersJSONClient
from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, sha256_file, stable_order
from src.temporal_books.p2_analysis import paired_bootstrap_ci
from src.temporal_books.p2_oracle import (
    Journal,
    call_phase,
    event_hit_at_5,
    event_key,
    event_ndcg_at_5,
    packet_prompt,
    rerank_prompt,
    stage_r_prompt,
    text,
    validate_ranking,
)


SCHEMA_VERSION = 1
ARMS = ("local", "oracle_two_hop", "oracle_three_hop")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(config: Mapping[str, Any], config_path: Path, *, force: bool) -> Dict[str, Any]:
    p3 = config["p3"]
    base_path = project_path(p3["base_prepared"])
    ledger_path = project_path(p3["ledger"])
    preflight_path = project_path(p3["preflight_manifest"])
    output_path = project_path(p3["prepared"])
    for path in (base_path, ledger_path, preflight_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight["inputs"]["config"]["sha256"] != sha256_file(config_path):
        raise RuntimeError("P3 preflight config hash does not match current P3 config")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if len(base["events"]) != int(config["experiment"]["evaluation_events"]):
        raise ValueError("base prepared event count does not match P3 experiment")
    if len(base["source_packets"]) != int(config["experiment"]["semantic_packet_sources"]):
        raise ValueError("base prepared packet count does not match P3 experiment")
    threshold = int(config["experiment"]["support_threshold"])
    origins3 = {
        str(row["endpoint_item_id"]): [str(value) for value in row["origin_users_by_path_support"]]
        for row in read_jsonl(ledger_path)
        if int(row["n_independent_origins"]) >= threshold
    }
    prepared = {
        **base,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "p3_inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "base_prepared": {"path": str(base_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(base_path)},
            "three_hop_ledger": {"path": str(ledger_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(ledger_path)},
            "preflight_manifest": {"path": str(preflight_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(preflight_path)},
        },
        "p3_protocol": {
            "arms": list(ARMS),
            "same_candidates_and_stage_r_across_arms": True,
            "oracle_scope": "2-hop and 3-hop may attach at most two source packets to labelled gold only",
            "three_hop_origin_order": "descending path support, then source user ID",
            "support_threshold": threshold,
        },
        "origins_by_three_hop_endpoint_support_gte_threshold": origins3,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return prepared


def overlay_memories(
    event: Mapping[str, Any],
    prepared: Mapping[str, Any],
    packet_outputs: Mapping[str, Mapping[str, Any]],
    *,
    arm: str,
    packet_cap: int,
) -> Dict[str, str]:
    memories = {item_id: prepared["item_info"][item_id]["base_memory"] for item_id in event["candidate_item_ids"]}
    if arm == "local":
        return memories
    gold = str(event["gold_item_id"])
    if arm == "oracle_two_hop":
        origins = prepared["origins_by_endpoint_support_gte_threshold"].get(gold, ())
        heading = "Oracle two-hop community packets"
    elif arm == "oracle_three_hop":
        origins = prepared["origins_by_three_hop_endpoint_support_gte_threshold"].get(gold, ())
        heading = "Oracle three-hop community packets"
    else:
        raise ValueError(f"unknown P3 arm: {arm}")
    packet_text = [text(str(packet_outputs[source]["memory"]), 180) for source in origins if source in packet_outputs][:packet_cap]
    if packet_text:
        memories[gold] = text(memories[gold] + "\n" + heading + ": " + " | ".join(packet_text), 600)
    return memories


def build_jobs(
    prepared: Mapping[str, Any],
    *,
    phase: str,
    packet_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    stage_r_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    packet_cap: int = 2,
) -> list[Any]:
    jobs: list[Any] = []
    if phase == "packet":
        for packet in prepared["source_packets"]:
            key = f"packet:{packet['source_user_id']}"
            messages, schema = packet_prompt(packet)
            jobs.append((key, "packet", messages, schema, lambda response: response))
    elif phase == "stage_r":
        for event in prepared["events"]:
            key = f"stage_r:{event_key(event)}"
            messages, schema = stage_r_prompt(event)
            jobs.append((key, "stage_r", messages, schema, lambda response: response))
    elif phase == "rerank":
        if packet_outputs is None or stage_r_outputs is None:
            raise ValueError("rerank requires packet and Stage-R outputs")
        packets = {key.removeprefix("packet:"): value for key, value in packet_outputs.items() if key.startswith("packet:")}
        for event in prepared["events"]:
            facets = stage_r_outputs[f"stage_r:{event_key(event)}"].get("facets", [])
            for arm in ARMS:
                memories = overlay_memories(event, prepared, packets, arm=arm, packet_cap=packet_cap)
                messages, schema = rerank_prompt(event, facets, memories)
                key = f"rerank:{arm}:{event_key(event)}"
                jobs.append((key, "rerank", messages, schema, lambda response, candidates=event["candidate_item_ids"]: validate_ranking(response, candidates)))
    else:
        raise ValueError(f"unknown phase: {phase}")
    return jobs


def smoke_subset(prepared: Mapping[str, Any], experiment: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    events = sorted(prepared["events"], key=lambda event: stable_order(f"p3-smoke-event\0{event_key(event)}"))[
        : int(experiment["smoke_events"])
    ]
    packets = sorted(
        prepared["source_packets"], key=lambda packet: stable_order(f"p3-smoke-packet\0{packet['source_user_id']}")
    )
    selected = {str(packet["source_user_id"]) for packet in packets[: int(experiment["smoke_base_packet_sources"])]}
    cap = int(experiment["overlay_packet_cap"])
    for event in events:
        gold = str(event["gold_item_id"])
        selected.update(prepared["origins_by_endpoint_support_gte_threshold"].get(gold, ())[:cap])
        selected.update(prepared["origins_by_three_hop_endpoint_support_gte_threshold"].get(gold, ())[:cap])
    selected_packets = [packet for packet in prepared["source_packets"] if str(packet["source_user_id"]) in selected]
    return selected_packets, events


def resource_snapshot() -> Dict[str, Any]:
    expected_job = os.environ.get("MEMREC_EXPECTED_SLURM_JOB_ID")
    actual_job = os.environ.get("SLURM_JOB_ID")
    if not expected_job or actual_job != expected_job:
        raise RuntimeError(f"expected Slurm job {expected_job!r}, running in {actual_job!r}")
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1:
        raise RuntimeError("P3 requires exactly one CUDA_VISIBLE_DEVICES entry")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        "slurm_job_id": actual_job,
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "cuda_visible_devices": visible,
        "all_gpu_snapshot": [line.strip() for line in output.splitlines() if line.strip()],
    }


def run_phases(
    prepared: Mapping[str, Any], config: Mapping[str, Any], client: TransformersJSONClient, journal: Journal
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    p3 = config["p3"]
    experiment = config["experiment"]
    max_tokens = config["self_host"]["max_output_tokens"]
    workers = int(p3["workers"])
    if workers != 1:
        raise ValueError("direct Transformers self-host runner is locked to one worker")
    packet_outputs = call_phase(
        jobs=build_jobs(prepared, phase="packet"), client=client, journal=journal, workers=workers, max_tokens=int(max_tokens["packet"])
    )
    stage_outputs = call_phase(
        jobs=build_jobs(prepared, phase="stage_r"), client=client, journal=journal, workers=workers, max_tokens=int(max_tokens["stage_r"])
    )
    rerank_outputs = call_phase(
        jobs=build_jobs(
            prepared,
            phase="rerank",
            packet_outputs=packet_outputs,
            stage_r_outputs=stage_outputs,
            packet_cap=int(experiment["overlay_packet_cap"]),
        ),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(max_tokens["rerank"]),
    )
    return packet_outputs, stage_outputs, rerank_outputs


def required_smoke_keys(packets: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = [f"packet:{packet['source_user_id']}" for packet in packets]
    for event in events:
        keys.append(f"stage_r:{event_key(event)}")
        keys.extend(f"rerank:{arm}:{event_key(event)}" for arm in ARMS)
    return keys


def model_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    self_host = config["self_host"]
    return {key: self_host[key] for key in ("model", "revision", "dtype", "tensor_parallel_size", "hard_memory_fraction", "max_model_len", "seed", "temperature", "structured_decoding", "max_output_tokens")}


def summarize(prepared: Mapping[str, Any], rerank_outputs: Mapping[str, Sequence[str]], config: Mapping[str, Any], client: TransformersJSONClient, journal: Journal) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["p3"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_journal": {"attempts": journal.attempts, "primary_attempts": journal.primary_attempts, "retry_attempts": journal.retry_attempts},
        "invocation_token_stats": client.get_token_stats(),
        "arms": {},
        "comparisons": {},
    }
    per_arm: Dict[str, list[float]] = {}
    for arm in ARMS:
        ndcgs: list[float] = []
        hits: list[float] = []
        for event in prepared["events"]:
            ranking = rerank_outputs[f"rerank:{arm}:{event_key(event)}"]
            ndcgs.append(event_ndcg_at_5(ranking, str(event["gold_item_id"])))
            hits.append(event_hit_at_5(ranking, str(event["gold_item_id"])))
        per_arm[arm] = ndcgs
        result["arms"][arm] = {"n_events": len(ndcgs), "ndcg_at_5": sum(ndcgs) / len(ndcgs), "hit_at_5": sum(hits) / len(hits), "per_event_ndcg_at_5": ndcgs}
    base = per_arm["local"]
    resamples = int(config["p3"]["bootstrap_resamples"])
    seed = int(config["p3"]["bootstrap_seed"])
    for arm in ("oracle_two_hop", "oracle_three_hop"):
        delta = [value - reference for value, reference in zip(per_arm[arm], base)]
        lower, upper = paired_bootstrap_ci(delta, resamples=resamples, seed=seed)
        result["arms"][arm].update(
            {
                "delta_ndcg_at_5_vs_local": sum(delta) / len(delta),
                "paired_bootstrap_95_ci": [lower, upper],
                "improved_events": sum(value > 0 for value in delta),
                "worsened_events": sum(value < 0 for value in delta),
                "unchanged_events": sum(value == 0 for value in delta),
            }
        )
    three_vs_two = [
        value - reference
        for value, reference in zip(per_arm["oracle_three_hop"], per_arm["oracle_two_hop"])
    ]
    lower, upper = paired_bootstrap_ci(three_vs_two, resamples=resamples, seed=seed)
    result["comparisons"]["oracle_three_hop_vs_oracle_two_hop"] = {
        "delta_ndcg_at_5": sum(three_vs_two) / len(three_vs_two),
        "paired_bootstrap_95_ci": [lower, upper],
        "improved_events": sum(value > 0 for value in three_vs_two),
        "worsened_events": sum(value < 0 for value in three_vs_two),
        "unchanged_events": sum(value == 0 for value in three_vs_two),
    }
    three = result["arms"]["oracle_three_hop"]
    comparison = result["comparisons"]["oracle_three_hop_vs_oracle_two_hop"]
    result["gate"] = {
        "criteria": {
            "three_hop_vs_local_min_delta": 0.05,
            "three_hop_vs_two_hop_min_delta": 0.02,
            "both_paired_ci_lower_gt_zero": True,
        },
        "decision": "pass"
        if float(three["delta_ndcg_at_5_vs_local"]) >= 0.05
        and float(three["paired_bootstrap_95_ci"][0]) > 0
        and float(comparison["delta_ndcg_at_5"]) >= 0.02
        and float(comparison["paired_bootstrap_95_ci"][0]) > 0
        else "hard_stop",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-hosted bounded 3-hop oracle")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p3_selfhost.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--request", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum(bool(value) for value in (args.prepare_only, args.smoke_only, args.request)) > 1:
        raise SystemExit("choose exactly one execution mode")
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    p3 = config["p3"]
    experiment = config["experiment"]
    prepared = prepare(config, config_path, force=args.force_prepare)
    if not args.smoke_only and not args.request:
        packets, events = smoke_subset(prepared, experiment)
        print(json.dumps({"prepared_events": len(prepared["events"]), "prepared_packets": len(prepared["source_packets"]), "smoke_events": len(events), "smoke_packets": len(packets), "generations": 0}))
        return
    if int(config["self_host"]["tensor_parallel_size"]) != 1:
        raise ValueError("shared allocation permits tensor_parallel_size=1 only")
    maximum = int(experiment["hard_generation_cap"])
    primary = int(experiment["semantic_packet_sources"]) + int(experiment["evaluation_events"]) * (1 + len(ARMS))
    if primary + int(experiment["retry_reserve"]) > maximum:
        raise ValueError("P3 generation budget exceeds hard cap")
    manifest_path = project_path(p3["manifest"])
    if args.request and manifest_path.exists():
        raise RuntimeError("P3 completion manifest already exists; do not overwrite")
    journal = Journal.load(
        project_path(p3["attempts"]),
        project_path(p3["calls"]),
        maximum=maximum,
        primary_limit=primary,
        retry_limit=int(experiment["retry_reserve"]),
    )
    smoke_packets, smoke_events = smoke_subset(prepared, experiment)
    smoke_path = project_path(p3["llm_smoke_manifest"])
    if args.request:
        if not smoke_path.exists():
            raise RuntimeError("full P3 is blocked: self-host smoke manifest is missing")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        expected_fields = {
            "decision": "pass",
            "config_sha256": sha256_file(config_path),
            "prepared_sha256": sha256_file(project_path(p3["prepared"])),
            "model_contract": model_contract(config),
            "packet_source_ids": [str(packet["source_user_id"]) for packet in smoke_packets],
            "event_keys": [event_key(event) for event in smoke_events],
        }
        for key, expected in expected_fields.items():
            if smoke.get(key) != expected:
                raise RuntimeError(f"full P3 is blocked: smoke field {key} does not match")
        missing = [key for key in required_smoke_keys(smoke_packets, smoke_events) if key not in (journal.completed or {})]
        if missing:
            raise RuntimeError(f"full P3 is blocked: {len(missing)} smoke cache key(s) missing")
    snapshot_before = resource_snapshot()
    self_host = config["self_host"]
    client = TransformersJSONClient(
        model_name=str(self_host["model"]),
        revision=str(self_host["revision"]),
        dtype=str(self_host["dtype"]),
        seed=int(self_host["seed"]),
        hard_memory_fraction=float(self_host["hard_memory_fraction"]),
        max_model_len=int(self_host["max_model_len"]),
    )
    if args.smoke_only:
        smoke_prepared = {**prepared, "source_packets": smoke_packets, "events": smoke_events}
        run_phases(smoke_prepared, config, client, journal)
        stats = client.get_token_stats()
        if float(stats["peak_vram_gib"]) > 20.0:
            raise RuntimeError(f"smoke exceeded 20 GiB VRAM: {stats['peak_vram_gib']:.3f}")
        if int(stats["repaired_responses"]) != 0:
            raise RuntimeError("grammar-constrained smoke must produce strict JSON without parser repairs")
        missing = [key for key in required_smoke_keys(smoke_packets, smoke_events) if key not in (journal.completed or {})]
        if missing:
            raise RuntimeError(f"smoke incomplete: {len(missing)} key(s) missing")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": p3["run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "pass",
            "config_sha256": sha256_file(config_path),
            "prepared_sha256": sha256_file(project_path(p3["prepared"])),
            "model_contract": model_contract(config),
            "resource_before": snapshot_before,
            "runtime_stats": stats,
            "packet_source_ids": [str(packet["source_user_id"]) for packet in smoke_packets],
            "event_keys": [event_key(event) for event in smoke_events],
            "attempts_after_smoke": journal.attempts,
        }
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"smoke": "pass", "events": len(smoke_events), "packets": len(smoke_packets), "attempts": journal.attempts, "peak_vram_gib": stats["peak_vram_gib"]}, ensure_ascii=False))
        return
    _, _, reranks = run_phases(prepared, config, client, journal)
    metrics = summarize(prepared, reranks, config, client, journal)
    metrics_path = project_path(p3["metrics"])
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    snapshot_after = resource_snapshot()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p3["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "prepared": {"path": str(project_path(p3["prepared"]).relative_to(PROJECT_ROOT)), "sha256": sha256_file(project_path(p3["prepared"]))},
        "smoke": {"path": str(smoke_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(smoke_path)},
        "metrics": {"path": str(metrics_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(metrics_path)},
        "attempts": {"path": str(project_path(p3["attempts"]).relative_to(PROJECT_ROOT)), "records": journal.attempts},
        "calls": {"path": str(project_path(p3["calls"]).relative_to(PROJECT_ROOT))},
        "resource_before": snapshot_before,
        "resource_after": snapshot_after,
        "model_contract": model_contract(config),
        "hard_generation_cap": maximum,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"attempts": journal.attempts, "arms": {arm: {key: value for key, value in row.items() if key != "per_event_ndcg_at_5"} for arm, row in metrics["arms"].items()}, "comparisons": metrics["comparisons"], "gate": metrics["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
