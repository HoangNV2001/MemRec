"""P2: bounded 1,000-request temporal item-memory oracle pilot.

The runner is intentionally split into a free preparation stage and an explicit
``--request`` stage.  It uses P1's 560 candidate-blind source packets, freezes a
single local Stage-R profile per future event, and makes three listwise rerank
calls per event:

* ``local``: metadata/review-derived candidate memory only;
* ``one_hop``: candidate-blind packet overlays on direct anchor items;
* ``oracle_two_hop``: a labelled upper bound that may attach support>=2 routed
  packet memories to the *gold* endpoint only.

The oracle is never described as deployable.  Its only role is to test whether
the routed packet pool has enough semantic headroom to justify a later
candidate-blind selector.  Every physical request is journalled before it is
sent; primary jobs are 960 and retries are hard-limited to the 40-request
reserve, even across resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from dotenv import load_dotenv

from src.models.llm_client import LLMClient
from src.temporal_books.common import PROJECT_ROOT, load_yaml, project_path, set_csv_field_limit, sha256_file, stable_order


SCHEMA_VERSION = 1
ARMS = ("local", "one_hop", "oracle_two_hop")
LETTERS = tuple("ABCDEFGHIJ")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.RLock) -> None:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def text(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: max(0, limit - 1)] + "…"


def event_ndcg_at_5(ranking: Sequence[str], gold_item_id: str) -> float:
    try:
        rank = list(ranking).index(gold_item_id) + 1
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 1) if rank <= 5 else 0.0


def event_hit_at_5(ranking: Sequence[str], gold_item_id: str) -> float:
    return float(gold_item_id in ranking[:5])


def stable_sample_candidates(pool: Sequence[str], known_items: set[str], gold_item_id: str, *, n_candidates: int, seed_key: str) -> list[str]:
    if n_candidates != len(LETTERS):
        raise ValueError("P2 uses exactly ten candidates so the prompt labels remain A-J")
    if gold_item_id not in pool:
        raise ValueError("gold item is not available in the pre-cutoff candidate pool")
    rng = random.Random(stable_order(seed_key))
    negatives: list[str] = []
    blocked = set(known_items)
    blocked.add(gold_item_id)
    attempts = 0
    while len(negatives) < n_candidates - 1:
        candidate = pool[rng.randrange(len(pool))]
        attempts += 1
        if candidate not in blocked:
            blocked.add(candidate)
            negatives.append(candidate)
        if attempts > len(pool) * 20:
            raise RuntimeError("unable to sample enough pre-cutoff negative items")
    candidates = [gold_item_id, *negatives]
    rng.shuffle(candidates)
    return candidates


def load_metadata(path: Path, needed_titles: set[str]) -> Dict[str, Dict[str, str]]:
    set_csv_field_limit()
    result: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            title_value = (row.get("Title") or "").strip()
            if title_value in needed_titles:
                result[title_value] = {
                    "description": row.get("description") or "",
                    "authors": row.get("authors") or "",
                    "categories": row.get("categories") or "",
                }
    return result


def event_key(event: Mapping[str, Any]) -> str:
    return f"{event['user_id']}:{event['timestamp']}"


def smoke_subset(prepared: Mapping[str, Any], p2: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Choose a deterministic smoke cohort and every packet it can reference.

    Rerank prompts in the smoke must be byte-for-byte equivalent to their full
    run prompt.  Consequently the packet subset includes both a representative
    base sample and every direct/oracle route source referenced by its events.
    """
    event_count = int(p2["smoke_evaluation_events"])
    packet_count = int(p2["smoke_packet_sources"])
    if event_count <= 0 or packet_count <= 0:
        raise ValueError("smoke sample sizes must be positive")
    events = sorted(
        prepared["events"],
        key=lambda event: stable_order(f"p2-smoke-event\0{event_key(event)}"),
    )[:event_count]
    if len(events) != event_count:
        raise ValueError("not enough prepared events for smoke cohort")
    required_sources: set[str] = set()
    anchors_by_item = prepared["anchors_by_item"]
    origins_by_endpoint = prepared["origins_by_endpoint_support_gte_threshold"]
    for event in events:
        for item_id in event["candidate_item_ids"]:
            required_sources.update(str(source) for source in anchors_by_item.get(item_id, ()))
        required_sources.update(str(source) for source in origins_by_endpoint.get(event["gold_item_id"], ()))
    ordered_sources = sorted(
        prepared["source_packets"],
        key=lambda packet: stable_order(f"p2-smoke-packet\0{packet['source_user_id']}"),
    )
    selected_source_ids = {str(packet["source_user_id"]) for packet in ordered_sources[:packet_count]}
    selected_source_ids.update(required_sources)
    packets = [packet for packet in prepared["source_packets"] if str(packet["source_user_id"]) in selected_source_ids]
    if len(packets) > len(prepared["source_packets"]):
        raise AssertionError("smoke packet selection exceeded prepared packet pool")
    return packets, events


def smoke_manifest_payload(
    *,
    config_path: Path,
    prepared_path: Path,
    run_id: str,
    packets: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    journal: "Journal",
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": sha256_file(prepared_path),
        "packet_source_ids": [str(packet["source_user_id"]) for packet in packets],
        "event_keys": [event_key(event) for event in events],
        "journal_attempts_after_smoke": journal.attempts,
    }


def candidate_base_memory(title_value: str, metadata: Mapping[str, str], cap: int) -> str:
    pieces = [title_value]
    if metadata.get("description"):
        pieces.append(metadata["description"])
    if metadata.get("authors"):
        pieces.append("Authors: " + metadata["authors"])
    if metadata.get("categories"):
        pieces.append("Categories: " + metadata["categories"])
    return text(". ".join(pieces), cap)


def prepare(config: Mapping[str, Any], config_path: Path, *, force: bool) -> Dict[str, Any]:
    """Materialize 100 fixed candidate sets and user histories without an LLM."""
    dataset = config["dataset"]
    p2 = config["p2"]
    p1_manifest_path = project_path(dataset["p1_manifest"])
    eval_path = project_path(dataset["p1_eval_events"])
    sources_path = project_path(dataset["p1_packet_sources"])
    ledger_path = project_path(dataset["p1_ledger"])
    reviews_path = project_path(dataset["reviews_file"])
    metadata_path = project_path(dataset["metadata_file"])
    prepared_path = project_path(p2["prepared"])
    required = (p1_manifest_path, eval_path, sources_path, ledger_path, reviews_path, metadata_path)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if prepared_path.exists() and not force:
        return json.loads(prepared_path.read_text(encoding="utf-8"))

    p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
    if p1_manifest["admission"]["decision"] != "pass":
        raise ValueError("P2 is forbidden because P1 did not pass")
    targets = read_jsonl(eval_path)
    sources = read_jsonl(sources_path)
    ledger_rows = read_jsonl(ledger_path)
    expected_events = int(config["llm_budget"]["evaluation_events"])
    expected_sources = int(config["llm_budget"]["semantic_packet_requests"])
    if len(targets) != expected_events or len(sources) != expected_sources:
        raise ValueError("P1 artifacts do not match the locked P2 request budget")
    p0 = json.loads(project_path(dataset["p0_audit"]).read_text(encoding="utf-8"))
    train_cutoff = int(p0["temporal_split"]["train_cutoff"])
    targets_by_user = {str(row["user_id"]): row for row in targets}
    if len(targets_by_user) != len(targets):
        raise ValueError("P2 requires exactly one fixed evaluation target per user")

    item_titles: Dict[str, str] = {}
    train_items: set[str] = set()
    histories: Dict[str, list[tuple[int, str, str, str, str]]] = {user_id: [] for user_id in targets_by_user}
    set_csv_field_limit()
    with reviews_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = (row.get("User_id") or "").strip()
            item_id = (row.get("Id") or "").strip()
            if not user_id or not item_id:
                continue
            try:
                timestamp = int(row["review/time"])
            except (KeyError, TypeError, ValueError):
                continue
            title_value = (row.get("Title") or "").strip()
            if title_value and item_id not in item_titles:
                item_titles[item_id] = title_value
            if timestamp < train_cutoff:
                train_items.add(item_id)
            target = targets_by_user.get(user_id)
            if target is not None and timestamp < int(target["timestamp"]):
                histories[user_id].append(
                    (timestamp, item_id, title_value, row.get("review/summary") or "", row.get("review/text") or "")
                )
    pool = sorted(train_items)
    n_candidates = int(p2["n_candidates"])
    prepared_events: list[Dict[str, Any]] = []
    needed_titles: set[str] = set()
    history_cap = int(p2["user_history_reviews"])
    for target in targets:
        user_id = str(target["user_id"])
        history = sorted(histories[user_id])
        known = {item_id for _, item_id, _, _, _ in history}
        candidates = stable_sample_candidates(
            pool,
            known,
            str(target["gold_item_id"]),
            n_candidates=n_candidates,
            seed_key=f"p2-candidates\0{user_id}\0{target['timestamp']}\0{target['gold_item_id']}",
        )
        for candidate in candidates:
            needed_titles.add(item_titles.get(candidate, ""))
        recent = history[-history_cap:]
        prepared_events.append(
            {
                **target,
                "candidate_item_ids": candidates,
                "history": [
                    {
                        "timestamp": timestamp,
                        "item_id": item_id,
                        "title": title_value,
                        "review_summary": text(summary, 180),
                        "review_text": text(review, 360),
                    }
                    for timestamp, item_id, title_value, summary, review in recent
                ],
            }
        )
    metadata = load_metadata(metadata_path, needed_titles - {""})
    item_info = {
        item_id: {
            "title": item_titles.get(item_id, f"Item {item_id}"),
            "base_memory": candidate_base_memory(
                item_titles.get(item_id, f"Item {item_id}"),
                metadata.get(item_titles.get(item_id, ""), {}),
                int(p2["candidate_memory_char_cap"]),
            ),
        }
        for event in prepared_events
        for item_id in event["candidate_item_ids"]
    }
    origins_by_endpoint = {
        str(row["endpoint_item_id"]): list(row["origin_users"])
        for row in ledger_rows
        if int(row["n_independent_origins"]) >= int(p2["two_hop_support_threshold"])
    }
    anchors_by_item: Dict[str, list[str]] = {}
    for source in sources:
        source_user = str(source["source_user_id"])
        for item_id in source["anchor_item_ids"]:
            anchors_by_item.setdefault(str(item_id), []).append(source_user)
    for values in anchors_by_item.values():
        values.sort()
    prepared = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
            "p1_manifest": {"path": str(p1_manifest_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(p1_manifest_path)},
            "reviews": p0["inputs"]["reviews"],
            "metadata": p0["inputs"]["metadata"],
        },
        "protocol": {
            "candidate_pool": "items observed strictly before global training cutoff",
            "candidate_sampling": "deterministic uniform negatives; gold plus nine unknown pre-cutoff items",
            "user_history": "up to six reviews with timestamp strictly less than target",
            "metadata_join": "exact review-title to metadata-title only",
            "oracle_scope": "two-hop overlay may use gold endpoint after candidate construction; no source packet/route is target-aware",
        },
        "source_packets": sources,
        "origins_by_endpoint_support_gte_threshold": origins_by_endpoint,
        "anchors_by_item": anchors_by_item,
        "item_info": item_info,
        "events": prepared_events,
    }
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return prepared


def packet_prompt(packet: Mapping[str, Any]) -> tuple[list[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    history = "\n".join(
        f"- Item {row['item_id']}: {row['review_summary']} {row['review_text']}" for row in packet["semantic_history"]
    )
    prompt = f"""You compress a reader's past reviews into one reusable, factual preference-memory packet for a recommender.
Do not recommend items. Do not mention user IDs. Infer only preferences supported by the review text; hedge when evidence is weak.

Past reviews (all before the packet time):
{history}

Return a short memory (at most 80 words) and at most five concise preference facets."""
    schema = {
        "memory": {"type": "string"},
        "facets": {"type": "array", "items": {"type": "string"}},
    }
    return [{"role": "user", "content": prompt}], schema


def stage_r_prompt(event: Mapping[str, Any]) -> tuple[list[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    history = "\n".join(
        f"- {row['title'] or row['item_id']}: {row['review_summary']} {row['review_text']}" for row in event["history"]
    ) or "(No usable prior review text.)"
    prompt = f"""You are the frozen retrieval stage of a book recommender. Infer the reader's current preferences only from the reviews before the target time below. Do not use or guess the next item.

Reader history:
{history}

Return at most five supported preference facets. Each confidence must be from 0 to 1."""
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
    event: Mapping[str, Any], facets: Sequence[Mapping[str, Any]], memories: Mapping[str, str]
) -> tuple[list[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    facet_text = "\n".join(f"- {row['facet']} (confidence {float(row['confidence']):.2f})" for row in facets) or "(No facets recovered.)"
    candidate_lines = []
    for label, item_id in zip(LETTERS, event["candidate_item_ids"]):
        candidate_lines.append(f"{label}. {item_id}: {memories[item_id]}")
    prompt = f"""You are a frozen listwise book recommender. Rank all ten candidates by how likely this reader is to review next, using the fixed preference facets and candidate memories. Do not infer relevance from memory length.

Preference facets:
{facet_text}

Candidates:
{chr(10).join(candidate_lines)}

Return JSON with `ranking` only: every label A through J exactly once, best to
worst. Do not include a rationale or any other field."""
    schema = {
        "ranking": {"type": "array", "items": {"type": "string"}},
    }
    return [{"role": "user", "content": prompt}], schema


def overlay_memories(
    event: Mapping[str, Any],
    item_info: Mapping[str, Mapping[str, str]],
    packet_outputs: Mapping[str, Mapping[str, Any]],
    anchors_by_item: Mapping[str, Sequence[str]],
    origins_by_endpoint: Mapping[str, Sequence[str]],
    *,
    arm: str,
    packet_cap: int,
) -> Dict[str, str]:
    memories = {item_id: item_info[item_id]["base_memory"] for item_id in event["candidate_item_ids"]}
    if arm == "local":
        return memories
    if arm == "one_hop":
        targets = {item_id: anchors_by_item.get(item_id, ()) for item_id in memories}
        heading = "Direct collaborative packets"
    elif arm == "oracle_two_hop":
        gold = str(event["gold_item_id"])
        targets = {item_id: (origins_by_endpoint.get(item_id, ()) if item_id == gold else ()) for item_id in memories}
        heading = "Oracle two-hop community packets"
    else:
        raise ValueError(f"unknown P2 arm: {arm}")
    for item_id, origins in targets.items():
        packet_text = [text(str(packet_outputs[source]["memory"]), 180) for source in origins if source in packet_outputs][:packet_cap]
        if packet_text:
            memories[item_id] = text(memories[item_id] + "\n" + heading + ": " + " | ".join(packet_text), 600)
    return memories


def validate_ranking(response: Mapping[str, Any], candidate_ids: Sequence[str]) -> list[str]:
    ranking_labels = response.get("ranking")
    if not isinstance(ranking_labels, list) or set(ranking_labels) != set(LETTERS) or len(ranking_labels) != len(LETTERS):
        raise ValueError("reranker must return every label A-J exactly once")
    by_label = dict(zip(LETTERS, candidate_ids))
    return [by_label[label] for label in ranking_labels]


@dataclass
class Journal:
    attempts_path: Path
    calls_path: Path
    maximum: int
    primary_limit: int
    retry_limit: int
    # ``reserve`` writes an attempt while holding this lock.  It must be
    # re-entrant because ``append_jsonl`` serializes the file write too.
    lock: threading.RLock
    attempts: int = 0
    primary_attempts: int = 0
    retry_attempts: int = 0
    completed: Dict[str, Dict[str, Any]] | None = None

    @classmethod
    def load(cls, attempts_path: Path, calls_path: Path, *, maximum: int, primary_limit: int, retry_limit: int) -> "Journal":
        attempts: list[Dict[str, Any]] = read_jsonl(attempts_path) if attempts_path.exists() else []
        calls: list[Dict[str, Any]] = read_jsonl(calls_path) if calls_path.exists() else []
        completed = {str(row["key"]): row for row in calls if row.get("status") == "success"}
        return cls(
            attempts_path=attempts_path,
            calls_path=calls_path,
            maximum=maximum,
            primary_limit=primary_limit,
            retry_limit=retry_limit,
            lock=threading.RLock(),
            attempts=len(attempts),
            primary_attempts=sum(not bool(row.get("retry")) for row in attempts),
            retry_attempts=sum(bool(row.get("retry")) for row in attempts),
            completed=completed,
        )

    def reserve(self, key: str, kind: str, prompt_hash: str, *, retry: bool) -> bool:
        with self.lock:
            if self.attempts >= self.maximum:
                return False
            if retry and self.retry_attempts >= self.retry_limit:
                return False
            if not retry and self.primary_attempts >= self.primary_limit:
                return False
            row = {
                "attempt_index": self.attempts + 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "key": key,
                "kind": kind,
                "prompt_sha256": prompt_hash,
                "retry": retry,
                "status": "started",
            }
            append_jsonl(self.attempts_path, row, self.lock)
            self.attempts += 1
            self.primary_attempts += int(not retry)
            self.retry_attempts += int(retry)
            return True


def call_phase(
    *,
    jobs: Sequence[tuple[str, str, list[Dict[str, str]], Dict[str, Dict[str, Any]], Callable[[Mapping[str, Any]], Any]]],
    client: LLMClient,
    journal: Journal,
    workers: int,
    max_tokens: int,
) -> Dict[str, Any]:
    """Run one phase; each job's parser turns API JSON into its cache value."""
    pending = [job for job in jobs if job[0] not in (journal.completed or {})]
    values: Dict[str, Any] = {
        key: row["value"] for key, row in (journal.completed or {}).items() if "value" in row
    }

    def invoke(job: tuple[str, str, list[Dict[str, str]], Dict[str, Dict[str, Any]], Callable[[Mapping[str, Any]], Any]], retry: bool) -> tuple[str, bool, Any]:
        key, kind, messages, schema, parser = job
        prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest()
        if not journal.reserve(key, kind, prompt_hash, retry=retry):
            return key, False, "request budget exhausted"
        try:
            response = client.generate_json(messages, schema, temperature=0.0, max_tokens=max_tokens, max_retries=1)
            value = parser(response)
            append_jsonl(
                journal.calls_path,
                {"key": key, "kind": kind, "status": "success", "value": value, "response": response},
                journal.lock,
            )
            with journal.lock:
                if journal.completed is None:
                    journal.completed = {}
                journal.completed[key] = {"key": key, "kind": kind, "status": "success", "value": value}
            return key, True, value
        except Exception as exc:  # Error is recorded; any retry is byte-identical.
            append_jsonl(journal.calls_path, {"key": key, "kind": kind, "status": "error", "error": repr(exc)}, journal.lock)
            return key, False, repr(exc)

    # Submit only one small batch at a time.  A malformed schema/output is a
    # deterministic configuration failure, not a reason to burn the rest of
    # the phase budget while hundreds of already-queued jobs repeat it.
    for start in range(0, len(pending), workers):
        batch = pending[start : start + workers]
        primary: list[tuple[Any, bool, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(invoke, job, False): job for job in batch}
            for future in as_completed(futures):
                job = futures[future]
                key, success, value = future.result()
                primary.append((job, success, value))
                if success:
                    values[key] = value
        failures = [job for job, success, _ in primary if not success]
        if not failures:
            continue
        retried: list[tuple[Any, bool, Any]] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(failures))) as executor:
            futures = {executor.submit(invoke, job, True): job for job in failures}
            for future in as_completed(futures):
                job = futures[future]
                key, success, value = future.result()
                retried.append((job, success, value))
                if success:
                    values[key] = value
        failures = [job for job, success, _ in retried if not success]
        if failures:
            failed_keys = ", ".join(job[0] for job in failures[:5])
            raise RuntimeError(f"phase failed after identical retry for {len(failures)} request(s): {failed_keys}")
    return values


def build_jobs(prepared: Mapping[str, Any], *, phase: str, packet_outputs: Mapping[str, Mapping[str, Any]] | None = None, stage_r_outputs: Mapping[str, Mapping[str, Any]] | None = None, p2: Mapping[str, Any] | None = None) -> list[tuple[str, str, list[Dict[str, str]], Dict[str, Dict[str, Any]], Callable[[Mapping[str, Any]], Any]]]:
    jobs = []
    if phase == "packet":
        for packet in prepared["source_packets"]:
            key = f"packet:{packet['source_user_id']}"
            messages, schema = packet_prompt(packet)
            jobs.append((key, "packet", messages, schema, lambda response: response))
    elif phase == "stage_r":
        for event in prepared["events"]:
            key = f"stage_r:{event['user_id']}:{event['timestamp']}"
            messages, schema = stage_r_prompt(event)
            jobs.append((key, "stage_r", messages, schema, lambda response: response))
    elif phase == "rerank":
        if packet_outputs is None or stage_r_outputs is None or p2 is None:
            raise ValueError("rerank jobs require packet and Stage-R outputs")
        for event in prepared["events"]:
            stage_key = f"stage_r:{event['user_id']}:{event['timestamp']}"
            facets = stage_r_outputs[stage_key].get("facets", [])
            for arm in ARMS:
                memories = overlay_memories(
                    event,
                    prepared["item_info"],
                    {key.removeprefix("packet:"): value for key, value in packet_outputs.items()},
                    prepared["anchors_by_item"],
                    prepared["origins_by_endpoint_support_gte_threshold"],
                    arm=arm,
                    packet_cap=int(p2["overlay_packet_cap"]),
                )
                messages, schema = rerank_prompt(event, facets, memories)
                key = f"rerank:{arm}:{event['user_id']}:{event['timestamp']}"
                jobs.append((key, "rerank", messages, schema, lambda response, candidates=event["candidate_item_ids"]: validate_ranking(response, candidates)))
    else:
        raise ValueError(f"unknown phase {phase}")
    return jobs


def summarize(prepared: Mapping[str, Any], rerank_outputs: Mapping[str, Sequence[str]], client: LLMClient, journal: Journal) -> Dict[str, Any]:
    values: Dict[str, Any] = {"arms": {}, "request_journal": {"attempts": journal.attempts, "primary_attempts": journal.primary_attempts, "retry_attempts": journal.retry_attempts}, "token_stats": client.get_token_stats()}
    for arm in ARMS:
        ndcgs = []
        hits = []
        for event in prepared["events"]:
            key = f"rerank:{arm}:{event['user_id']}:{event['timestamp']}"
            ranking = rerank_outputs[key]
            ndcgs.append(event_ndcg_at_5(ranking, str(event["gold_item_id"])))
            hits.append(event_hit_at_5(ranking, str(event["gold_item_id"])))
        values["arms"][arm] = {"n_events": len(ndcgs), "ndcg_at_5": sum(ndcgs) / len(ndcgs), "hit_at_5": sum(hits) / len(hits), "per_event_ndcg_at_5": ndcgs}
    base = values["arms"]["local"]["per_event_ndcg_at_5"]
    for arm in ("one_hop", "oracle_two_hop"):
        delta = [value - reference for value, reference in zip(values["arms"][arm]["per_event_ndcg_at_5"], base)]
        values["arms"][arm]["delta_ndcg_at_5_vs_local"] = sum(delta) / len(delta)
    return values


def run_phases(
    prepared: Mapping[str, Any],
    *,
    p2: Mapping[str, Any],
    client: LLMClient,
    journal: Journal,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    workers = int(p2["workers"])
    packet_outputs = call_phase(
        jobs=build_jobs(prepared, phase="packet"),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(p2["packet_max_tokens"]),
    )
    stage_r_outputs = call_phase(
        jobs=build_jobs(prepared, phase="stage_r"),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(p2["stage_r_max_tokens"]),
    )
    rerank_outputs = call_phase(
        jobs=build_jobs(prepared, phase="rerank", packet_outputs=packet_outputs, stage_r_outputs=stage_r_outputs, p2=p2),
        client=client,
        journal=journal,
        workers=workers,
        max_tokens=int(p2["rerank_max_tokens"]),
    )
    return packet_outputs, stage_r_outputs, rerank_outputs


def assert_smoke_complete(
    *,
    smoke_manifest_path: Path,
    config_path: Path,
    prepared_path: Path,
    run_id: str,
    packets: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    completed: Mapping[str, Mapping[str, Any]],
) -> None:
    if not smoke_manifest_path.exists():
        raise RuntimeError("full P2 run is blocked: first run --smoke-only and inspect its manifest")
    recorded = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_sha256": sha256_file(config_path),
        "prepared_sha256": sha256_file(prepared_path),
        "packet_source_ids": [str(packet["source_user_id"]) for packet in packets],
        "event_keys": [event_key(event) for event in events],
    }
    for field in ("schema_version", "run_id", "config_sha256", "prepared_sha256", "packet_source_ids", "event_keys"):
        if recorded.get(field) != expected.get(field):
            raise RuntimeError(f"full P2 run is blocked: smoke manifest {field} does not match current inputs")
    required_keys = [f"packet:{packet['source_user_id']}" for packet in packets]
    for event in events:
        required_keys.append(f"stage_r:{event_key(event)}")
        required_keys.extend(f"rerank:{arm}:{event_key(event)}" for arm in ARMS)
    missing = [key for key in required_keys if key not in completed]
    if missing:
        raise RuntimeError(f"full P2 run is blocked: smoke journal is incomplete ({len(missing)} cached output(s) missing)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2 bounded temporal item-memory oracle")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/pilot.yaml")
    parser.add_argument("--prepare-only", action="store_true", help="materialize candidate sets/history only; no LLM request")
    parser.add_argument("--smoke-only", action="store_true", help="run the locked small smoke cohort before a full request run")
    parser.add_argument("--request", action="store_true", help="make the explicitly budgeted LLM requests")
    parser.add_argument("--force-prepare", action="store_true", help="replace the derived prepared artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum(bool(value) for value in (args.prepare_only, args.smoke_only, args.request)) > 1:
        raise SystemExit("choose at most one of --prepare-only, --smoke-only, and --request")
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    prepared = prepare(config, config_path, force=args.force_prepare)
    p2 = config["p2"]
    if not args.request and not args.smoke_only:
        print(f"P2 prepared {len(prepared['events'])} evaluation events and {len(prepared['source_packets'])} source packets; LLM requests made=0")
        print("Use --smoke-only before --request to start a locked P2 pilot.")
        return
    project_env = PROJECT_ROOT / ".env"
    load_dotenv(project_env)
    if args.request and project_path(p2["manifest"]).exists():
        raise RuntimeError("this P2 run already has a completion manifest; do not resume or overwrite its metrics")
    client = LLMClient(provider_name="azure_openai")
    budget = config["llm_budget"]
    primary = int(budget["semantic_packet_requests"]) + int(budget["evaluation_events"]) * (int(budget["shared_stage_r_requests_per_event"]) + int(budget["rerank_arms"]))
    journal = Journal.load(
        project_path(p2["attempts"]),
        project_path(p2["calls"]),
        maximum=int(budget["max_total_requests"]),
        primary_limit=primary,
        retry_limit=int(budget["retry_reserve_requests"]),
    )
    if journal.attempts and p2.get("block_existing_journal_reason"):
        raise RuntimeError(str(p2["block_existing_journal_reason"]))
    if journal.attempts > int(budget["max_total_requests"]):
        raise RuntimeError("existing journal already exceeds the hard request cap")
    smoke_packets, smoke_events = smoke_subset(prepared, p2)
    smoke_manifest_path = project_path(p2["smoke_manifest"])
    if args.smoke_only:
        smoke_prepared = {**prepared, "source_packets": smoke_packets, "events": smoke_events}
        run_phases(smoke_prepared, p2=p2, client=client, journal=journal)
        payload = smoke_manifest_payload(
            config_path=config_path,
            prepared_path=project_path(p2["prepared"]),
            run_id=str(p2["run_id"]),
            packets=smoke_packets,
            events=smoke_events,
            journal=journal,
        )
        smoke_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"smoke": "pass", "request_attempts": journal.attempts, "packet_sources": len(smoke_packets), "evaluation_events": len(smoke_events)}, ensure_ascii=False))
        return
    assert_smoke_complete(
        smoke_manifest_path=smoke_manifest_path,
        config_path=config_path,
        prepared_path=project_path(p2["prepared"]),
        run_id=str(p2["run_id"]),
        packets=smoke_packets,
        events=smoke_events,
        completed=journal.completed or {},
    )
    packet_outputs, stage_r_outputs, rerank_outputs = run_phases(prepared, p2=p2, client=client, journal=journal)
    metrics = summarize(prepared, rerank_outputs, client, journal)
    metrics.update({"schema_version": SCHEMA_VERSION, "run_id": p2["run_id"], "created_at_utc": datetime.now(timezone.utc).isoformat()})
    metrics_path = project_path(p2["metrics"])
    manifest_path = project_path(p2["manifest"])
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": p2["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path)},
        "prepared": {"path": str(project_path(p2["prepared"]).relative_to(PROJECT_ROOT)), "sha256": sha256_file(project_path(p2["prepared"]))},
        "attempts": {"path": str(project_path(p2["attempts"]).relative_to(PROJECT_ROOT)), "records": journal.attempts},
        "calls": {"path": str(project_path(p2["calls"]).relative_to(PROJECT_ROOT))},
        "metrics": {"path": str(metrics_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(metrics_path)},
        "request_cap": int(budget["max_total_requests"]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"request_attempts": journal.attempts, "metrics": metrics["arms"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
