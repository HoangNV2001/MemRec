"""Standalone support code for the active temporal transition-PPR experiment.

This module deliberately owns the small amount of data preparation, ranking,
journalling, and self-hosted runner logic needed by P7.  It prevents the active
method from depending on implementations of the retired P1--P6 experiments.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import itertools
import json
import os
import random
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Mapping, Sequence

from src.temporal_common.metrics import (
    event_hit_at_5,
    event_ndcg_at_5,
    paired_bootstrap_ci,
    rank_by_scores,
)
from src.temporal_books.common import project_path, set_csv_field_limit, stable_order


LETTERS = tuple("ABCDEFGHIJ")
ScoredEvent = tuple[int, str, float]


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


def event_key(event: Mapping[str, Any]) -> str:
    return f"{event['user_id']}:{event['timestamp']}"


def successful_calls(path: Path) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") != "success":
            continue
        key = str(row["key"])
        if key in values:
            raise ValueError(f"duplicate frozen call: {key}")
        values[key] = row
    return values


def stable_sample_candidates(
    pool: Sequence[str], known_items: set[str], gold_item_id: str, *, n_candidates: int, seed_key: str
) -> list[str]:
    if n_candidates != len(LETTERS):
        raise ValueError("the listwise ranker requires exactly ten candidates")
    if gold_item_id not in pool:
        raise ValueError("gold item is not available in the pre-cutoff candidate pool")
    rng = random.Random(stable_order(seed_key))
    negatives: list[str] = []
    blocked = {*known_items, gold_item_id}
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


def candidate_base_memory(title_value: str, metadata: Mapping[str, str], cap: int) -> str:
    pieces = [title_value]
    if metadata.get("description"):
        pieces.append(metadata["description"])
    if metadata.get("authors"):
        pieces.append("Authors: " + metadata["authors"])
    if metadata.get("categories"):
        pieces.append("Categories: " + metadata["categories"])
    return text(". ".join(pieces), cap)


def load_positive_graph(
    config: Mapping[str, Any],
) -> tuple[Dict[str, list[ScoredEvent]], Dict[str, list[tuple[int, str, float]]]]:
    """Load positive interactions; callers still enforce strict target-time filters."""
    minimum = float(config["p7"]["positive_rating_min"])
    histories: DefaultDict[str, list[ScoredEvent]] = defaultdict(list)
    item_events: DefaultDict[str, list[tuple[int, str, float]]] = defaultdict(list)
    set_csv_field_limit()
    with project_path(config["dataset"]["reviews_file"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = (row.get("User_id") or "").strip()
            item_id = (row.get("Id") or "").strip()
            if not user_id or not item_id:
                continue
            try:
                timestamp, rating = int(row["review/time"]), float(row["review/score"])
            except (KeyError, TypeError, ValueError):
                continue
            if rating >= minimum:
                histories[user_id].append((timestamp, item_id, rating))
                item_events[item_id].append((timestamp, user_id, rating))
    for values in histories.values():
        values.sort()
    for values in item_events.values():
        values.sort()
    return dict(histories), dict(item_events)


def first_novel_positive_targets(
    histories: Mapping[str, Sequence[ScoredEvent]],
    *,
    start: int,
    end: int | None,
    candidate_pool: set[str],
    minimum_history: int,
    excluded_users: set[str],
) -> list[Dict[str, Any]]:
    targets: list[Dict[str, Any]] = []
    for user_id, events in histories.items():
        if user_id in excluded_users:
            continue
        seen: set[str] = set()
        previous = 0
        for timestamp, batch_iter in itertools.groupby(events, key=lambda event: event[0]):
            batch = list(batch_iter)
            in_window = timestamp >= start and (end is None or timestamp < end)
            eligible = sorted(item_id for _, item_id, _ in batch if item_id not in seen and item_id in candidate_pool)
            if in_window and previous >= minimum_history and eligible:
                targets.append({"user_id": user_id, "timestamp": timestamp, "gold_item_id": eligible[0]})
                break
            if end is None or timestamp < end:
                seen.update(item_id for _, item_id, _ in batch)
                previous += len(batch)
    return targets


def attach_candidates(
    targets: Sequence[Mapping[str, Any]],
    histories: Mapping[str, Sequence[ScoredEvent]],
    candidate_pool: Sequence[str],
    *,
    n_candidates: int,
    prefix: str,
) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    for target in targets:
        user_id, timestamp, gold = str(target["user_id"]), int(target["timestamp"]), str(target["gold_item_id"])
        history = histories[user_id]
        end = bisect.bisect_left(history, (timestamp, "", float("-inf")))
        known = {item_id for _, item_id, _ in history[:end]}
        candidates = stable_sample_candidates(
            candidate_pool,
            known,
            gold,
            n_candidates=n_candidates,
            seed_key=f"{prefix}\0{user_id}\0{timestamp}\0{gold}",
        )
        events.append({**target, "candidate_item_ids": candidates})
    return events


def add_test_prompt_context(
    events: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    by_user = {str(event["user_id"]): int(event["timestamp"]) for event in events}
    needed_items = {str(item_id) for event in events for item_id in event["candidate_item_ids"]}
    titles: Dict[str, str] = {}
    histories: DefaultDict[str, list[tuple[int, str, str, str, str]]] = defaultdict(list)
    set_csv_field_limit()
    with project_path(config["dataset"]["reviews_file"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = (row.get("User_id") or "").strip()
            item_id = (row.get("Id") or "").strip()
            if not item_id:
                continue
            title = (row.get("Title") or "").strip()
            if item_id in needed_items and title and item_id not in titles:
                titles[item_id] = title
            target_time = by_user.get(user_id)
            if target_time is None:
                continue
            try:
                timestamp = int(row["review/time"])
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp < target_time:
                histories[user_id].append(
                    (timestamp, item_id, title, row.get("review/summary") or "", row.get("review/text") or "")
                )
    needed_titles = {titles.get(item_id, "") for item_id in needed_items} - {""}
    metadata = load_metadata(project_path(config["dataset"]["metadata_file"]), needed_titles)
    enriched: list[Dict[str, Any]] = []
    for event in events:
        recent = sorted(histories[str(event["user_id"])])[-6:]
        enriched.append(
            {
                **event,
                "history": [
                    {
                        "timestamp": timestamp,
                        "item_id": item_id,
                        "title": title,
                        "review_summary": text(summary, 180),
                        "review_text": text(review, 360),
                    }
                    for timestamp, item_id, title, summary, review in recent
                ],
            }
        )
    item_info = {
        item_id: {
            "title": titles.get(item_id, f"Item {item_id}"),
            "base_memory": candidate_base_memory(
                titles.get(item_id, f"Item {item_id}"), metadata.get(titles.get(item_id, ""), {}), 420
            ),
        }
        for item_id in needed_items
    }
    return enriched, item_info


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
    facet_text = "\n".join(
        f"- {row['facet']} (confidence {float(row['confidence']):.2f})" for row in facets
    ) or "(No facets recovered.)"
    candidate_lines = [
        f"{label}. {item_id}: {memories[item_id]}" for label, item_id in zip(LETTERS, event["candidate_item_ids"])
    ]
    prompt = f"""You are a frozen listwise book recommender. Rank all ten candidates by how likely this reader is to review next, using the fixed preference facets and candidate memories. Do not infer relevance from memory length.

Preference facets:
{facet_text}

Candidates:
{chr(10).join(candidate_lines)}

Return JSON with `ranking` only: every label A through J exactly once, best to
worst. Do not include a rationale or any other field."""
    return [{"role": "user", "content": prompt}], {"ranking": {"type": "array", "items": {"type": "string"}}}


@dataclass
class Journal:
    attempts_path: Path
    calls_path: Path
    maximum: int
    primary_limit: int
    retry_limit: int
    lock: threading.RLock
    attempts: int = 0
    primary_attempts: int = 0
    retry_attempts: int = 0
    completed: Dict[str, Dict[str, Any]] | None = None

    @classmethod
    def load(
        cls, attempts_path: Path, calls_path: Path, *, maximum: int, primary_limit: int, retry_limit: int
    ) -> "Journal":
        attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
        calls = read_jsonl(calls_path) if calls_path.exists() else []
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
            append_jsonl(
                self.attempts_path,
                {
                    "attempt_index": self.attempts + 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "key": key,
                    "kind": kind,
                    "prompt_sha256": prompt_hash,
                    "retry": retry,
                    "status": "started",
                },
                self.lock,
            )
            self.attempts += 1
            self.primary_attempts += int(not retry)
            self.retry_attempts += int(retry)
            return True


Job = tuple[
    str,
    str,
    list[Dict[str, str]],
    Dict[str, Dict[str, Any]],
    Callable[[Mapping[str, Any]], Any],
]


def call_phase(*, jobs: Sequence[Job], client: Any, journal: Journal, workers: int, max_tokens: int) -> Dict[str, Any]:
    pending = [job for job in jobs if job[0] not in (journal.completed or {})]
    values = {key: row["value"] for key, row in (journal.completed or {}).items() if "value" in row}

    def invoke(job: Job, retry: bool) -> tuple[str, bool, Any]:
        key, kind, messages, schema, parser = job
        prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest()
        if not journal.reserve(key, kind, prompt_hash, retry=retry):
            return key, False, "request budget exhausted"
        try:
            response = client.generate_json(messages, schema, temperature=0.0, max_tokens=max_tokens, max_retries=1)
            value = parser(response)
            row = {"key": key, "kind": kind, "status": "success", "value": value, "response": response}
            append_jsonl(journal.calls_path, row, journal.lock)
            with journal.lock:
                if journal.completed is None:
                    journal.completed = {}
                journal.completed[key] = row
            return key, True, value
        except Exception as exc:
            append_jsonl(
                journal.calls_path,
                {"key": key, "kind": kind, "status": "error", "error": repr(exc)},
                journal.lock,
            )
            return key, False, repr(exc)

    for start in range(0, len(pending), workers):
        batch = pending[start : start + workers]
        primary: list[tuple[Job, bool, Any]] = []
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
        retried: list[tuple[Job, bool, Any]] = []
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


def resource_snapshot() -> Dict[str, Any]:
    expected = os.environ.get("MEMREC_EXPECTED_SLURM_JOB_ID")
    actual = os.environ.get("SLURM_JOB_ID")
    if not expected or actual != expected:
        raise RuntimeError(f"expected Slurm job {expected!r}, running in {actual!r}")
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1:
        raise RuntimeError("transition-PPR local baseline requires exactly one visible GPU")
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
        raise ValueError("LLM smoke must contain 20-30 events")
    # This salt is historical experiment state and must remain byte-identical.
    return sorted(
        prepared["test_events"], key=lambda event: stable_order(f"p6-llm-smoke\0{event_key(event)}")
    )[:count]


def stage_jobs(events: Sequence[Mapping[str, Any]]) -> list[Job]:
    jobs: list[Job] = []
    for event in events:
        messages, schema = stage_r_prompt(event)
        jobs.append((f"stage_r:{event_key(event)}", "stage_r", messages, schema, lambda response: response))
    return jobs


def required_keys(events: Sequence[Mapping[str, Any]]) -> list[str]:
    return [key for event in events for key in (f"stage_r:{event_key(event)}", f"rerank:local:{event_key(event)}")]
