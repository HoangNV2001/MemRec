"""CE0 — materialize candidate-conditioned evidence from frozen 1-hop context.

This is a ranking-time retrieval pilot, not another graph-expansion experiment.
It reads only the MH0 1-hop control and materializes two deterministic evidence
views for each locked validation user:

* request evidence, selected from 1-hop snippets using the request text; and
* candidate evidence, selected independently for each visible candidate using
  that candidate's title/memory plus the request.

The candidate selector never receives the gold item ID or ranking outcome.  It
does see candidate text because it is intentionally a *ranking-time*,
candidate-conditioned mechanism.  No selected evidence is written back to the
graph or supplied to Stage-R.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from .mh0 import PROJECT_ROOT, canonical_hash, git_sha, project_path, safe_relative, sha256
from .mh2 import iter_jsonl


SCHEMA_VERSION = 1
_WORD_RE = re.compile(r"[^\W\d_]{3,}", flags=re.UNICODE)
_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "before", "being", "but", "can",
    "each", "for", "from", "have", "into", "its", "more", "not", "of", "only", "or", "our", "out", "that",
    "the", "their", "them", "then", "these", "this", "those", "through", "under", "very", "was", "were",
    "what", "when", "which", "while", "with", "would", "your", "you", "book", "books", "item", "items",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize candidate-conditioned 1-hop evidence")
    parser.add_argument("--config", default="configs/multihop/mh0_books.yaml")
    parser.add_argument("--limit", type=int, default=None, help="override candidate_evidence.pilot_users")
    parser.add_argument("--force", action="store_true", help="replace CE0 artifact after validating inputs")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def words(value: Any) -> set[str]:
    return {word for word in _WORD_RE.findall(str(value).lower()) if word not in _STOPWORDS}


def lexical_score(query_terms: set[str], text: str) -> float:
    """Length-normalized lexical overlap, stable under tied scores."""
    text_terms = words(text)
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / math.sqrt(len(query_terms) * len(text_terms))


def truncate_words(text: str, limit: int) -> str:
    tokens = str(text).split()
    if limit < 1:
        raise ValueError("word limit must be positive")
    return " ".join(tokens[:limit])


def estimated_tokens(rows: Sequence[Mapping[str, str]]) -> int:
    serialized = "\n".join(f"[{row['node_id']}] {row['text']}" for row in rows)
    return math.ceil(len(serialized) / 4)


def fit_token_budget(rows: Sequence[Mapping[str, str]], budget: int) -> List[Dict[str, str]]:
    """Shorten evidence deterministically until its serialized estimate fits."""
    if budget < 1:
        raise ValueError("evidence token budget must be positive")
    fitted = [{"node_id": str(row["node_id"]), "text": str(row["text"])} for row in rows]
    while fitted and estimated_tokens(fitted) > budget:
        longest = max(range(len(fitted)), key=lambda index: (len(fitted[index]["text"].split()), -index))
        tokens = fitted[longest]["text"].split()
        if len(tokens) <= 1:
            fitted.pop(longest)
        else:
            fitted[longest]["text"] = " ".join(tokens[:-1])
    if estimated_tokens(fitted) > budget:
        raise AssertionError("unable to enforce evidence budget")
    return fitted


def rank_nodes(node_snippets: Mapping[str, str], query: str) -> List[str]:
    query_terms = words(query)
    # dict insertion order is the frozen 1-hop/pruner order and is the tie-break.
    return sorted(node_snippets, key=lambda node_id: -lexical_score(query_terms, node_snippets[node_id]))


def materialize_evidence(
    node_snippets: Mapping[str, str],
    *,
    query: str,
    n_rows: int,
    snippet_words: int,
    token_budget: int,
) -> List[Dict[str, str]]:
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    ranked = rank_nodes(node_snippets, query)
    if len(ranked) < n_rows:
        raise ValueError(f"insufficient frozen 1-hop nodes: expected {n_rows}, got {len(ranked)}")
    rows = [
        {"node_id": node_id, "text": truncate_words(node_snippets[node_id], snippet_words)}
        for node_id in ranked[:n_rows]
    ]
    fitted = fit_token_budget(rows, token_budget)
    if len(fitted) != n_rows:
        raise ValueError("evidence budget is too small to preserve the registered evidence-slot count")
    return fitted


def candidate_query(instruction: str, candidate_title: str, candidate_memory: str) -> str:
    # Candidate texts are ranking inputs, not labels/outcomes.  Do not add the
    # gold ID, item position, or information from another candidate here.
    return "\n".join((instruction, candidate_title, candidate_memory))


def evidence_row(
    control: Mapping[str, Any], *, token_budget: int, snippet_words: int, evidence_per_candidate: int
) -> Dict[str, Any]:
    stage = control["stage_r_context"]
    ranking = control["ranking_context"]
    # JSON artifacts are serialized with sorted mapping keys.  Rebuild the
    # mapping in ``selected_node_ids`` order so a lexical tie uses the frozen
    # MH0 pruner order, not alphabetical node IDs.
    raw_snippets = stage["neighbor_snippets"]
    selected_node_ids = list(stage["selected_node_ids"])
    if set(raw_snippets) != set(selected_node_ids):
        raise ValueError(f"user {control['user_id']}: selected 1-hop IDs/snippets disagree")
    node_snippets = {node_id: raw_snippets[node_id] for node_id in selected_node_ids}
    candidates = [int(candidate) for candidate in ranking["candidates"]]
    if len(candidates) != 10:
        raise ValueError(f"user {control['user_id']}: expected fixed 10 candidates")
    if len(node_snippets) < len(candidates):
        raise ValueError(f"user {control['user_id']}: fewer 1-hop nodes than candidates")
    request_rows = materialize_evidence(
        node_snippets,
        query=str(ranking.get("instruction", "")),
        n_rows=len(candidates) * evidence_per_candidate,
        snippet_words=snippet_words,
        token_budget=token_budget,
    )
    per_candidate_budget = max(1, token_budget // len(candidates))
    candidate_rows: Dict[str, List[Dict[str, str]]] = {}
    titles = ranking.get("candidate_titles", {})
    memories = ranking.get("candidate_memories", {})
    for candidate_id in candidates:
        candidate_rows[str(candidate_id)] = materialize_evidence(
            node_snippets,
            query=candidate_query(
                str(ranking.get("instruction", "")),
                str(titles.get(str(candidate_id), f"Item {candidate_id}")),
                str(memories.get(str(candidate_id), "")),
            ),
            n_rows=evidence_per_candidate,
            snippet_words=snippet_words,
            token_budget=per_candidate_budget,
        )
    candidate_flat = [row for candidate_id in candidates for row in candidate_rows[str(candidate_id)]]
    candidate_flat = fit_token_budget(candidate_flat, token_budget)
    if len(candidate_flat) != len(candidates) * evidence_per_candidate:
        raise ValueError("candidate evidence budget removed a registered slot")
    # Re-split after global fitting so exact slot equality is visible in the artifact.
    index = 0
    for candidate_id in candidates:
        candidate_rows[str(candidate_id)] = candidate_flat[index:index + evidence_per_candidate]
        index += evidence_per_candidate
    assert estimated_tokens(request_rows) <= token_budget
    assert estimated_tokens(candidate_flat) <= token_budget
    return {
        "schema_version": SCHEMA_VERSION,
        "user_id": int(control["user_id"]),
        "evaluation_contract": {
            "candidate_order_sha256": ranking["candidate_order_sha256"],
            "candidate_count": len(candidates),
            "candidate_conditioned_ranking_time_only": True,
            "gold_item_id_used_by_selector": False,
        },
        "source_one_hop": {
            "K_u": int(stage["K_u"]),
            "T_u": int(stage["T_u"]),
            "node_ids": list(node_snippets),
        },
        "request_evidence": {
            "rows": request_rows,
            "slot_count": len(request_rows),
            "estimated_tokens": estimated_tokens(request_rows),
        },
        "candidate_evidence": {
            "by_candidate": candidate_rows,
            "slots_per_candidate": evidence_per_candidate,
            "total_slot_count": len(candidate_flat),
            "estimated_tokens": estimated_tokens(candidate_flat),
            "unique_source_nodes": len({row["node_id"] for row in candidate_flat}),
        },
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    import hashlib

    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    section = config.get("candidate_evidence", {})
    controls_path = project_path(section["mh0_control_val"])
    bundles_path = project_path(section["mh1_bundles_val"])
    output_path = project_path(section["artifact"])
    manifest_path = project_path(section["manifest"])
    limit = int(args.limit if args.limit is not None else section["pilot_users"])
    token_budget = int(section["evidence_token_budget"])
    snippet_words = int(section["snippet_words"])
    evidence_per_candidate = int(section["evidence_per_candidate"])
    if limit < 1:
        raise ValueError("pilot_users must be positive")
    required = [config_path, controls_path, bundles_path]
    missing = [safe_relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("CE0 required input(s) missing: " + ", ".join(missing))
    if (output_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError("CE0 output exists; use --force only to replace a validated derived artifact")

    controls = {int(row["user_id"]): row for row in iter_jsonl(controls_path)}
    eligible = []
    for bundle in iter_jsonl(bundles_path):
        if not bundle["eligibility"]["eligible_all_arms"]:
            continue
        user_id = int(bundle["user_id"])
        control = controls.get(user_id)
        if control is None:
            raise ValueError(f"eligible bundle user {user_id} lacks MH0 control")
        if bundle["evaluation_contract"]["candidate_order_sha256"] != control["ranking_context"]["candidate_order_sha256"]:
            raise ValueError(f"candidate order mismatch for user {user_id}")
        eligible.append(control)
    selected = sorted(eligible, key=lambda row: int(row["user_id"]))[:limit]
    if len(selected) != limit:
        raise ValueError(f"requested {limit} pilot users, only {len(selected)} eligible")
    rows = [
        evidence_row(
            control,
            token_budget=token_budget,
            snippet_words=snippet_words,
            evidence_per_candidate=evidence_per_candidate,
        )
        for control in selected
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count, output_hash = write_jsonl(output_path, rows)
    request_token_counts = [row["request_evidence"]["estimated_tokens"] for row in rows]
    candidate_token_counts = [row["candidate_evidence"]["estimated_tokens"] for row in rows]
    unique_candidate_nodes = [row["candidate_evidence"]["unique_source_nodes"] for row in rows]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "protocol": {
            "config": {"path": safe_relative(config_path), "sha256": sha256(config_path)},
            "mh0_control_val": {"path": safe_relative(controls_path), "sha256": sha256(controls_path)},
            "mh1_bundles_val": {"path": safe_relative(bundles_path), "sha256": sha256(bundles_path)},
            "pilot_users": limit,
            "evidence_token_budget": token_budget,
            "snippet_words": snippet_words,
            "evidence_per_candidate": evidence_per_candidate,
            "selector": "length-normalized lexical overlap; frozen 1-hop order breaks ties",
            "selection_scope": "ranking-time only; candidate title/memory + request; never gold/outcome",
        },
        "output": {"path": safe_relative(output_path), "records": count, "sha256": output_hash},
        "audit": {
            "request_estimated_tokens": {"min": min(request_token_counts), "max": max(request_token_counts)},
            "candidate_estimated_tokens": {"min": min(candidate_token_counts), "max": max(candidate_token_counts)},
            "candidate_unique_source_nodes": {"min": min(unique_candidate_nodes), "max": max(unique_candidate_nodes)},
            "users": [int(row["user_id"]) for row in rows],
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"CE0 complete: {count} users -> {safe_relative(output_path)}")
    print(f"  artifact sha256: {output_hash}")
    print(f"  request/candidate evidence estimates: {min(request_token_counts)}–{max(request_token_counts)} / {min(candidate_token_counts)}–{max(candidate_token_counts)}")


if __name__ == "__main__":
    main()
