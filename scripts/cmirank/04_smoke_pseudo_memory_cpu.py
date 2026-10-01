#!/usr/bin/env python3
"""20-user fake-LLM Stage-W/Stage-R wiring smoke; no ranking result or GPU.

Warm-up candidates are deterministic *smoke-only* lists, not the frozen mixed-
hardness policy sampler. This does not prove real-LM propagation safety.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.smoke_full_memrec_cpu import FakeJSONClient
from src.cmirank.policy_data import (
    NEUTRAL_PSEUDO_INSTRUCTION, draft_policy_split, pseudo_target_from_train,
)
from src.cmirank.request import RankRequest
from src.cmirank.provenance import file_sha256
from src.cmirank.snapshot import make_prefix_snapshot, snapshot_sha256
from src.data.books_protocol import books_cohorts, books_dev_cost_subset, cohort_digest
from src.data.dataset_base import RecDataset
from src.models.memrec_agent import MemRecAgent


def smoke_candidates(user_id: int, positive: int, forbidden: set[int],
                     n_items: int) -> list[int]:
    """Baseline-shaped random list for wiring only; never a research sampler."""
    rng = random.Random(42 * 1_000_003 + user_id * 100_003 + positive)
    if n_items - len(forbidden) < 9:
        raise ValueError("Not enough smoke negatives")
    negatives: set[int] = set()
    while len(negatives) < 9:
        item = rng.randrange(n_items)
        if item not in forbidden:
            negatives.add(item)
    candidates = [positive, *sorted(negatives)]
    rng.shuffle(candidates)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=20)
    args = parser.parse_args()
    if not 20 <= args.users <= 30:
        parser.error("Smoke is deliberately limited to 20–30 users")

    manifest = json.loads((ROOT / "configs/cmirank/policy_split_manifest.json")
                          .read_text(encoding="utf-8"))
    if (manifest["approval_status"] != "researcher_approved_2026-09-30"
            or manifest["training_instruction"] != NEUTRAL_PSEUDO_INSTRUCTION):
        raise ValueError("Locked policy protocol differs")
    cohorts, _ = books_cohorts(range(7377))
    _, exposed200 = books_dev_cost_subset(cohorts["all"], cohorts["dev"])
    split = draft_policy_split(cohorts["dev"], exposed200, seed=manifest["seed"])
    if any(cohort_digest(users) != manifest["split_sha256"][name]
           for name, users in split.items()):
        raise ValueError("Locked policy split differs")

    source = ROOT / "data/processed/instructrec-books/instructrec-books.inter"
    if file_sha256(source) != manifest["source_inter_sha256"]:
        raise ValueError("Interaction source differs from locked manifest")
    dataset = RecDataset(
        str(source),
        seed=42, precompute_negatives=False,
    )
    dataset.load_item_metadata()
    eligible = []
    for name in ("policy_train", "policy_val"):
        group = [user_id for user_id in split[name]
                 if pseudo_target_from_train(
                     dataset.train_data[user_id],
                     min_prefix_length=manifest["min_prefix_length"],
                 ) is not None]
        if cohort_digest(group) != manifest["eligible_sha256"][name]:
            raise ValueError(f"Approved eligibility differs: {name}")
        eligible.extend(group)
    eligible.sort()
    query_ids = eligible[:args.users]
    snapshot, warmup_events, pseudo_targets = make_prefix_snapshot(
        dataset.train_data, query_ids,
        min_prefix_length=manifest["min_prefix_length"],
        item_metadata=dataset.item_metadata,
    )
    snapshot_id = snapshot_sha256(snapshot)
    client = FakeJSONClient()
    agent = MemRecAgent(
        dataset=snapshot, llm_client=client,
        k=16, tau=1800, n_facets=7, temperature=0.0, max_tokens=4000,
        mix_min_users=4, mix_min_items=6, fanout_cap=8,
        reranker_mode="llm", pruner_mode="llm_rules",
        upstream_empty_facets_prompt=True,
    )
    if agent.storage.n_updates:
        raise AssertionError("Stage-W smoke did not start from fresh memory")

    observed_writes: list[tuple[int, int]] = []
    for user_id in query_ids:
        warmup_item = warmup_events[user_id]
        forbidden = set(dataset.train_data[user_id])
        warmup_candidates = smoke_candidates(
            user_id, warmup_item, forbidden, dataset.n_items,
        )
        if pseudo_targets[user_id] in warmup_candidates:
            raise AssertionError("Own pseudo-target entered warm-up candidates")
        _, details = agent.rerank(
            user_id, warmup_candidates, NEUTRAL_PSEUDO_INSTRUCTION,
            return_details=True,
        )
        agent.write(
            user_id=user_id,
            feedback={"action": "CLICK", "item_id": warmup_item, "position": 0},
            recent_facets=details["facets"],
            pruned_subgraph=details["pruned_subgraph"],
        )
        observed_writes.append((user_id, warmup_item))
    if observed_writes != [(u, warmup_events[u]) for u in query_ids]:
        raise AssertionError("Stage-W feedback differs from train[-2]")
    if agent.n_stage_w_calls != args.users:
        raise AssertionError("Expected one Stage-W write per smoke user")

    request_hashes = []
    for user_id in query_ids:
        pseudo_target = pseudo_targets[user_id]
        candidates = smoke_candidates(
            user_id, pseudo_target, set(dataset.train_data[user_id]),
            dataset.n_items,
        )
        _, details = agent.rerank(
            user_id, candidates, NEUTRAL_PSEUDO_INSTRUCTION,
            return_details=True, capture_rank_request=True,
            rank_snapshot_id=snapshot_id,
        )
        request = RankRequest.from_dict(details["rank_request"])
        if (request.snapshot_id != snapshot_id
                or request.instruction != NEUTRAL_PSEUDO_INSTRUCTION
                or [row["id"] for row in request.candidates] != candidates
                or "target_item_id" in details["rank_request"]):
            raise AssertionError("RankRequest capture violated pseudo protocol")
        request_hashes.append(request.sha256())
    if agent.n_stage_w_calls != args.users:
        raise AssertionError("Pseudo-evaluation accidentally wrote Stage-W")

    print(json.dumps({
        "status": "CPU_FAKE_LLM_PSEUDO_MEMORY_SMOKE_NOT_RANKING_RESULT",
        "users": args.users,
        "graph_snapshot_sha256": snapshot_id,
        "warmup_stage_w_calls": agent.n_stage_w_calls,
        "pseudo_stage_w_calls": 0,
        "stage_r_calls": agent.n_stage_r_calls,
        "stage_rerank_calls": agent.n_stage_rr_calls,
        "target_blind_rank_requests": len(request_hashes),
        "unique_rank_request_hashes": len(set(request_hashes)),
        "fake_logical_requests": client.total_requests,
        "real_llm_requests": 0,
        "real_model_propagation_proven_safe": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
