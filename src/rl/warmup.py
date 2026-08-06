"""
Test-clean memory warm-up for the RL environment.

Why this exists instead of reusing ``results/m0_memrec_full_1k/memory.jsonl``:
that file is the memory state *after* the M0 eval loop, and the main MemRec config
leaves ``enable_stage_w`` at its ``True`` default, so ground-truth clicks on the
**test** item were written into shared memory during eval. Using it as the frozen
RL graph would leak the answer into every rollout.

Warm-up writes are clean: the target is ``all_interactions[-2]`` (the valid item),
history is everything before it, and the test item is never touched. This module
replays exactly that -- the same Stage-R -> Stage-ReRank -> Stage-W sequence as
``MemRecTrainer._warmup_single_user`` -- with two changes:

1. candidates come from ``splits.sample_candidates`` (seeded per user, not per
   thread), so the warm-up is reproducible;
2. no eval loop runs afterwards, so nothing test-derived is ever written.

CPU + API only. No GPU (RL_PLAN.md M1).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Sequence

import threading

from tqdm import tqdm

from src.rl.splits import sample_candidates


def warmup_single_user(
    agent,
    user_id: int,
    n_candidates: int = 10,
    seed: int = 42,
) -> Dict:
    """
    One warm-up interaction for ``user_id``, writing M_u / M_i / neighbour patches.

    Returns a small status dict; never raises, so one bad user cannot abort a
    2000-user run.
    """
    dataset = agent.dataset
    history = dataset.get_user_history(user_id, "test")
    if len(history) <= 1:
        return {"user_id": user_id, "ok": False, "reason": "history too short"}

    # target = valid item (second from last). The test item is history[-1] and is
    # deliberately excluded from both the target and the history.
    target_item = history[-2]
    train_history = history[:-2]

    positives = set(train_history) | {target_item, history[-1]}
    negative_pool = [i for i in dataset.user_negatives.get(user_id, []) if i not in positives]
    if len(negative_pool) < n_candidates - 1:
        return {"user_id": user_id, "ok": False, "reason": "negative pool too small"}

    candidates = sample_candidates(
        user_id=user_id,
        target_item=target_item,
        negative_pool=negative_pool,
        n_candidates=n_candidates,
        seed=seed,
    )

    instruction = None
    if dataset.instructions and user_id in dataset.instructions:
        instruction = dataset.instructions[user_id].get("instruction")

    try:
        ranked_items, details = agent.rerank(
            user_id=user_id,
            candidates=candidates,
            instruction=instruction,
            return_details=True,
            debug_logger=None,
        )
        if target_item not in ranked_items:
            return {"user_id": user_id, "ok": False, "reason": "target dropped by reranker"}

        write_result = agent.write(
            user_id=user_id,
            feedback={
                "action": "CLICK",
                "item_id": target_item,
                "position": ranked_items.index(target_item),
            },
            recent_facets=details.get("facets", []),
            pruned_subgraph=details.get("pruned_subgraph", None),
            debug_logger=None,
        )
        return {
            "user_id": user_id,
            "ok": True,
            "n_applied": write_result.get("stats", {}).get("total_applied", 0),
        }
    except Exception as exc:  # noqa: BLE001 - a single user must not kill the run
        return {"user_id": user_id, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def run_warmup(
    agent,
    user_ids: Sequence[int],
    n_candidates: int = 10,
    seed: int = 42,
    n_workers: int = 16,
    desc: str = "RL warmup",
) -> List[Dict]:
    """Warm up every user in ``user_ids``, in parallel. Returns per-user results."""
    results: List[Dict] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(warmup_single_user, agent, int(uid), n_candidates, seed)
            for uid in user_ids
        ]
        with tqdm(total=len(futures), desc=desc) as pbar:
            for future in as_completed(futures):
                with lock:
                    results.append(future.result())
                pbar.update(1)

    return results


def summarize(results: Sequence[Dict]) -> Dict:
    ok = [r for r in results if r.get("ok")]
    reasons: Dict[str, int] = {}
    for r in results:
        if not r.get("ok"):
            reasons[r.get("reason", "unknown")] = reasons.get(r.get("reason", "unknown"), 0) + 1
    return {
        "n_users": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "avg_patches_applied": (sum(r.get("n_applied", 0) for r in ok) / len(ok)) if ok else 0.0,
        "failure_reasons": reasons,
    }
