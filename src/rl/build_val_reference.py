"""
M2 Part A: cache the gpt-4o-mini side of reward validation. CPU + API, no GPU.

RL_PLAN.md §11.6 is explicit that this must exist on disk *before* renting the
H100 -- once the machine is running, the only remaining work for Validation A is
to score the same cached pairs with the frozen 1.5B ranker and correlate.

For each val user we build five arms, which together serve both validations:

| arm        | M_collab                        | serves |
|------------|---------------------------------|--------|
| `sample1`  | gpt-4o-mini, temperature 1.0    | A + B  |
| `sample2`  | gpt-4o-mini, temperature 1.0    | A      |
| `shuffled` | another user's `sample1`        | B      |
| `lorem`    | lorem-ipsum facets              | B      |
| `empty`    | none (this is the `r_null` arm) | A + B  |

Two samples per user matter: Validation A asks whether the cheap proxy *orders*
memories the way the real ranker does, so the correlation needs a spread of
memory quality per user, not one point.

The scoring side deliberately uses the repo's own ``LLMReranker`` -- that *is*
the real ``LLM_Rec``. Using a look-alike prompt would make the correlation
measure "proxy vs real" confounded with "prompt A vs prompt B". It is used
read-only; the original eval path is untouched.

    python -m src.rl.build_val_reference --config configs/rl/m1_env_books.yaml
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import argparse
import json
import sys
import threading
import time

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.models import LLMClient                       # noqa: E402
from src.models.reranker_llm import LLMReranker        # noqa: E402
from src.rl.dataset import load_records                # noqa: E402
from src.rl.policy import parse_facets                 # noqa: E402
from src.rl.reward.metrics import hit_at_k, ndcg_at_k, rank_of   # noqa: E402
from src.utils import load_config                      # noqa: E402

LOREM_FACETS = [
    {"facet": "lorem ipsum dolor sit amet consectetur adipiscing elit",
     "confidence": 0.9, "supporting_neighbors": []},
    {"facet": "sed do eiusmod tempor incididunt ut labore et dolore magna",
     "confidence": 0.8, "supporting_neighbors": []},
    {"facet": "ut enim ad minim veniam quis nostrud exercitation ullamco",
     "confidence": 0.7, "supporting_neighbors": []},
]


def parse_args():
    p = argparse.ArgumentParser(description="Cache gpt-4o-mini reward reference (M2-A)")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out", default="data/rl/m2_val_reference_books.json")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None, help="smoke test on N users")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _generate_m_collab(client: LLMClient, record: Dict, n_facets: int, temperature: float) -> List[Dict]:
    """Sample one M_collab from gpt-4o-mini using the candidate-blind RL prompt."""
    from src.rl.policy import to_chat

    raw = client.generate(
        messages=to_chat(record["prompt"]),
        temperature=temperature,
        max_tokens=1024,
    )
    parsed = parse_facets(
        raw,
        valid_node_ids=list(record.get("neighbor_snippets") or {}),
        max_facets=n_facets,
    )
    return parsed.facets


def _score(reranker: LLMReranker, record: Dict, facets: List[Dict], k: int) -> Dict:
    """Rank this user's candidates with the real LLM_Rec, given these facets."""
    candidates = [
        {"id": int(c), "title": record["candidate_titles"].get(str(c), f"Item-{c}"), "tags": []}
        for c in record["candidates"]
    ]
    item_mems = {int(c): m for c, m in (record.get("candidate_memories") or {}).items()}

    scores = reranker.rerank(
        user_id=record["user_id"],
        retrieval_bundle={"facets": facets, "vector_profile": None, "support_edges": []},
        candidates=candidates,
        item_mems=item_mems,
        instruction=record.get("instruction"),
        temperature=0.0,
        max_tokens=2048,
    )
    ranked = [s["item_id"] for s in sorted(scores, key=lambda s: s.get("score", 0), reverse=True)]
    gold = int(record["gold_item_id"])
    return {
        "ranking": ranked,
        "gold_rank": rank_of(ranked, gold),
        "ndcg_at_5": ndcg_at_k(ranked, gold, k),
        "hit_at_1": hit_at_k(ranked, gold, 1),
    }


def main():
    args = parse_args()
    out_path = PROJECT_ROOT / args.out
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} exists; rebuilding re-spends the API budget. Use --force.")

    config = load_config(args.config)
    n_facets = config.get("memrec", {}).get("n_facets", 7)
    prefix = config["rl"]["out_prefix"]
    records = load_records(str(PROJECT_ROOT / f"{prefix}_{args.split}.jsonl"))
    if args.limit:
        records = records[: args.limit]
    print(f"Loaded {len(records)} {args.split} records")

    provider = config.get("provider", {})
    gen_client = LLMClient(
        api_endpoint=provider.get("endpoint"), api_key=provider.get("api_key"),
        model=provider.get("model", "gpt-4o-mini"), provider_name=provider.get("name", "openai"),
    )
    rr_client = LLMClient(
        api_endpoint=provider.get("endpoint"), api_key=provider.get("api_key"),
        model=config.get("llm_model", "gpt-4o-mini"), provider_name=provider.get("name", "openai"),
    )
    reranker = LLMReranker(rr_client)

    t0 = time.time()

    # ---- stage 1: sample two M_collab per user ----------------------------
    print(f"\n[1/2] Sampling 2 M_collab per user (temperature 1.0), {args.workers} workers...")
    samples: Dict[int, List[List[Dict]]] = {}
    lock = threading.Lock()

    def gen(record):
        got = []
        for _ in range(2):
            try:
                got.append(_generate_m_collab(gen_client, record, n_facets, 1.0))
            except Exception as exc:  # noqa: BLE001
                print(f"  gen failed for user {record['user_id']}: {exc}")
                got.append([])
        return record["user_id"], got

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(gen, r) for r in records]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generate"):
            uid, got = fut.result()
            with lock:
                samples[uid] = got

    # shuffled arm: pair each user with the next user's first sample
    order = [r["user_id"] for r in records]
    shuffled_source = {uid: order[(i + 1) % len(order)] for i, uid in enumerate(order)}

    # ---- stage 2: score every arm with the real LLM_Rec -------------------
    print(f"\n[2/2] Scoring 5 arms per user with gpt-4o-mini...")
    jobs = []
    for r in records:
        uid = r["user_id"]
        jobs.append((uid, "sample1", samples[uid][0]))
        jobs.append((uid, "sample2", samples[uid][1]))
        jobs.append((uid, "shuffled", samples[shuffled_source[uid]][0]))
        jobs.append((uid, "lorem", LOREM_FACETS))
        jobs.append((uid, "empty", []))

    by_id = {r["user_id"]: r for r in records}
    results: Dict[int, Dict[str, Dict]] = {r["user_id"]: {} for r in records}

    def run(job):
        uid, arm, facets = job
        try:
            return uid, arm, _score(reranker, by_id[uid], facets, k=5)
        except Exception as exc:  # noqa: BLE001
            return uid, arm, {"error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="score"):
            uid, arm, res = fut.result()
            with lock:
                results[uid][arm] = res

    wall = time.time() - t0
    gen_tokens = gen_client.get_token_stats()
    rr_tokens = rr_client.get_token_stats()

    payload = {
        "meta": {
            "split": args.split,
            "n_users": len(records),
            "n_facets": n_facets,
            "model": config.get("llm_model", "gpt-4o-mini"),
            "ndcg_k": 5,
            "arms": ["sample1", "sample2", "shuffled", "lorem", "empty"],
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_seconds": round(wall, 1),
            "token_stats": {"generation": gen_tokens, "reranker": rr_tokens},
            "note": "gpt-4o-mini side of M2 Validation A/B. Scored with the repo's "
                    "own LLMReranker, i.e. the real LLM_Rec.",
        },
        "m_collab": {str(u): s for u, s in samples.items()},
        "scores": {str(u): a for u, a in results.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"\n✓ cached -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {wall/60:.1f} min)")
    _summarize(payload)


def _summarize(payload: Dict):
    """Print the headroom numbers, which are informative on their own."""
    arms = payload["meta"]["arms"]
    print(f"\n{'arm':<10} {'NDCG@5':>8} {'H@1':>8} {'n':>5}")
    print("-" * 34)
    stats = {}
    for arm in arms:
        vals = [s[arm]["ndcg_at_5"] for s in payload["scores"].values()
                if arm in s and "ndcg_at_5" in s[arm]]
        hits = [s[arm]["hit_at_1"] for s in payload["scores"].values()
                if arm in s and "hit_at_1" in s[arm]]
        if vals:
            stats[arm] = (sum(vals) / len(vals), sum(hits) / len(hits), len(vals))
            print(f"{arm:<10} {stats[arm][0]:>8.4f} {stats[arm][1]:>8.4f} {stats[arm][2]:>5d}")

    if "sample1" in stats and "empty" in stats:
        delta = stats["sample1"][0] - stats["empty"][0]
        print(f"\nheadroom of M_collab over no-memory: {delta:+.4f} NDCG@5")
        if abs(delta) < 0.01:
            print("  ⚠ memory barely moves the real ranker — reward will be near-flat.")


if __name__ == "__main__":
    main()
