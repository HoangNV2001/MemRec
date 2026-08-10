"""
Re-score the cached M2 reference arms with a DIFFERENT judge model. CPU + API.

`build_val_reference.py` generated 5 real `M_collab` per val user plus three
corrupted arms, and scored all eight with gpt-4o-mini acting as `LLM_Rec`. The
memories are the expensive half and they are already on disk, so swapping the
judge costs one scoring pass and nothing else -- no generation, no GPU.

What this answers (docs/RESULTS.md, "M2 Reward Validation"):

1. **Does collaborative memory still help a stronger `LLM_Rec`?** The whole
   project rests on `real - empty = +0.1112` NDCG@5 measured on gpt-4o-mini. A
   stronger reranker may infer the same preferences straight from the neighbour
   table, which would shrink that headroom toward zero -- the premise would fail
   for a reason that has nothing to do with the policy.
2. **Is the 80.5% within-user tie rate gpt-4o-mini's limitation, or the task's?**
   If the new judge separates far more pairs, upgrading `LLM_Rec` is worth it.
   If it ties just as often, the ceiling is the protocol (10 candidates, and
   NDCG@5 takes only 6 distinct values) and no judge upgrade can help.
3. **How well does the new judge agree with gpt-4o-mini?** That is Validation A
   with the judge in the proxy's seat: it says whether the new model could serve
   as the reward while gpt-4o-mini stays the deployed `LLM_Rec`.

### The determinism caveat -- read before trusting a small difference

`build_val_reference.py` scored at `temperature=0.0`. The gpt-5 family rejects
every temperature except the default 1.0 (verified against the live API), so a
judge from that family is **stochastic**. Two things follow:

* An arm-to-arm difference can be judge noise rather than a memory effect.
  `--repeat_arm` scores one arm N times per user to measure that noise floor
  directly, and every comparison below should be read against it.
* A stochastic judge used as an in-loop reward injects variance straight into
  GRPO's advantages, which is exactly what RL_PLAN.md §5.1 chose the
  one-forward-pass design to avoid. Measuring the noise here is what decides
  whether that is affordable.

    python -m src.rl.rescore_reference --config configs/rl/m1_env_books.yaml \
        --judge_model gpt-5.6-luna --limit 10 --repeat_arm empty --repeats 3
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

from src.models import LLMClient                              # noqa: E402
from src.models.reranker_llm import LLMReranker               # noqa: E402
from src.rl.build_val_reference import LOREM_FACETS, _score   # noqa: E402
from src.rl.dataset import load_records                       # noqa: E402
from src.utils import load_config                             # noqa: E402

# $ per 1M tokens, for the cost line only. Update when the price list moves.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (None, None),
}


def parse_args():
    p = argparse.ArgumentParser(description="re-score cached M2 arms with another judge")
    p.add_argument("--config", required=True)
    p.add_argument("--judge_model", required=True, help="e.g. gpt-5.6-luna")
    p.add_argument("--split", default="val")
    p.add_argument("--reference", default="data/rl/m2_val_reference_books.json")
    p.add_argument("--out", default=None, help="default: <reference>_<judge>.json")
    p.add_argument("--arms", nargs="+", default=None, help="subset; default all 8")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None, help="pilot on N users")
    p.add_argument("--repeat_arm", default=None,
                   help="score this arm --repeats times per user to measure judge noise")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_arm_facets(reference: Dict, records: List[Dict]) -> Dict[int, Dict[str, List[Dict]]]:
    """
    Reconstruct the exact facet list behind every arm.

    ``m_collab`` only stores the real samples; the corrupted arms were derived at
    build time. ``shuffled`` in particular took the *next* user's first sample in
    file order, so it is reproduced from the same record order rather than
    re-derived -- a different pairing would silently compare against a different
    arm than gpt-4o-mini saw.
    """
    m_collab = reference["m_collab"]
    order = [int(r["user_id"]) for r in records]
    next_user = {uid: order[(i + 1) % len(order)] for i, uid in enumerate(order)}

    out: Dict[int, Dict[str, List[Dict]]] = {}
    for uid in order:
        samples = m_collab.get(str(uid))
        if not samples:
            continue
        arms = {f"sample{i + 1}": s for i, s in enumerate(samples)}
        donor = m_collab.get(str(next_user[uid]))
        if donor:
            arms["shuffled"] = donor[0]
        arms["lorem"] = LOREM_FACETS
        arms["empty"] = []
        out[uid] = arms
    return out


def main():
    args = parse_args()
    ref_path = PROJECT_ROOT / args.reference
    out_path = PROJECT_ROOT / (args.out or
                               args.reference.replace(".json", f"_{args.judge_model}.json"))
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} exists; re-running re-spends the API budget. Use --force.")

    with open(ref_path, "r", encoding="utf-8") as f:
        reference = json.load(f)
    baseline_judge = reference["meta"].get("model", "gpt-4o-mini")

    config = load_config(args.config)
    prefix = config["rl"]["out_prefix"]
    records = load_records(str(PROJECT_ROOT / f"{prefix}_{args.split}.jsonl"))

    arm_facets = build_arm_facets(reference, records)
    users = [int(r["user_id"]) for r in records if int(r["user_id"]) in arm_facets]
    if args.limit:
        users = users[: args.limit]
    by_id = {int(r["user_id"]): r for r in records}

    arms = args.arms or reference["meta"]["arms"]
    print(f"judge:    {args.judge_model}   (baseline reference judge: {baseline_judge})")
    print(f"users:    {len(users)}    arms: {arms}")

    provider = config.get("provider", {})
    rr_client = LLMClient(
        api_endpoint=provider.get("endpoint"), api_key=provider.get("api_key"),
        model=args.judge_model, provider_name=provider.get("name", "openai"),
    )
    reranker = LLMReranker(rr_client)

    jobs = [(uid, arm, 0) for uid in users for arm in arms if arm in arm_facets[uid]]
    if args.repeat_arm:
        jobs += [(uid, args.repeat_arm, rep)
                 for uid in users for rep in range(1, args.repeats)
                 if args.repeat_arm in arm_facets[uid]]
    print(f"calls:    {len(jobs)}\n")

    results: Dict[int, Dict[str, Dict]] = {uid: {} for uid in users}
    lock = threading.Lock()

    def run(job):
        uid, arm, rep = job
        try:
            return uid, arm, rep, _score(reranker, by_id[uid], arm_facets[uid][arm], k=5)
        except Exception as exc:  # noqa: BLE001
            return uid, arm, rep, {"error": f"{type(exc).__name__}: {exc}"}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="score"):
            uid, arm, rep, res = fut.result()
            with lock:
                results[uid][arm if rep == 0 else f"{arm}#rep{rep}"] = res

    wall = time.time() - t0
    tokens = rr_client.get_token_stats()

    payload = {
        "meta": {
            "judge_model": args.judge_model,
            "baseline_judge": baseline_judge,
            "reference": args.reference,
            "split": args.split,
            "n_users": len(users),
            "arms": arms,
            "repeat_arm": args.repeat_arm,
            "repeats": args.repeats if args.repeat_arm else 0,
            "ndcg_k": 5,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_seconds": round(wall, 1),
            "token_stats": tokens,
            "note": "Same cached M_collab as the reference, scored by a different "
                    "LLM_Rec. Judges in the gpt-5 family cannot be run at "
                    "temperature 0, so their scores are stochastic -- see repeat_arm.",
        },
        "scores": {str(u): a for u, a in results.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    _report(payload, reference, args)
    print(f"\n✓ wrote {out_path}")


def _report(payload: Dict, reference: Dict, args):
    scores, ref_scores = payload["scores"], reference["scores"]
    arms = payload["meta"]["arms"]
    judge = payload["meta"]["judge_model"]

    def vals(src, arm):
        return {u: s[arm]["ndcg_at_5"] for u, s in src.items()
                if arm in s and "ndcg_at_5" in s[arm]}

    print(f"\n{'arm':<10} {'NDCG@5 ' + judge:>22} {'NDCG@5 baseline':>18} {'n':>5}")
    print("-" * 58)
    new_means = {}
    for arm in arms:
        a, b = vals(scores, arm), vals(ref_scores, arm)
        common = sorted(set(a) & set(b))
        if not common:
            continue
        ma = sum(a[u] for u in common) / len(common)
        mb = sum(b[u] for u in common) / len(common)
        new_means[arm] = ma
        print(f"{arm:<10} {ma:>22.4f} {mb:>18.4f} {len(common):>5}")

    real = [a for a in arms if a.startswith("sample")]
    if real and "empty" in new_means:
        mean_real = sum(new_means[a] for a in real if a in new_means) / len(real)
        print(f"\n  headroom  real - empty = {mean_real - new_means['empty']:+.4f}"
              f"   (gpt-4o-mini: +0.1112)")

    # judge noise: same arm, same memory, scored more than once
    if args.repeat_arm:
        arm = args.repeat_arm
        keys = [arm] + [f"{arm}#rep{i}" for i in range(1, args.repeats)]
        diffs, flips, n = [], 0, 0
        for u, s in scores.items():
            got = [s[k]["ndcg_at_5"] for k in keys if k in s and "ndcg_at_5" in s[k]]
            if len(got) < 2:
                continue
            n += 1
            diffs.append(max(got) - min(got))
            flips += len(set(got)) > 1
        if n:
            print(f"\n  JUDGE NOISE on arm '{arm}' ({args.repeats}x per user, n={n}):")
            print(f"    users whose score changed between identical calls: {flips}/{n} "
                  f"= {100 * flips / n:.1f}%")
            print(f"    mean spread {sum(diffs) / n:.4f}   max {max(diffs):.4f}")
            print(f"    -> any arm gap below this is judge noise, not a memory effect")

    tok = payload["meta"]["token_stats"]
    pin, pout = PRICES.get(judge, (None, None))
    ti, to = tok.get("total_input_tokens", 0), tok.get("total_output_tokens", 0)
    nreq = max(tok.get("total_requests", 0), 1)
    print(f"\n  tokens: {ti:,} in + {to:,} out over {nreq} calls "
          f"({ti / nreq:.0f} + {to / nreq:.0f} per call)")
    if pin:
        cost = (ti * pin + to * pout) / 1e6
        print(f"  cost:   ${cost:.3f}   -> ${cost / nreq * 1192:.2f} for the full 1192-pair pass")
    print(f"  wall:   {payload['meta']['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
