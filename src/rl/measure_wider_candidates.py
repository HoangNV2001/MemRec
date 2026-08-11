"""
Does a wider reward-side candidate list break the within-user tie ceiling?

The ceiling that caps this project's accuracy gain is not the judge. Every scorer
tried ties on roughly the same share of within-user pairs -- gpt-4o-mini 80.1%,
gpt-5.6-luna 79.1%, Qwen3.5-4B 84.1% -- because NDCG@5 over **ten** candidates
takes six values, and two memories that put the gold in the same slot are
identical by construction. That is a property of the evaluation protocol (§2,
N=10 from the paper), not of the model doing the scoring.

But the protocol only has to govern the **reported metric**. The reward is a
training signal, and nothing requires it to read the same candidate list. Widen
the reward-side list and the gold rank has more places to land, so fewer pairs
collapse to the same value -- while the results table still reports N=10 and
stays comparable with the paper.

26 is the ceiling of this design rather than a preference: the scorer reads one
prefill-only forward pass and restricts the next-token logits to the label
tokens, so a label must be a single token. A-Z are verified single-token and
collision-free; two-character labels would break the one-pass trick that §5.1
chose for determinism.

What this measures, on the same 149 val users and the same five cached teacher
memories per user, so it is directly comparable with everything already in
docs/RESULTS.md:

  * tie rate over within-user pairs at N=10 vs N=26, per metric;
  * agreement with the real LLM_Rec, judged on gpt-4o-mini's own N=10 verdicts --
    the 296 pairs it can separate. The ground truth deliberately stays at N=10:
    the question is whether a wider *reward* tracks the deployed protocol better,
    not whether a wider protocol is easier to score.

Extra distractors are drawn deterministically per user from the snapshot's item
memories, excluding that user's existing candidates. With 40080 items and ~14
history entries the chance of drawing a genuine positive is negligible, and it
is the same risk the original ten already carry.

    python -m src.rl.measure_wider_candidates --config configs/rl/m1_env_books.yaml
"""
from pathlib import Path
from typing import Dict, List

import argparse
import itertools
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.dataset import load_records                        # noqa: E402
from src.rl.reward.metrics import ndcg_at_k, rank_of           # noqa: E402
from src.rl.reward.ranker import FrozenRanker                  # noqa: E402
from src.rl.rescore_reference import build_arm_facets          # noqa: E402
from src.rl.splits import user_rng                             # noqa: E402
from src.rl.validate_reward import _wilson                     # noqa: E402
from src.utils import load_config                              # noqa: E402

ARMS = [f"sample{i}" for i in range(1, 6)]
WIDE_CANDIDATE_SALT = 991733          # distinct from the warmup/eval salts


def parse_args():
    p = argparse.ArgumentParser(description="wider reward-side candidate list")
    p.add_argument("--config", default="configs/rl/m1_env_books.yaml")
    p.add_argument("--reference", default="data/rl/m2_val_reference_books.json")
    p.add_argument("--snapshot", default="data/rl/graph_snapshot_books.json")
    p.add_argument("--split", default="val")
    p.add_argument("--n_candidates", type=int, default=26, help="max 26, see module docstring")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="data/rl/m2_wide_candidates.json")
    return p.parse_args()


def title_of(memory: str) -> str:
    """Item memories start with the title; take it up to the first sentence break."""
    head = memory.split(". [")[0].split("\n")[0]
    return head[:120] if head else "unknown"


def widen(records, item_memories: Dict[str, str], n: int) -> Dict[int, Dict]:
    """Extend each user's candidate list to n, deterministically, gold preserved."""
    pool = list(item_memories)
    out: Dict[int, Dict] = {}
    for r in records:
        uid = int(r["user_id"])
        existing = [int(c) for c in r["candidates"]]
        blocked = set(existing)
        rng = user_rng(uid, seed=42, salt=WIDE_CANDIDATE_SALT)
        extra: List[int] = []
        # Oversample then filter: rejection-sampling one at a time makes the
        # result depend on how many draws were rejected, which is not stable
        # across pool changes.
        picks = rng.choice(len(pool), size=min(len(pool), (n - len(existing)) * 4),
                           replace=False)
        for idx in picks:
            cid = int(pool[idx])
            if cid in blocked:
                continue
            extra.append(cid)
            blocked.add(cid)
            if len(existing) + len(extra) >= n:
                break
        candidates = existing + extra
        titles = dict(r.get("candidate_titles", {}))
        memories = dict(r.get("candidate_memories", {}))
        for cid in extra:
            mem = item_memories.get(str(cid), "")
            titles[str(cid)] = title_of(mem)
            memories[str(cid)] = mem
        out[uid] = {"candidates": candidates, "titles": titles, "memories": memories}
    return out


def score(ranker, records, arm_facets, wide, batch_size) -> Dict:
    by_id = {int(r["user_id"]): r for r in records}
    jobs = [(u, a) for u in arm_facets for a in ARMS if a in arm_facets[u]]
    res: Dict[str, Dict[str, Dict]] = {}
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        requests = [dict(candidates=wide[u]["candidates"],
                         candidate_titles=wide[u]["titles"],
                         candidate_memories=wide[u]["memories"],
                         m_collab=arm_facets[u][a] or None,
                         instruction=by_id[u].get("instruction"), user_id=u)
                    for u, a in chunk]
        for (u, a), out in zip(chunk, ranker.score_batch(requests)):
            gold = int(by_id[u]["gold_item_id"])
            rank = rank_of(out.ranking, gold)
            others = [v for i, v in out.logits.items() if i != gold]
            res.setdefault(str(u), {})[a] = {
                "gold_rank": rank,
                "mrr": 1.0 / (rank + 1) if rank is not None else 0.0,
                "ndcg_at_5": ndcg_at_k(out.ranking, gold, 5),
                "ndcg_at_10": ndcg_at_k(out.ranking, gold, 10),
                "margin_logit": float(out.logits[gold] - max(others))
                if (gold in out.logits and others) else 0.0,
            }
        print(f"  {min(start + batch_size, len(jobs))}/{len(jobs)}", end="\r", flush=True)
    return res


def report(wide_scores, narrow_path, reference, n_candidates):
    ref = {u: {a: v.get("ndcg_at_5") for a, v in arms.items()}
           for u, arms in reference["scores"].items()}
    narrow = json.load(open(narrow_path))["scores"] if Path(narrow_path).exists() else {}

    def tie_rate(src, field):
        tot = sep = 0
        for u in src:
            for p, q in itertools.combinations(ARMS, 2):
                a, b = src[u].get(p, {}).get(field), src[u].get(q, {}).get(field)
                if a is None or b is None:
                    continue
                tot += 1
                sep += abs(a - b) > 1e-12
        return (tot - sep) / tot if tot else 0.0, tot

    def agreement(src, field):
        c = n = 0
        for u in src:
            if u not in ref:
                continue
            for p, q in itertools.combinations(ARMS, 2):
                rp, rq = ref[u].get(p), ref[u].get(q)
                if rp is None or rq is None or abs(rp - rq) < 1e-9:
                    continue
                a, b = src[u].get(p, {}).get(field), src[u].get(q, {}).get(field)
                if a is None or b is None or abs(a - b) < 1e-12:
                    continue
                n += 1
                c += (a - b > 0) == (rp - rq > 0)
        return c, n

    print(f"\n=== TI LE HOA trong-user (N={n_candidates} vs N=10) ===")
    print(f"  {'metric':<16} {'hoa N=26':>10} {'hoa N=10':>10}")
    print("  " + "-" * 40)
    for field in ("ndcg_at_5", "ndcg_at_10", "mrr", "margin_logit"):
        w, _ = tie_rate(wide_scores, field)
        base = ""
        if narrow and field in ("ndcg_at_5", "margin_logit"):
            b, _ = tie_rate(narrow, field)
            base = f"{100*b:>9.1f}%"
        print(f"  {field:<16} {100*w:>9.1f}% {base:>10}")

    print(f"\n=== DONG Y voi LLM_Rec that (296 cap gpt-4o-mini phan biet duoc, N=10) ===")
    print(f"  {'metric':<16} {'dung/tong':>12} {'ti le':>8}  {'CI 95%':<16}")
    print("  " + "-" * 58)
    for field in ("ndcg_at_5", "ndcg_at_10", "mrr", "margin_logit"):
        c, n = agreement(wide_scores, field)
        if not n:
            continue
        lo, hi = _wilson(c, n)
        flag = "TREN ngau nhien" if lo > 0.5 else ("DUOI" if hi < 0.5 else "= ngau nhien")
        print(f"  {field:<16} {c:>5}/{n:<6} {100*c/n:>7.1f}%  "
              f"[{100*lo:4.1f},{100*hi:4.1f}]  {flag}")
    print("\n  doi chieu N=10: ndcg_at_5 70.6% tren 102 cap · gop voi margin 62.9% tren 275 cap")


def main():
    args = parse_args()
    if args.n_candidates > 26:
        sys.exit("n_candidates > 26: labels stop being single tokens, see the docstring")
    config = load_config(args.config)
    records = load_records(str(PROJECT_ROOT / f"{config['rl']['out_prefix']}_{args.split}.jsonl"))
    if args.limit:
        records = records[: args.limit]
    with open(PROJECT_ROOT / args.reference, "r", encoding="utf-8") as f:
        reference = json.load(f)
    with open(PROJECT_ROOT / args.snapshot, "r", encoding="utf-8") as f:
        item_memories = json.load(f)["item_memories"]

    arm_facets = build_arm_facets(reference, records)
    arm_facets = {u: v for u, v in arm_facets.items() if int(u) in {int(r["user_id"]) for r in records}}
    wide = widen(records, item_memories, args.n_candidates)
    sizes = {len(v["candidates"]) for v in wide.values()}
    print(f"{len(records)} user · candidate/user = {sorted(sizes)} · {len(ARMS)} arm")

    ranker = FrozenRanker(mode="hf", device=args.device, dtype=args.dtype)
    print(f"ranker {ranker.model_name} {ranker.dtype}")
    t0 = time.time()
    scores = score(ranker, records, arm_facets, wide, args.batch_size)
    wall = time.time() - t0
    print(f"\ncham {sum(len(v) for v in scores.values())} cap trong {wall/60:.1f} phut")

    out = PROJECT_ROOT / args.out
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"n_candidates": args.n_candidates, "ranker": ranker.model_name,
                            "dtype": ranker.dtype, "wall_seconds": round(wall, 1),
                            "salt": WIDE_CANDIDATE_SALT}, "scores": scores}, f)
    report(scores, PROJECT_ROOT / "data/rl/m2_margin_qwen35_4b.json", reference, args.n_candidates)
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()
