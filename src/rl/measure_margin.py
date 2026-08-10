"""
Does `gold_margin` survive on the local ranker? CPU-free, one GPU pass.

Qwen3.5-4B cleared Validation A (rho = 0.7726) and B, but it still ties 84.1% of
within-user pairs, and a tie inside a GRPO group is std(r)=0 and no gradient
(§9.2). A passing correlation with a mostly-flat reward still trains nothing.

`gold_margin` fixed exactly that on gpt-4o-mini -- degenerate groups fell from
67.8% to 42.6% while agreeing with NDCG@5 88.5% of the time (76.2% replicating
across independent runs), which is what separates it from soft_weight's p_gold,
a continuous term that was anti-informative at 40.1%. This asks whether the same
holds for the model that will actually compute the reward.

Two margins are recorded because the choice is not obvious:

``margin_prob``   p(gold) - max p(other), from the softmax over letters A-J.
                  Bounded, directly comparable to gpt-4o-mini's 0-1 scores, but
                  this ranker is extremely peaked (letter mass 0.9997, ~99% of
                  it on one letter), so it saturates near +-1 and can re-tie at
                  the extremes.
``margin_logit``  the same difference on raw logits. Unbounded and numerically
                  stable. The pointwise experiment already lost a run to a
                  softmax underflowing to exactly 0.0 and re-creating the tie
                  problem it was meant to solve, so the raw form is kept too.

Every separation rate is reported against its own noise floor, measured by
scoring the same memories twice under different batch composition. That is not
optional here: the fast-path kernels are not batch-invariant (3/48 users move),
so some apparent discrimination is the kernel, not the memory.

    python -m src.rl.measure_margin --config configs/rl/m1_env_books.yaml
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
from src.rl.validate_reward import _wilson, spearman           # noqa: E402
from src.utils import load_config                              # noqa: E402

ARMS = [f"sample{i}" for i in range(1, 6)]
BAD = ("shuffled", "lorem", "empty")


def parse_args():
    p = argparse.ArgumentParser(description="gold_margin on the local frozen ranker")
    p.add_argument("--config", default="configs/rl/m1_env_books.yaml")
    p.add_argument("--reference", default="data/rl/m2_val_reference_books.json")
    p.add_argument("--split", default="val")
    p.add_argument("--ranker_model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--noise_batch_size", type=int, default=8,
                   help="second pass over the sample arms at a different batch "
                        "composition; the gap is the kernel's own noise floor")
    p.add_argument("--out", default="data/rl/m2_margin_qwen35_4b.json")
    return p.parse_args()


def score_all(ranker, records, arm_facets, arms, batch_size) -> Dict:
    by_id = {int(r["user_id"]): r for r in records}
    jobs = [(u, a) for u in arm_facets for a in arms if a in arm_facets[u]]
    out: Dict[str, Dict[str, Dict]] = {}

    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        requests = []
        for uid, arm in chunk:
            r = by_id[uid]
            facets = arm_facets[uid][arm]
            requests.append(dict(
                candidates=r["candidates"],
                candidate_titles=r.get("candidate_titles", {}),
                candidate_memories=r.get("candidate_memories", {}),
                m_collab=facets or None,
                instruction=r.get("instruction"),
                user_id=uid,
            ))
        for (uid, arm), res in zip(chunk, ranker.score_batch(requests)):
            gold = int(by_id[uid]["gold_item_id"])
            probs, logits = res.scores, res.logits
            others_p = [v for i, v in probs.items() if i != gold]
            others_l = [v for i, v in logits.items() if i != gold]
            out.setdefault(str(uid), {})[arm] = {
                "ndcg_at_5": ndcg_at_k(res.ranking, gold, 5),
                "gold_rank": rank_of(res.ranking, gold),
                "p_gold": float(probs.get(gold, 0.0)),
                "margin_prob": float(probs.get(gold, 0.0) - max(others_p)) if others_p else None,
                "margin_logit": float(logits.get(gold, 0.0) - max(others_l)) if others_l else None,
            }
        print(f"  {min(start + batch_size, len(jobs))}/{len(jobs)}", end="\r", flush=True)
    return out


def _pairs(S, field, arms=ARMS):
    for u in S:
        for p, q in itertools.combinations(arms, 2):
            a, b = S[u].get(p, {}).get(field), S[u].get(q, {}).get(field)
            if a is not None and b is not None:
                yield u, p, q, a, b


def report(S, noise, reference):
    ref = {u: {a: v.get("ndcg_at_5") for a, v in arms.items()}
           for u, arms in reference["scores"].items()}

    print("\n=== 1. Tach duoc bao nhieu, sau khi tru nen nhieu ===")
    print(f"  {'dai luong':<16} {'tach':>8} {'nen nhieu':>11} {'THAT':>10} {'group suy bien':>16}")
    print("  " + "-" * 65)
    for field in ("ndcg_at_5", "margin_prob", "margin_logit"):
        tot = sep = 0
        for _, _, _, a, b in _pairs(S, field):
            tot += 1
            sep += abs(a - b) > 1e-12
        n = d = 0
        for u in noise:
            for a in ARMS:
                x = S.get(u, {}).get(a, {}).get(field)
                y = noise.get(u, {}).get(a, {}).get(field)
                if x is None or y is None:
                    continue
                n += 1
                d += abs(x - y) > 1e-12
        flat = t = 0
        for u in S:
            v = [S[u][a][field] for a in ARMS if a in S[u] and S[u][a][field] is not None]
            if len(v) < 5:
                continue
            t += 1
            flat += len(set(round(x, 12) for x in v)) == 1
        sr, nf = sep / tot, (d / n if n else 0.0)
        print(f"  {field:<16} {100*sr:>7.1f}% {100*nf:>10.1f}% {100*(sr-nf):>8.1f} d "
              f"{100*flat/t:>15.1f}%")
    print("  doi chieu gpt-4o-mini: ndcg 19.3%/8.6%/67.8%  ·  margin 34.1%/13.4%/42.6%")

    print("\n=== 2. Margin co dong huong voi NDCG@5, hay la nhieu lien tuc? ===")
    for field in ("margin_prob", "margin_logit"):
        c = n = 0
        for u in S:
            for p, q in itertools.combinations(ARMS, 2):
                dn = S[u].get(p, {}).get("ndcg_at_5"), S[u].get(q, {}).get("ndcg_at_5")
                dm = S[u].get(p, {}).get(field), S[u].get(q, {}).get(field)
                if None in dn or None in dm:
                    continue
                x, y = dn[0] - dn[1], dm[0] - dm[1]
                if abs(x) < 1e-12 or abs(y) < 1e-12:
                    continue
                n += 1
                c += (x > 0) == (y > 0)
        if n:
            lo, hi = _wilson(c, n)
            print(f"  {field:<14} {c:>4}/{n:<4} = {100*c/n:5.1f}%  CI [{100*lo:4.1f},{100*hi:4.1f}]")
    print("  doi chieu: gpt-4o-mini margin 88.5% · soft_weight p_gold 40.1% (DUOI ngau nhien)")

    print("\n=== 3. Within-user vs gpt-4o-mini that (296 cap no phan biet duoc) ===")
    for field in ("ndcg_at_5", "margin_prob", "margin_logit"):
        c = n = 0
        for u in S:
            for p, q in itertools.combinations(ARMS, 2):
                if u not in ref or p not in ref[u] or q not in ref[u]:
                    continue
                rd = ref[u][p] - ref[u][q]
                if abs(rd) < 1e-9:
                    continue
                a, b = S[u].get(p, {}).get(field), S[u].get(q, {}).get(field)
                if a is None or b is None or abs(a - b) < 1e-12:
                    continue
                n += 1
                c += (a - b > 0) == (rd > 0)
        if n:
            lo, hi = _wilson(c, n)
            v = "TREN ngau nhien" if lo > 0.5 else ("DUOI" if hi < 0.5 else "= ngau nhien")
            print(f"  {field:<14} {c:>4}/{n:<4} = {100*c/n:5.1f}%  "
                  f"CI [{100*lo:4.1f},{100*hi:4.1f}]  {v}")

    print("\n=== 4. Headroom: margin co mang tin hieu chat luong memory khong? ===")
    for field in ("ndcg_at_5", "margin_prob", "margin_logit"):
        m = {}
        for a in ARMS + list(BAD):
            v = [S[u][a][field] for u in S if a in S[u] and S[u][a][field] is not None]
            if v:
                m[a] = sum(v) / len(v)
        real = sum(m[a] for a in ARMS if a in m) / len([a for a in ARMS if a in m])
        worst = max(m[a] for a in BAD if a in m)
        print(f"  {field:<14} real {real:+.4f}  arm hong te nhat {worst:+.4f}  "
              f"headroom {real-worst:+.4f}")

    xs = [v["margin_logit"] for u in S for a, v in S[u].items() if v.get("margin_logit") is not None]
    ys = [S[u][a]["ndcg_at_5"] for u in S for a in S[u] if S[u][a].get("margin_logit") is not None]
    print(f"\n  rho(margin_logit, ndcg_at_5) tren {len(xs)} cap = {spearman(xs, ys):.4f}")


def main():
    args = parse_args()
    config = load_config(args.config)
    prefix = config["rl"]["out_prefix"]
    records = load_records(str(PROJECT_ROOT / f"{prefix}_{args.split}.jsonl"))
    with open(PROJECT_ROOT / args.reference, "r", encoding="utf-8") as f:
        reference = json.load(f)

    arm_facets = build_arm_facets(reference, records)
    arms = reference["meta"]["arms"]
    print(f"ranker {args.ranker_model} {args.dtype} | {len(arm_facets)} user x {len(arms)} arm")

    ranker = FrozenRanker(mode="hf", model_name=args.ranker_model,
                          device=args.device, dtype=args.dtype)
    t0 = time.time()
    print(f"\npass 1: moi arm, batch {args.batch_size}")
    scores = score_all(ranker, records, arm_facets, arms, args.batch_size)
    print(f"\npass 2 (nen nhieu): 5 arm mau, batch {args.noise_batch_size}")
    noise = score_all(ranker, records, arm_facets, ARMS, args.noise_batch_size)
    wall = time.time() - t0

    out = PROJECT_ROOT / args.out
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"ranker": args.ranker_model, "dtype": args.dtype,
                            "batch_size": args.batch_size,
                            "noise_batch_size": args.noise_batch_size,
                            "wall_seconds": round(wall, 1),
                            "built_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                   "scores": scores, "noise": noise}, f)
    report(scores, noise, reference)
    print(f"\n✓ {out}  ({wall/60:.1f} phut)")


if __name__ == "__main__":
    main()
