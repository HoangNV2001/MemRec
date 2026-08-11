"""
M3 Part B step 1: score the teacher samples and keep the best per user.

§M3 warm-starts the policy on the best of ``n`` teacher memories, accepted only
when it beats the no-memory baseline: ``keep top-1 if r > r_null``. This is that
selection, using the full M2 reward -- NDCG@5 plus the margin tie-breaker,
grounding, and the length and format penalties -- rather than NDCG alone, so the
SFT target is the memory the *reward* prefers and not merely the one that ranks
well. Training the policy toward a different objective than M4 will optimise is
how a warm start turns into a fight with the RL stage.

``r_null`` must come from the same ranker. It does not travel: it was written
into the jsonl by Qwen2.5-3B and is meaningless for Qwen3.5-4B, which is what
``data/rl/baselines_provenance.json`` is for. Re-run ``backfill_baselines``
first; this script refuses to start if the provenance disagrees with the ranker
it was given.

The by-product is worth as much as the dataset. Scoring eight samples per user
gives the reward spread inside a group of exactly the shape M4 will sample, on
the *training* split rather than the 149 val users -- the first direct
measurement of how many GRPO groups will be degenerate (§9.2) before renting
anything.

    python -m src.rl.select_sft_data --config configs/rl/m1_env_books.yaml
"""
from pathlib import Path
from typing import Dict, List

import argparse
import json
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.dataset import load_records                       # noqa: E402
from src.rl.reward.composite import RewardConfig, StageRReward  # noqa: E402
from src.rl.reward.grounding import GroundingScorer           # noqa: E402
from src.rl.reward.ranker import FrozenRanker                 # noqa: E402
from src.utils import load_config                             # noqa: E402

PROVENANCE = "data/rl/baselines_provenance.json"


def parse_args():
    p = argparse.ArgumentParser(description="M3-B: score teacher samples, keep best-of-n")
    p.add_argument("--config", default="configs/rl/m1_env_books.yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--teacher", default="data/rl/m3_teacher_books.jsonl")
    p.add_argument("--out", default="data/rl/m3_sft_books.jsonl")
    p.add_argument("--stats_out", default="data/rl/m3_group_spread.json")
    p.add_argument("--ranker_model", default=None, help="default: FrozenRanker's")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow_stale_baselines", action="store_true",
                   help="skip the provenance check; only for a deliberate test run")
    return p.parse_args()


def check_provenance(split: str, ranker: FrozenRanker, allow_stale: bool):
    """
    Refuse to run against `r_null` produced by a different ranker.

    The acceptance rule is `r > r_null`, so a mismatch quietly compares two
    models and changes which users survive. This exact failure already happened
    once at M2 (a duplicated default silently back-filled all three splits with
    an unvalidated ranker) and was only caught because two independent code paths
    disagreed about one number.
    """
    path = PROJECT_ROOT / PROVENANCE
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f).get(split, {})
    except (OSError, ValueError):
        rec = {}
    written_by = rec.get("ranker_model")
    if written_by == ranker.model_name:
        return
    msg = (f"baselines for split {split!r} were written by {written_by!r}, but the "
           f"ranker here is {ranker.model_name!r}. `r_null` is a property of one "
           f"specific ranker and the acceptance rule compares against it.\n"
           f"    python -m src.rl.backfill_baselines --config <cfg> --device cuda")
    if not allow_stale:
        sys.exit("REFUSING TO RUN: " + msg)
    print("WARNING (--allow_stale_baselines): " + msg)


def load_teacher(path: Path, limit=None) -> Dict[int, List[List[Dict]]]:
    out: Dict[int, List[List[Dict]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("samples"):
                out[int(row["user_id"])] = row["samples"]
            if limit and len(out) >= limit:
                break
    return out


def as_completion(facets: List[Dict]) -> str:
    """
    Serialise facets the way the policy is expected to emit them.

    Canonical rather than verbatim: the teacher's raw text varies in whitespace
    and key order, and an SFT target with a single stable shape is both easier to
    learn and easier to parse, which feeds `format_valid_rate` (§9.4).
    """
    return json.dumps({"facets": facets}, ensure_ascii=False)


def main():
    args = parse_args()
    config = load_config(args.config)
    prefix = config["rl"]["out_prefix"]
    records = load_records(str(PROJECT_ROOT / f"{prefix}_{args.split}.jsonl"))
    by_id = {int(r["user_id"]): r for r in records}

    teacher = load_teacher(PROJECT_ROOT / args.teacher, args.limit)
    print(f"teacher: {len(teacher)} user x "
          f"{len(next(iter(teacher.values()))) if teacher else 0} mau")

    kwargs = {} if args.ranker_model is None else {"model_name": args.ranker_model}
    ranker = FrozenRanker(mode="hf", **kwargs, device=args.device, dtype=args.dtype)
    print(f"ranker: {ranker.model_name} dtype={ranker.dtype}")
    check_provenance(args.split, ranker, args.allow_stale_baselines)

    cfg = RewardConfig()
    print(f"reward: ndcg@{cfg.ndcg_k} + {cfg.margin_weight}*margin_logit "
          f"+ {cfg.lambda_ground}*ground - len - fmt")
    reward = StageRReward(ranker=ranker,
                          grounding=GroundingScorer(n_facets=cfg.n_facets),
                          config=cfg)

    jobs = [(uid, i) for uid in teacher for i in range(len(teacher[uid])) if uid in by_id]
    per_user: Dict[int, List] = {uid: [] for uid in teacher if uid in by_id}
    t0 = time.time()
    # Batched through StageRReward.score_many: scoring one prompt at a time costs
    # ~1/s against ~2.8/s at batch 32 on this ranker, i.e. hours rather than
    # minutes for 9480 samples.
    chunk_size = args.batch_size * 8
    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start:start + chunk_size]
        outs = reward.score_many([as_completion(teacher[uid][i]) for uid, i in chunk],
                                 [by_id[uid] for uid, _ in chunk],
                                 batch_size=args.batch_size)
        for (uid, i), b in zip(chunk, outs):
            per_user[uid].append((b.total, i, b))
        print(f"  {min(start + chunk_size, len(jobs))}/{len(jobs)}", end="\r", flush=True)
    wall = time.time() - t0

    kept, dropped, spreads, stats = [], 0, [], {}
    for uid, scored in per_user.items():
        if not scored:
            continue
        totals = [s[0] for s in scored]
        ndcgs = [s[2].r_ndcg for s in scored]
        r_null = by_id[uid].get("r_null")
        spread = max(totals) - min(totals)
        spreads.append(spread)
        stats[str(uid)] = {
            "rewards": totals, "r_ndcg": ndcgs, "r_null": r_null,
            "std": statistics.pstdev(totals), "spread": spread,
            # Recorded separately because "std(r) > 0" and "the group contains a
            # real ranking difference" are different claims. NDCG@5 over ten
            # candidates takes six values, so a group whose NDCG is flat gets its
            # entire gradient direction from the continuous terms -- and
            # margin_logit only agrees with the real LLM_Rec 58.4% of the time on
            # exactly those cases. Pure noise also produces a non-zero spread;
            # that is how gpt-5.6-luna + margin looked like a win.
            "ndcg_spread": max(ndcgs) - min(ndcgs),
        }
        # Filter on the acceptance rule FIRST, then take the best of what passed.
        # Picking argmax(total) and only then testing its r_ndcg discards the whole
        # user whenever the highest-reward sample is not the highest-NDCG one --
        # a sample that would have qualified is never considered. The two
        # quantities are also on different scales: r_null is a bare NDCG@5, while
        # total carries grounding, the margin term and the penalties.
        eligible = [s for s in scored if r_null is None or s[2].r_ndcg > r_null]
        if not eligible:
            dropped += 1
            continue
        best_total, best_i, best_b = max(eligible, key=lambda s: s[0])
        kept.append({
            "user_id": uid,
            "prompt": by_id[uid]["prompt"],
            "completion": as_completion(teacher[uid][best_i]),
            "reward": best_total,
            "r_ndcg": best_b.r_ndcg,
            "r_null": r_null,
            "sample_index": best_i,
            "n_candidates_scored": len(scored),
        })

    out = PROJECT_ROOT / args.out
    with open(out, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(PROJECT_ROOT / args.stats_out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"ranker": ranker.model_name, "dtype": ranker.dtype,
                            "margin_weight": cfg.margin_weight,
                            "n_users": len(per_user), "wall_seconds": round(wall, 1)},
                   "per_user": stats}, f)

    _report(kept, dropped, per_user, spreads, wall, out)


def _report(kept, dropped, per_user, spreads, wall, out):
    n = len(per_user)
    print(f"\nchấm {sum(len(v) for v in per_user.values())} mẫu trong {wall/60:.1f} phút "
          f"({sum(len(v) for v in per_user.values())/wall:.2f}/s)")
    print(f"giữ {len(kept)}/{n} user ({100*len(kept)/n:.1f}%) · loại {dropped} vì không thắng r_null")

    # The number M4 actually depends on. Anything with zero spread is a group
    # with std(r)=0: no advantage, no gradient (§9.2), and dynamic sampling
    # (§6.4) drops it. The M4 kill criteria call >60% an alarm.
    flat = sum(1 for s in spreads if s < 1e-9)
    spreads_sorted = sorted(spreads)
    print(f"\nDỰ BÁO GROUP SUY BIẾN (8 mẫu/user, đúng hình dạng M4 sẽ lấy):")
    print(f"  group phẳng hoàn toàn: {flat}/{n} = {100*flat/n:.1f}%  (ngưỡng báo động §M4: 60%)")
    print(f"  spread reward: median {spreads_sorted[len(spreads_sorted)//2]:.4f}  "
          f"p90 {spreads_sorted[int(0.9*len(spreads_sorted))]:.4f}")
    # "std(r) > 0" is not the same claim as "this group contains a real ranking
    # difference". Reported apart so the headline cannot be read as more than it is.
    nd_flat = sum(1 for u in per_user
                  if max(s[2].r_ndcg for s in per_user[u]) - min(s[2].r_ndcg for s in per_user[u]) < 1e-9)
    print(f"  trong đó group KHÔNG có khác biệt thứ hạng nào (NDCG@5 phẳng): "
          f"{nd_flat}/{n} = {100*nd_flat/n:.1f}%")
    print(f"    -> ở những group này toàn bộ hướng gradient do margin/grounding quyết định,")
    print(f"       và margin chỉ đồng ý với LLM_Rec thật 58.4% [50.9, 65.5] trên đúng loại cặp đó")

    if kept:
        rw = sorted(k["reward"] for k in kept)
        print(f"\nreward của mẫu được chọn: median {rw[len(rw)//2]:.4f}  "
              f"min {rw[0]:.4f}  max {rw[-1]:.4f}")
        idx = [k["sample_index"] for k in kept]
        print(f"  chỉ số mẫu thắng phân bố đều? mean {sum(idx)/len(idx):.2f} "
              f"(đều = 3.5 với 8 mẫu)")
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()
