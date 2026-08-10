"""
Back-fill the per-user baseline fields left null by ``src.rl.build_dataset``
(RL_PLAN.md M1 checklist, M2-B checklist).

One batched pass of the frozen ranker over every record with **no collaborative
memory** -- the ``r_null`` arm of §5.4 -- writing three fields per user:

``r_null``
    NDCG@5 with ``M_collab = None``. Diagnostic only: it never enters the reward
    (§5.4 -- it is constant per prompt, so GRPO's group mean-centring already
    cancels it). Recorded because "% rollouts beating null" is the best health
    metric M4 has.

``baseline_h1``
    Hit@1 with no memory, exactly as the field is named. **Binary** for a frozen
    deterministic ranker: the gold is either first or it is not.

``baseline_p_gold``
    The ranker's softmax probability on the gold candidate with no memory.
    Continuous difficulty in [0, 1] -- see the note below.

    Why this field exists. §6.4 asks to train only on users whose baseline
    difficulty lies in ``[0.2, 0.8]``, and ``src.rl.dataset.filter_by_difficulty``
    implements that band. A binary ``baseline_h1`` can only ever be 0.0 or 1.0,
    so that band would match **nothing** and silently empty the training set --
    the kind of bug that is only noticed after a rented session has already been
    paid for. ``p_gold`` expresses the same intent ("drop users the frozen ranker
    already nails, and users it never gets") on a continuous scale, so the band
    means what §6.4 wanted it to mean. Both fields are written; the choice of
    which one drives the curriculum stays in ``filter_by_difficulty``.

Run this only **after** the M2 ranker has passed validation: every number here is
a property of that specific ranker, so switching rankers invalidates all of them.

    python -m src.rl.backfill_baselines --ranker_model Qwen/Qwen2.5-3B-Instruct \
        --device cuda
"""
from pathlib import Path
from typing import Dict

import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.dataset import backfill, load_records          # noqa: E402
from src.rl.reward.metrics import hit_at_k, ndcg_at_k      # noqa: E402
from src.rl.reward.ranker import FrozenRanker              # noqa: E402
from src.utils import load_config                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="back-fill r_null / baseline_h1 / baseline_p_gold")
    p.add_argument("--config", default="configs/rl/m1_env_books.yaml")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--ranker_mode", choices=["stub", "hf"], default="hf")
    # No default here on purpose: it must be whatever FrozenRanker validated at M2,
    # and a second copy of the model name is a second thing to forget to update.
    # An earlier version of this file duplicated the 1.5B default and silently
    # back-filled every split with the *unvalidated* ranker.
    p.add_argument("--ranker_model", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scoring", choices=["listwise", "pointwise"], default="listwise",
                   help="pointwise = one yes/no forward pass per candidate (§M2 fallback)")
    # Same reason as --ranker_model above: no default here. A duplicated dtype
    # default silently overrode the validated one and OOM'd a 4B model into fp32
    # on a 24 GB card -- the second time this file grew a second source of truth.
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no_instruction", action="store_true")
    p.add_argument("--ndcg_k", type=int, default=5)
    p.add_argument("--dry_run", action="store_true",
                   help="score and report, but do not write the jsonl files")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    prefix = config["rl"]["out_prefix"]

    ranker_kwargs = {} if args.ranker_model is None else {"model_name": args.ranker_model}
    ranker = FrozenRanker(
        mode=args.ranker_mode,
        **ranker_kwargs,
        device=args.device,
        **({} if args.dtype is None else {"dtype": args.dtype}),
        scoring=args.scoring,
        include_instruction=not args.no_instruction,
    )
    # Printed because every number written below is a property of *this* ranker:
    # re-running with a different one silently invalidates all three fields.
    print(f"ranker: {ranker.model_name}  mode={ranker.mode}  dtype={ranker.dtype}  "
          f"device={args.device}  instruction={'on' if ranker.include_instruction else 'off'}")

    for split in args.splits:
        path = PROJECT_ROOT / f"{prefix}_{split}.jsonl"
        if not path.exists():
            print(f"skip {split}: {path} not found")
            continue

        records = load_records(str(path))
        print(f"\n{split}: {len(records)} records -> {path.name}")

        values: Dict[int, Dict] = {}
        t0 = time.time()
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            requests = [dict(
                candidates=r["candidates"],
                candidate_titles=r.get("candidate_titles", {}),
                candidate_memories=r.get("candidate_memories", {}),
                m_collab=None,                      # <- the null arm
                instruction=r.get("instruction"),
                user_id=int(r["user_id"]),
            ) for r in chunk]

            for r, out in zip(chunk, ranker.score_batch(requests)):
                gold = int(r["gold_item_id"])
                values[int(r["user_id"])] = {
                    "r_null": ndcg_at_k(out.ranking, gold, args.ndcg_k),
                    "baseline_h1": hit_at_k(out.ranking, gold, 1),
                    "baseline_p_gold": float(out.scores.get(gold, 0.0)),
                }
            done = min(start + args.batch_size, len(records))
            print(f"  {done}/{len(records)}", end="\r", flush=True)

        elapsed = time.time() - t0
        _summarise(values, elapsed)

        if args.dry_run:
            print("  dry run: nothing written")
            continue
        n = backfill(str(path), values)
        _record_provenance(split, path, ranker, n)
        print(f"  wrote {n} records")


PROVENANCE_PATH = "data/rl/baselines_provenance.json"


def _record_provenance(split: str, path: Path, ranker: FrozenRanker, n: int):
    """
    Record which ranker produced the baselines in each split.

    These three fields are properties of one specific ranker, but they live inside
    the same jsonl as everything else, so a partial or wrong-model run leaves
    values that look perfectly valid. That happened twice while building this:
    once from a stale duplicated default (1.5B instead of the validated 3B), and
    once from an OOM that killed one split and left the other two updated. Neither
    was visible from the data. A per-split stamp makes both trivially checkable.
    """
    out = PROJECT_ROOT / PROVENANCE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        payload = {}
    payload[split] = {
        "file": path.name,
        "n_records": n,
        "ranker_model": ranker.model_name,
        "dtype": ranker.dtype,
        "scoring": ranker.scoring,
        "include_instruction": ranker.include_instruction,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _summarise(values: Dict[int, Dict], elapsed: float):
    if not values:
        return
    n = len(values)
    r_null = [v["r_null"] for v in values.values()]
    h1 = [v["baseline_h1"] for v in values.values()]
    pg = sorted(v["baseline_p_gold"] for v in values.values())
    in_band = sum(1 for p in pg if 0.2 <= p <= 0.8)
    print(f"  {n} scored in {elapsed:.1f}s ({n / elapsed:.1f}/s)")
    print(f"  mean r_null (NDCG@5, no memory) = {sum(r_null) / n:.4f}")
    print(f"  baseline_h1 rate               = {sum(h1) / n:.4f}  "
          f"({int(sum(h1))}/{n} users, binary per user)")
    print(f"  baseline_p_gold  median={pg[n // 2]:.4f}  "
          f"in curriculum band [0.2, 0.8]: {in_band}/{n} ({100 * in_band / n:.1f}%)")


if __name__ == "__main__":
    main()
