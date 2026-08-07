"""
M2 Validation A/B/C against the cached gpt-4o-mini reference.

Split by tier so the expensive machine does the least possible work
(RL_PLAN.md §2.5.1, §11.6):

* the gpt-4o-mini half is already on disk from ``src.rl.build_val_reference``;
* this script scores the *same* cached ``(user, M_collab)`` pairs with the frozen
  ranker and reports the three DoD numbers.

With ``--ranker_mode stub`` it runs end-to-end on CPU, so the plumbing is proven
before the H100 is booted; only ``--ranker_mode hf`` needs the GPU.

    # CPU, proves the pipeline
    python -m src.rl.validate_reward --ranker_mode stub

    # on the H100, the real thing
    python -m src.rl.validate_reward --ranker_mode hf --device cuda

DoD (§7 M2): Spearman rho >= 0.6 · r(real) > r(other user) > r(lorem) ~ r(empty)
· throughput >= 20 reward/s at batch 64.
"""
from pathlib import Path
from typing import Dict, List, Sequence

import argparse
import itertools
import json
import sys
import time

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.rl.dataset import load_records                          # noqa: E402
from src.rl.reward.metrics import hit_at_k, ndcg_at_k            # noqa: E402
from src.rl.reward.ranker import FrozenRanker                    # noqa: E402
from src.utils import load_config                                # noqa: E402

ARMS = ("sample1", "sample2", "shuffled", "lorem", "empty")


def _sample_arms(reference: Dict) -> List[str]:
    """
    The ``sampleN`` arms actually present, in order.

    M2 Part B ran with two. ``src.rl.extend_val_reference`` adds more, because one
    comparable pair per user is not enough to conclude anything about within-user
    discrimination -- which is the number that decides whether GRPO can learn.
    """
    n = min((len(v) for v in reference["m_collab"].values()), default=0)
    have = {a for a in reference["meta"].get("arms", []) if a.startswith("sample")}
    return [f"sample{i}" for i in range(1, n + 1) if f"sample{i}" in have]


def _effective_arms(reference: Dict) -> List[str]:
    return _sample_arms(reference) + ["shuffled", "lorem", "empty"]


def parse_args():
    p = argparse.ArgumentParser(description="M2 reward validation")
    p.add_argument("--config", default="configs/rl/m1_env_books.yaml")
    p.add_argument("--reference", default="data/rl/m2_val_reference_books.json")
    p.add_argument("--split", default="val")
    p.add_argument("--ranker_mode", choices=["stub", "hf"], default="stub")
    p.add_argument("--ranker_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--no_instruction", action="store_true",
                   help="drop the InstructRec instruction from the ranker prompt")
    p.add_argument("--out", default="data/rl/m2_validation_report.json")
    p.add_argument("--dump_pairs", default=None,
                   help="also write every (user, arm) proxy/reference score pair to "
                        "this path, so a failed run can be diagnosed without paying "
                        "for the forward passes again")
    return p.parse_args()


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, average ranks for ties. No scipy dependency."""
    if len(xs) < 2:
        return float("nan")

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(list(xs)), ranks(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    args = parse_args()
    config = load_config(args.config)
    prefix = config["rl"]["out_prefix"]

    ref_path = PROJECT_ROOT / args.reference
    if not ref_path.exists():
        sys.exit(f"missing reference cache: {ref_path}\n"
                 f"Run: python -m src.rl.build_val_reference --config {args.config}")
    with open(ref_path, "r", encoding="utf-8") as f:
        reference = json.load(f)

    records = {r["user_id"]: r for r in
               load_records(str(PROJECT_ROOT / f"{prefix}_{args.split}.jsonl"))}
    print(f"Reference: {reference['meta']['n_users']} users, arms {reference['meta']['arms']}")

    ranker = FrozenRanker(
        mode=args.ranker_mode,
        model_name=args.ranker_model,
        device=args.device,
        dtype=args.dtype,
        include_instruction=not args.no_instruction,
    )

    # ---- build every (user, arm) request ---------------------------------
    sample_arms = _sample_arms(reference)
    arms_in_play = _effective_arms(reference)
    print(f"Arms in play: {arms_in_play}")
    requests, keys = [], []
    for uid_str, arms in reference["scores"].items():
        uid = int(uid_str)
        record = records.get(uid)
        if record is None:
            continue
        samples = reference["m_collab"].get(uid_str, [[], []])
        shuffled_uid = _shuffled_partner(reference, uid_str)
        arm_facets = {
            "shuffled": reference["m_collab"].get(shuffled_uid, [[]])[0],
            "lorem": _lorem(),
            "empty": [],
        }
        for i, arm in enumerate(sample_arms):
            arm_facets[arm] = samples[i] if len(samples) > i else []
        for arm in arms_in_play:
            if arm not in arms or "ndcg_at_5" not in arms[arm]:
                continue
            requests.append(dict(
                candidates=record["candidates"],
                candidate_titles=record["candidate_titles"],
                candidate_memories=record.get("candidate_memories", {}),
                m_collab=arm_facets[arm] or None,
                instruction=record.get("instruction"),
                user_id=uid,
            ))
            keys.append((uid, arm))

    print(f"Scoring {len(requests)} (user, arm) pairs with the frozen ranker "
          f"[mode={args.ranker_mode}, instruction={'off' if args.no_instruction else 'on'}]...")

    # ---- Validation C: throughput ----------------------------------------
    proxy: Dict = {}
    t0 = time.time()
    for start in range(0, len(requests), args.batch_size):
        chunk = requests[start:start + args.batch_size]
        for (uid, arm), out in zip(keys[start:start + args.batch_size], ranker.score_batch(chunk)):
            gold = int(records[uid]["gold_item_id"])
            proxy[(uid, arm)] = {
                "ndcg_at_5": ndcg_at_k(out.ranking, gold, 5),
                "hit_at_1": hit_at_k(out.ranking, gold, 1),
                "p_gold": float(out.scores.get(gold, 0.0)),
            }
    elapsed = time.time() - t0
    throughput = len(requests) / elapsed if elapsed else float("inf")

    # ---- Validation A: correlation with the real LLM_Rec -----------------
    paired = [
        (proxy[(uid, arm)]["ndcg_at_5"], reference["scores"][str(uid)][arm]["ndcg_at_5"])
        for (uid, arm) in keys if (uid, arm) in proxy
    ]
    rho_all = spearman([p[0] for p in paired], [p[1] for p in paired])

    per_arm_rho = {}
    for arm in arms_in_play:
        pts = [(proxy[(u, a)]["ndcg_at_5"], reference["scores"][str(u)][a]["ndcg_at_5"])
               for (u, a) in keys if a == arm and (u, a) in proxy]
        if len(pts) > 2:
            per_arm_rho[arm] = spearman([p[0] for p in pts], [p[1] for p in pts])

    # ---- Validation B: sensitivity ---------------------------------------
    arm_means = {}
    for arm in arms_in_play:
        vals = [proxy[(u, a)]["ndcg_at_5"] for (u, a) in keys if a == arm and (u, a) in proxy]
        if vals:
            arm_means[arm] = sum(vals) / len(vals)

    # ---- tie rate: decides soft_weight (docs/RESULTS.md, M2 Part A) -------
    tie = _tie_rate_between_samples(proxy, keys, sample_arms)

    # ---- within-user agreement: the number M4 actually depends on --------
    within = _within_user_agreement(proxy, keys, reference, sample_arms)

    report = {
        "ranker_mode": args.ranker_mode,
        "ranker_model": args.ranker_model if args.ranker_mode == "hf" else None,
        "dtype": args.dtype if args.ranker_mode == "hf" else None,
        "include_instruction": not args.no_instruction,
        "n_pairs": len(paired),
        "validation_a": {"spearman_rho": rho_all, "per_arm": per_arm_rho, "threshold": 0.6,
                         "pass": rho_all >= 0.6},
        "validation_b": {"proxy_ndcg_by_arm": arm_means,
                         "reference_ndcg_by_arm": _ref_means(reference),
                         "pass": _sensitivity_ok(arm_means)},
        "validation_c": {"reward_per_second": throughput, "batch_size": args.batch_size,
                         "threshold": 20.0, "pass": throughput >= 20.0},
        "tie_rate": tie,
        "within_user": within,
        "sample_arms": sample_arms,
    }
    _print_report(report)

    if args.dump_pairs:
        dump_path = PROJECT_ROOT / args.dump_pairs
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"user_id": uid, "arm": arm,
             "proxy_ndcg_at_5": proxy[(uid, arm)]["ndcg_at_5"],
             "proxy_hit_at_1": proxy[(uid, arm)]["hit_at_1"],
             "proxy_p_gold": proxy[(uid, arm)]["p_gold"],
             "reference_ndcg_at_5": reference["scores"][str(uid)][arm]["ndcg_at_5"]}
            for (uid, arm) in keys if (uid, arm) in proxy
        ]
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {k: report[k] for k in
                                ("ranker_mode", "ranker_model", "include_instruction")},
                       "pairs": rows}, f, indent=2)
        print(f"pairs  -> {dump_path}")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {out_path}")


def _tie_rate_between_samples(proxy: Dict, keys: Sequence,
                              sample_arms: Sequence[str] = ("sample1", "sample2")) -> Dict:
    """
    How often do two independently sampled M_collab give the SAME reward?

    This is the number that decides ``soft_weight``. A tie inside a GRPO group
    means std(r)=0 -> advantage 0 -> no gradient (§9.2), and dynamic sampling
    (§6.4) would filter past the 60% alarm in M4's kill criteria.

    Measured on gpt-4o-mini at Part A: 80.5% under NDCG@5. Part B measured the
    real ranker (Qwen2.5-3B-Instruct, fp32) at 71.1% under NDCG@5 and 0.0% under
    ``p_gold``, which tripped this rule and turned ``soft_weight`` on at 0.3.
    """
    users = sorted({u for (u, a) in keys if a == sample_arms[0]})
    ndcg_ties = prob_ties = compared = 0
    for u in users:
        for x, y in itertools.combinations(sample_arms, 2):
            a, b = proxy.get((u, x)), proxy.get((u, y))
            if not a or not b:
                continue
            compared += 1
            if abs(a["ndcg_at_5"] - b["ndcg_at_5"]) < 1e-9:
                ndcg_ties += 1
            if abs(a["p_gold"] - b["p_gold"]) < 1e-9:
                prob_ties += 1
    if not compared:
        return {}
    return {
        "n_compared": compared,
        "ndcg_at_5": ndcg_ties / compared,
        "p_gold": prob_ties / compared,
        "recommend_soft_weight": (ndcg_ties / compared) > 0.5,
    }


def _within_user_agreement(proxy: Dict, keys: Sequence, reference: Dict,
                           sample_arms: Sequence[str] = ("sample1", "sample2")) -> Dict:
    """
    Does the proxy prefer the same ``M_collab`` as the real ``LLM_Rec``, *for the
    same user*?

    Validation A's pooled Spearman over all 745 pairs is dominated by user-to-user
    difficulty ("this user is easy for everyone"), which GRPO never sees: every
    rollout in a group belongs to **one** user and differs only in ``M_collab``.
    A proxy can therefore score a healthy pooled rho while being useless as a
    reward -- it would be agreeing about which users are easy, not about which
    memory is better. This measures the latter directly.

    Two views, both reported:

    * ``mean_per_user_rho`` -- Spearman across the arms *within* each user,
      averaged over users where the reference itself distinguishes the arms
      (a user the reference scores flat carries no signal to agree with).
    * ``sample_pairs`` -- the cleanest case: two independent draws of a real
      ``M_collab``. Among the pairs where gpt-4o-mini ranks one above the other,
      how often does the proxy agree? That is exactly a GRPO group of size 2.
      Every pair of sample arms is used, so with five samples each user
      contributes up to C(5,2) = 10 pairs instead of one -- M2 Part B had only 29
      usable pairs in total, whose 95% interval was far too wide to conclude
      anything.
    """
    per_user_rho: List[float] = []
    for uid in sorted({u for (u, _) in keys}):
        px, rx = [], []
        for arm in list(sample_arms) + ["shuffled", "lorem", "empty"]:
            if (uid, arm) not in proxy:
                continue
            ref_arm = reference["scores"].get(str(uid), {}).get(arm, {})
            if "ndcg_at_5" not in ref_arm:
                continue
            px.append(proxy[(uid, arm)]["ndcg_at_5"])
            rx.append(ref_arm["ndcg_at_5"])
        if len(px) < 3 or len(set(rx)) < 2:
            continue                      # reference is flat here: nothing to agree with
        rho = spearman(px, rx)
        if rho == rho:                    # skip NaN (proxy flat across all arms)
            per_user_rho.append(rho)

    concordant = discordant = proxy_tied = 0
    ref_flat = 0
    for uid in sorted({u for (u, a) in keys if a == sample_arms[0]}):
        ref = reference["scores"].get(str(uid), {})
        for x, y in itertools.combinations(sample_arms, 2):
            a, b = proxy.get((uid, x)), proxy.get((uid, y))
            if not a or not b or x not in ref or y not in ref:
                continue
            ref_delta = ref[x]["ndcg_at_5"] - ref[y]["ndcg_at_5"]
            if abs(ref_delta) < 1e-9:
                ref_flat += 1
                continue                  # reference cannot tell them apart
            proxy_delta = a["ndcg_at_5"] - b["ndcg_at_5"]
            if abs(proxy_delta) < 1e-9:
                proxy_tied += 1
            elif (proxy_delta > 0) == (ref_delta > 0):
                concordant += 1
            else:
                discordant += 1

    decided = concordant + discordant
    return {
        "mean_per_user_rho": (sum(per_user_rho) / len(per_user_rho)) if per_user_rho else None,
        "n_users_with_signal": len(per_user_rho),
        "sample_pairs": {
            "arms": list(sample_arms),
            "n_pairs_reference_is_flat": ref_flat,
            "n_reference_distinguishes": decided + proxy_tied,
            "proxy_concordant": concordant,
            "proxy_discordant": discordant,
            "proxy_tied": proxy_tied,
            "accuracy_when_proxy_decides": (concordant / decided) if decided else None,
        },
    }


def _wilson(successes: int, n: int, z: float = 1.96):
    """Wilson score interval -- the small-n honesty check on an agreement rate."""
    if not n:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _shuffled_partner(reference: Dict, uid_str: str) -> str:
    order = list(reference["m_collab"])
    i = order.index(uid_str)
    return order[(i + 1) % len(order)]


def _lorem() -> List[Dict]:
    from src.rl.build_val_reference import LOREM_FACETS
    return LOREM_FACETS


def _ref_means(reference: Dict) -> Dict[str, float]:
    out = {}
    for arm in _effective_arms(reference):
        vals = [s[arm]["ndcg_at_5"] for s in reference["scores"].values()
                if arm in s and "ndcg_at_5" in s[arm]]
        if vals:
            out[arm] = sum(vals) / len(vals)
    return out


def _sensitivity_ok(means: Dict[str, float], margin: float = 0.02) -> bool:
    """
    Validation B: the reward must reward *this user's* memory, not memory in general.

    RL_PLAN.md §7 writes the target ordering as
    ``r(real) > r(other user) > r(lorem) ~ r(empty)``. Measured on the real
    ``LLM_Rec`` (gpt-4o-mini, 149 val users, docs/RESULTS.md M2 Part A), the middle
    inequality does **not** hold and should not be required:

        real      0.7204
        shuffled  0.6090   paired delta vs empty  -0.0002, 95% CI [-0.037, +0.037]
        lorem     0.6079   paired delta vs empty  -0.0013, 95% CI [-0.025, +0.023]
        empty     0.6092

    The real ranker simply *ignores* irrelevant memory rather than being misled by
    it, so another user's memory is worth no more than lorem ipsum. Requiring the
    proxy to reproduce ``shuffled > lorem`` would demand it be more confusable than
    the model it stands in for.

    What actually matters is kept, and tightened with a margin: the real memory
    must beat every corrupted arm. The corrupted arms clustering together is a
    good property, so it is reported but not gated.
    """
    required = {"sample1", "shuffled", "lorem", "empty"}
    if not required <= set(means):
        return False
    corrupted = max(means["shuffled"], means["lorem"], means["empty"])
    return means["sample1"] >= corrupted + margin


def _print_report(report: Dict):
    a, b, c = report["validation_a"], report["validation_b"], report["validation_c"]
    print(f"\n{'=' * 62}\nM2 Reward Validation ({report['ranker_mode']} ranker, "
          f"instruction={'on' if report['include_instruction'] else 'off'})\n{'=' * 62}")
    if report["ranker_mode"] == "stub":
        print("NOTE: the stub ranker scores by hash and has no semantics. Only\n"
              "      Validation C (throughput) and 'the harness runs end to end'\n"
              "      mean anything here. A and B are EXPECTED to fail; they are\n"
              "      decided by the `hf` run on GPU. Do not record these numbers.\n")
    print(f"A  Spearman rho = {a['spearman_rho']:.4f}  (need >= {a['threshold']})  "
          f"{'PASS' if a['pass'] else 'FAIL'}   n={report['n_pairs']}")
    for arm, rho in a["per_arm"].items():
        print(f"     per-arm {arm:<9} {rho:.4f}")
    print(f"\nB  NDCG@5 by arm{'':<8}{'proxy':>10}{'gpt-4o-mini':>14}")
    for arm in report.get("sample_arms", list(ARMS)) + ["shuffled", "lorem", "empty"]:
        p = b["proxy_ndcg_by_arm"].get(arm)
        r = b["reference_ndcg_by_arm"].get(arm)
        if p is not None:
            print(f"     {arm:<20}{p:>10.4f}{(f'{r:.4f}' if r is not None else '-'):>14}")
    print(f"   ordering real > shuffled > lorem ~ empty: {'PASS' if b['pass'] else 'FAIL'}")
    print(f"\nC  throughput = {c['reward_per_second']:.1f} reward/s at batch "
          f"{c['batch_size']}  (need >= {c['threshold']})  {'PASS' if c['pass'] else 'FAIL'}")

    tie = report.get("tie_rate") or {}
    if tie:
        print(f"\nTIE RATE between two sampled M_collab  (n={tie['n_compared']})")
        print(f"     NDCG@5 identical  {100 * tie['ndcg_at_5']:.1f}%"
              f"   (gpt-4o-mini at Part A: 80.5%)")
        print(f"     p_gold identical  {100 * tie['p_gold']:.1f}%")
    w = report.get("within_user") or {}
    if w:
        s = w["sample_pairs"]
        rho = w["mean_per_user_rho"]
        print(f"\nWITHIN-USER agreement  (what a GRPO group actually sees)")
        print(f"     mean per-user rho across arms  "
              f"{'n/a' if rho is None else f'{rho:.4f}'}   "
              f"(n={w['n_users_with_signal']} users with reference signal)")
        acc = s["accuracy_when_proxy_decides"]
        decided = s["proxy_concordant"] + s["proxy_discordant"]
        total = s["n_pairs_reference_is_flat"] + s["n_reference_distinguishes"]
        print(f"     pairs over {len(s['arms'])} sample arms: {total} total; "
              f"gpt-4o-mini flat on {s['n_pairs_reference_is_flat']},")
        print(f"       separates {s['n_reference_distinguishes']}, proxy tied on {s['proxy_tied']},")
        print(f"       of the {decided} the proxy decided, "
              f"{'n/a' if acc is None else f'{100 * acc:.1f}% agree'}  (coin flip = 50%)")
        if acc is not None and decided >= 30:
            lo, hi = _wilson(s["proxy_concordant"], decided)
            print(f"       95% CI [{100 * lo:.1f}%, {100 * hi:.1f}%]"
                  f"{'  -> indistinguishable from chance' if lo <= 0.5 <= hi else ''}")

    if tie:
        if tie["recommend_soft_weight"]:
            print("     -> ties > 50%: turn ON soft_weight (start 0.3). A tie inside a")
            print("        GRPO group is std(r)=0, i.e. no gradient at all (§9.2).")
        else:
            print("     -> ties <= 50%: keep soft_weight = 0.0, the reward as written in §5.")


if __name__ == "__main__":
    main()
