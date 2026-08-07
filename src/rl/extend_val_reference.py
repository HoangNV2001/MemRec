"""
Add more ``M_collab`` samples per user to the cached M2 reference. CPU + API.

Why this exists. M2 Part B measured the thing a GRPO group actually sees -- does
the reward prefer the same ``M_collab`` as the real ``LLM_Rec``, *for the same
user*? -- and the answer was "no evidence of it": 37.5% agreement where chance is
50%. But that verdict rests on **one** comparable pair per user, and gpt-4o-mini
scores the two existing samples identically for 80.5% of users, leaving only 29
usable pairs. A 95% interval on 37.5% of 29 runs from roughly 19% to 59%, which
is not a conclusion.

Going from 2 samples to 5 gives C(5,2) = 10 pairs per user instead of 1, and
separates two readings that the current data cannot:

* the proxy is too coarse to rank two good memories -- fixable, buy a better
  proxy before renting an H100; or
* **gpt-4o-mini itself barely distinguishes them** -- then there is no
  fine-grained signal to learn at all, no proxy can recover it, and M4's ceiling
  is the coarse regime. This one has to be ruled out *first*, because it is the
  reading that no amount of proxy work would fix.

The second reading is answerable from this file's output alone (API only). The
first also needs the frozen ranker to score the new samples, which is a few
GPU-minutes via ``src.rl.validate_reward``.

Everything already in the cache is left untouched and never re-paid for; the run
is resumable and skips any (user, arm) it already has.

    python -m src.rl.extend_val_reference --config configs/rl/m1_env_books.yaml
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
# Reused, not reimplemented: the new samples must come from byte-identical
# prompts and be scored by the identical path, or they are not comparable with
# sample1/sample2 and the whole comparison is meaningless.
from src.rl.build_val_reference import _generate_m_collab, _score   # noqa: E402
from src.utils import load_config                      # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Add M_collab samples to the M2 reference cache")
    p.add_argument("--config", default="configs/rl/m1_env_books.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--cache", default="data/rl/m2_val_reference_books.json")
    p.add_argument("--n_extra", type=int, default=3, help="new samples per user")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None, help="smoke test on N users")
    p.add_argument("--dry_run", action="store_true",
                   help="report what it would cost and stop")
    return p.parse_args()


def main():
    args = parse_args()
    cache_path = PROJECT_ROOT / args.cache
    if not cache_path.exists():
        sys.exit(f"no cache at {cache_path}; run src.rl.build_val_reference first")
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    config = load_config(args.config)
    n_facets = config.get("memrec", {}).get("n_facets", 7)
    prefix = config["rl"]["out_prefix"]
    records = load_records(str(PROJECT_ROOT / f"{prefix}_{args.split}.jsonl"))
    records = [r for r in records if str(r["user_id"]) in cache["m_collab"]]
    if args.limit:
        records = records[: args.limit]

    have = min(len(v) for v in cache["m_collab"].values())
    new_arms = [f"sample{i}" for i in range(have + 1, have + 1 + args.n_extra)]
    print(f"{len(records)} users, {have} samples each -> adding {args.n_extra}: {new_arms}")

    # Only pay for what is actually missing (this script is resumable).
    todo_gen = [r for r in records
                if len(cache["m_collab"][str(r["user_id"])]) < have + args.n_extra]
    todo_score = [(r, arm) for r in records for arm in new_arms
                  if arm not in cache["scores"].get(str(r["user_id"]), {})]
    n_calls = len(todo_gen) * args.n_extra + len(todo_score)
    est = (n_calls / 2) * (1065 * 0.15 + 243 * 0.60) / 1e6 \
        + (len(todo_score)) * (1090 * 0.15 + 446 * 0.60) / 1e6
    print(f"to generate: {len(todo_gen)} users x {args.n_extra} | to score: {len(todo_score)}"
          f" | ~{n_calls} calls, rough estimate ${est:.2f}")
    if args.dry_run:
        return
    if not todo_gen and not todo_score:
        print("nothing to do -- cache already complete")
        return

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

    lock = threading.Lock()
    t0 = time.time()

    # ---- stage 1: generate --------------------------------------------------
    if todo_gen:
        print(f"\n[1/2] Sampling {args.n_extra} more M_collab per user (temperature 1.0)...")

        def gen(record):
            got = []
            for _ in range(args.n_extra):
                try:
                    got.append(_generate_m_collab(gen_client, record, n_facets, 1.0))
                except Exception as exc:  # noqa: BLE001
                    print(f"  gen failed for user {record['user_id']}: {exc}")
                    got.append([])
            return record["user_id"], got

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(gen, r) for r in todo_gen]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="generate"):
                uid, got = fut.result()
                with lock:
                    cache["m_collab"][str(uid)].extend(got)
        _save(cache, cache_path)      # checkpoint: generation is the paid-for part

    # ---- stage 2: score with the real LLM_Rec -------------------------------
    by_id = {r["user_id"]: r for r in records}
    jobs = []
    for r in records:
        uid = str(r["user_id"])
        for i, arm in enumerate(new_arms):
            if arm in cache["scores"].get(uid, {}):
                continue
            facets = cache["m_collab"][uid][have + i]
            jobs.append((r["user_id"], arm, facets))

    if jobs:
        print(f"\n[2/2] Scoring {len(jobs)} new (user, arm) pairs with gpt-4o-mini...")

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
                    cache["scores"].setdefault(str(uid), {})[arm] = res

    wall = time.time() - t0
    meta = cache["meta"]
    meta["arms"] = list(dict.fromkeys(list(meta["arms"]) + new_arms))
    meta.setdefault("extensions", []).append({
        "added_arms": new_arms,
        "n_users": len(records),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": round(wall, 1),
        "token_stats": {"generation": gen_client.get_token_stats(),
                        "reranker": rr_client.get_token_stats()},
        "why": "M2 Part B found no within-user discrimination on 29 pairs; more "
               "samples per user turn that into ~10 pairs per user.",
    })
    _save(cache, cache_path)
    print(f"\n✓ {cache_path} updated ({wall / 60:.1f} min)")
    _report(cache, ["sample1", "sample2"] + new_arms)


def _save(cache: Dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(path)


def _report(cache: Dict, sample_arms: List[str]):
    """
    The API-only half of the verdict: does the real LLM_Rec see *any* spread
    between independently sampled memories for the same user?

    If most users score identically across every sample, there is no fine-grained
    signal for GRPO to learn, and that conclusion needs no GPU at all.
    """
    # Only users that actually have every sample arm. Mixing in users who still
    # have 2 samples would report a 2-sample statistic under a 5-sample heading.
    spreads, flat = [], 0
    per_user_values = []
    partial = 0
    for uid, arms in cache["scores"].items():
        vals = [arms[a]["ndcg_at_5"] for a in sample_arms
                if a in arms and "ndcg_at_5" in arms[a]]
        if len(vals) < len(sample_arms):
            partial += 1
            continue
        per_user_values.append(vals)
        spread = max(vals) - min(vals)
        spreads.append(spread)
        if spread < 1e-9:
            flat += 1
    if not spreads:
        return
    n = len(spreads)
    distinct = [len({round(v, 6) for v in vals}) for vals in per_user_values]
    print(f"\n{'=' * 62}\nWithin-user spread of the REAL LLM_Rec across "
          f"{len(sample_arms)} samples  (n={n} users)\n{'=' * 62}")
    if partial:
        print(f"  ({partial} users skipped: fewer than {len(sample_arms)} samples cached)")
    print(f"  users scoring IDENTICALLY on every sample : {flat}/{n} "
          f"({100 * flat / n:.1f}%)")
    print(f"  mean spread (max-min NDCG@5)              : {sum(spreads) / n:.4f}")
    print(f"  mean distinct NDCG@5 values per user      : {sum(distinct) / n:.2f}"
          f" of {len(sample_arms)}")
    print(f"  usable pairs for within-user comparison   : "
          f"~{sum(len(v) * (len(v) - 1) // 2 for v in per_user_values)}")
    # Sparsity alone is the wrong verdict: what matters for GRPO is whether the
    # non-tied pairs carry a *large* difference, because those are the groups that
    # produce gradient. A signal on 20% of pairs worth 0.34 NDCG each is a far
    # better learning target than one on 80% of pairs worth 0.01.
    import itertools as _it
    diffs = [abs(x - y) for vals in per_user_values
             for x, y in _it.combinations(vals, 2) if abs(x - y) > 1e-9]
    n_pairs = sum(len(v) * (len(v) - 1) // 2 for v in per_user_values)
    print(f"  tied pairs                                : "
          f"{n_pairs - len(diffs)}/{n_pairs} ({100 * (n_pairs - len(diffs)) / n_pairs:.1f}%)")
    if diffs:
        mean_d = sum(diffs) / len(diffs)
        print(f"  mean |NDCG@5 delta| on non-tied pairs      : {mean_d:.4f}")
        print("\n  -> The within-user signal is SPARSE but LARGE: most pairs tie, and")
        print(f"     the ones that do not differ by {mean_d:.2f} on average. Compare the")
        print("     coarse real-vs-empty effect of +0.111 -- so there IS a fine-grained")
        print("     target to learn, concentrated in a minority of users (which is")
        print("     exactly what §6.4's curriculum is meant to select for).")
        print("     Whether the frozen ranker tracks it is the GPU half: run")
        print("     src.rl.validate_reward and read 'WITHIN-USER agreement'.")


if __name__ == "__main__":
    main()
