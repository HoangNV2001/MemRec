"""
M3 Part A: sample teacher `M_collab` from gpt-4o-mini. CPU + API, no GPU.

RL_PLAN.md §M3 warm-starts the policy by rejection sampling: draw `n_samples`
memories per training user from a teacher, score them with the M2 reward, and
SFT on the best one. This script is the sampling half. Scoring stays in Part B
because it needs the GPU ranker, and because separating them means a failed
scoring run does not re-spend the API budget.

Why gpt-4o-mini and not a local 7B (the LEAN change in §2.5.2 ①): 1185 users x 8
samples is ~$7 of API against tens of GPU-hours of local inference.

Three properties this file has to get right, all learned the expensive way:

**Byte-identical prompts.** ``_generate_m_collab`` is imported from
``build_val_reference`` rather than reimplemented. The M2 reference, the val
extension, and this file must all pose the same question to the teacher, or the
samples are not comparable with anything already measured.

**Resumable, checkpointed.** ~70 minutes of paid API calls. A crash at minute 60
that loses everything is a real cost, so completed users are flushed to disk as
they finish and a re-run skips them.

**Temperature 1.0, and that is the point.** The whole method needs *spread*
within a user -- eight identical memories give rejection sampling nothing to
choose between, and give M4 a group with std(r)=0. §6.2 uses the same setting
for rollouts.

    python -m src.rl.build_sft_teacher --config configs/rl/m1_env_books.yaml
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

from src.models import LLMClient                                  # noqa: E402
from src.rl.build_val_reference import _generate_m_collab         # noqa: E402
from src.rl.dataset import load_records                           # noqa: E402
from src.utils import load_config                                 # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="M3-A: teacher M_collab samples")
    p.add_argument("--config", default="configs/rl/m1_env_books.yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--n_samples", type=int, default=8, help="§M3: 8 per user")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None, help="pilot on N users first")
    p.add_argument("--out", default="data/rl/m3_teacher_books.jsonl")
    p.add_argument("--restart", action="store_true",
                   help="ignore existing output instead of resuming")
    return p.parse_args()


def load_done(path: Path) -> Dict[int, Dict]:
    """Users already sampled, so a resumed run does not pay for them twice."""
    done: Dict[int, Dict] = {}
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:      # half-written last line after a hard kill
                continue
            if row.get("samples"):
                done[int(row["user_id"])] = row
    return done


def main():
    args = parse_args()
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    n_facets = config.get("memrec", {}).get("n_facets", 7)
    prefix = config["rl"]["out_prefix"]
    records = load_records(str(PROJECT_ROOT / f"{prefix}_{args.split}.jsonl"))
    if args.limit:
        records = records[: args.limit]

    done = {} if args.restart else load_done(out_path)
    todo = [r for r in records if int(r["user_id"]) not in done]
    print(f"{args.split}: {len(records)} user · {len(done)} da co · {len(todo)} con lai")
    print(f"{args.n_samples} mau/user, temperature {args.temperature}, {args.workers} worker")
    if not todo:
        print("khong con gi de lam")
        return

    provider = config.get("provider", {})
    client = LLMClient(
        api_endpoint=provider.get("endpoint"), api_key=provider.get("api_key"),
        model=provider.get("model", "gpt-4o-mini"), provider_name=provider.get("name", "openai"),
    )

    lock = threading.Lock()
    fh = open(out_path, "a" if not args.restart else "w", encoding="utf-8")
    t0 = time.time()
    n_empty = 0

    def work(record):
        samples: List[List[Dict]] = []
        for _ in range(args.n_samples):
            try:
                samples.append(_generate_m_collab(client, record, n_facets, args.temperature))
            except Exception as exc:      # noqa: BLE001 - one bad draw is not fatal
                print(f"  gen failed u={record['user_id']}: {type(exc).__name__}")
                samples.append([])
        return record, samples

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, r) for r in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sample"):
            record, samples = fut.result()
            n_empty += sum(1 for s in samples if not s)
            with lock:
                # Flushed per user: this is ~70 minutes of paid calls and a crash
                # must not cost more than the user it was in the middle of.
                fh.write(json.dumps({
                    "user_id": int(record["user_id"]),
                    "n_samples": len(samples),
                    "samples": samples,
                }, ensure_ascii=False) + "\n")
                fh.flush()
    fh.close()

    wall = time.time() - t0
    tok = client.get_token_stats()
    ti, to = tok.get("total_input_tokens", 0), tok.get("total_output_tokens", 0)
    total = len(load_done(out_path))
    print(f"\n✓ {total} user -> {out_path}  ({wall/60:.1f} phut)")
    print(f"  token: {ti:,} in + {to:,} out over {tok.get('total_requests', 0)} call")
    print(f"  chi phi gpt-4o-mini: ${(ti*0.15 + to*0.60)/1e6:.2f}")
    print(f"  mau rong (parser tra ve 0 facet): {n_empty}/{len(todo)*args.n_samples}")
    _report_spread(out_path)


def _report_spread(path: Path):
    """
    How much do a user's eight samples actually differ?

    Not cosmetic. Rejection sampling has nothing to select from if the teacher
    returns the same memory eight times, and M4 gets a group with std(r)=0 for
    the same reason (§9.2). Text-level spread is a cheap upper bound on that,
    available before spending any GPU time on scoring.
    """
    rows = load_done(path)
    if not rows:
        return
    ident = flat = 0
    lens = []
    for row in rows.values():
        texts = [json.dumps(s, sort_keys=True) for s in row["samples"] if s]
        if not texts:
            continue
        flat += 1
        lens.append(sum(len(t) for t in texts) / len(texts) / 4.0)
        ident += len(set(texts)) == 1
    if flat:
        lens.sort()
        print(f"  user co ca 8 mau GIONG HET nhau: {ident}/{flat} ({100*ident/flat:.1f}%)")
        print(f"  do dai mau (~token): median {lens[len(lens)//2]:.0f}")


if __name__ == "__main__":
    main()
