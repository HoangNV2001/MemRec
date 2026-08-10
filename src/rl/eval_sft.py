"""
M3 DoD: does the SFT checkpoint write valid, useful memory? Base vs SFT on val.

§M3 sets two gates, and neither is "beat the teacher":

    valid JSON on val >= 95%      (base models typically land at 40-70%)
    val NDCG@5(SFT)  >= val NDCG@5(base)

The first is the real point of the warm start. RL on structured JSON from a base
model collapses the format long before it learns anything about memory, so M3
exists to make M4 possible rather than to be good on its own. The second is a
non-regression check: distillation must not trade format for ranking quality.

Everything is generated with thinking disabled (``src.rl.policy.chat_text``) and
scored with the same frozen ranker M2 validated, so these numbers sit on the
same axis as every other measurement in the project. Three extra columns are
reported because they decide what M4 inherits:

``% beating r_null``  the health metric §5.4 keeps r_null for. A policy that
                      cannot beat "no memory at all" has nothing for GRPO to
                      improve on.
``completion tokens`` the cost axis of Figure 4, and §9.3's hacking alarm. The
                      teacher writes ~276; a policy drifting far above that is
                      already losing the argument the thesis makes.
``group spread``      NDCG@5 range across n sampled generations per user. This is
                      std(r) inside a GRPO group (§9.2) measured on the actual
                      policy rather than on teacher samples.

    python -m src.rl.eval_sft --config configs/rl/m3_sft_books.yaml \
        --checkpoint checkpoints/rl/sft_books
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

from src.rl.dataset import load_records                     # noqa: E402
from src.rl.policy import chat_text, parse_facets           # noqa: E402
from src.rl.reward.metrics import hit_at_k, ndcg_at_k       # noqa: E402
from src.rl.reward.ranker import FrozenRanker               # noqa: E402
from src.utils import load_config                           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="M3 DoD: base vs SFT on val")
    p.add_argument("--config", default="configs/rl/m3_sft_books.yaml")
    p.add_argument("--env_config", default="configs/rl/m1_env_books.yaml")
    p.add_argument("--checkpoint", default=None, help="LoRA dir; omit to eval base only")
    p.add_argument("--split", default="val")
    p.add_argument("--n_samples", type=int, default=1,
                   help=">1 also reports the within-user spread a GRPO group sees")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen_batch_size", type=int, default=8)
    p.add_argument("--rank_batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="data/rl/m3_eval_val.json")
    return p.parse_args()


def generate(model, tokenizer, records, cfg, args) -> Dict[int, List[str]]:
    import torch

    texts = [chat_text(tokenizer, r["prompt"]) for r in records]
    out: Dict[int, List[str]] = {int(r["user_id"]): [] for r in records}
    tokenizer.padding_side = "left"      # decoder-only batched generation

    for _ in range(args.n_samples):
        for start in range(0, len(records), args.gen_batch_size):
            chunk = records[start:start + args.gen_batch_size]
            batch = tokenizer(texts[start:start + args.gen_batch_size],
                              return_tensors="pt", padding=True, truncation=True,
                              max_length=cfg["policy"]["max_prompt_tokens"],
                              add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **batch,
                    max_new_tokens=cfg["policy"]["max_completion_tokens"],
                    do_sample=args.temperature > 0,
                    temperature=args.temperature or None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for r, row in zip(chunk, gen):
                completion = tokenizer.decode(row[batch["input_ids"].shape[1]:],
                                              skip_special_tokens=True)
                out[int(r["user_id"])].append(completion)
            print(f"  {min(start + args.gen_batch_size, len(records))}/{len(records)}",
                  end="\r", flush=True)
    return out


def evaluate(tag, generations, records, ranker, args) -> Dict:
    by_id = {int(r["user_id"]): r for r in records}
    valid = total = 0
    lengths, ndcgs, hits, beats, spreads = [], [], [], [], []

    jobs = []
    for uid, comps in generations.items():
        for i, c in enumerate(comps):
            parsed = parse_facets(c, valid_node_ids=list(by_id[uid].get("neighbor_snippets") or {}),
                                  max_facets=7)
            total += 1
            valid += bool(parsed.is_valid)
            lengths.append(len(c) / 4.0)
            jobs.append((uid, parsed.facets or None))

    per_user: Dict[int, List[float]] = {}
    for start in range(0, len(jobs), args.rank_batch_size):
        chunk = jobs[start:start + args.rank_batch_size]
        requests = [dict(candidates=by_id[u]["candidates"],
                         candidate_titles=by_id[u].get("candidate_titles", {}),
                         candidate_memories=by_id[u].get("candidate_memories", {}),
                         m_collab=f, instruction=by_id[u].get("instruction"), user_id=u)
                    for u, f in chunk]
        for (u, _), res in zip(chunk, ranker.score_batch(requests)):
            gold = int(by_id[u]["gold_item_id"])
            v = ndcg_at_k(res.ranking, gold, 5)
            per_user.setdefault(u, []).append(v)
            ndcgs.append(v)
            hits.append(hit_at_k(res.ranking, gold, 1))
            null = by_id[u].get("r_null")
            if null is not None:
                beats.append(1.0 if v > null else 0.0)

    for u, vals in per_user.items():
        if len(vals) > 1:
            spreads.append(max(vals) - min(vals))

    res = {
        "tag": tag,
        "n_users": len(per_user),
        "n_generations": total,
        "format_valid_rate": valid / total if total else 0.0,
        "ndcg_at_5": statistics.fmean(ndcgs) if ndcgs else 0.0,
        "hit_at_1": statistics.fmean(hits) if hits else 0.0,
        "pct_beating_null": statistics.fmean(beats) if beats else None,
        "completion_tokens_median": statistics.median(lengths) if lengths else 0.0,
        "group_flat_rate": (sum(1 for s in spreads if s < 1e-9) / len(spreads)) if spreads else None,
    }
    return res


def _print(rows: List[Dict]):
    keys = [("format_valid_rate", "JSON hop le", "{:.1%}", ">= 95%"),
            ("ndcg_at_5", "NDCG@5 (proxy)", "{:.4f}", "SFT >= base"),
            ("hit_at_1", "H@1 (proxy)", "{:.4f}", ""),
            ("pct_beating_null", "% thang r_null", "{:.1%}", ""),
            ("completion_tokens_median", "token/memory", "{:.0f}", "teacher ~276"),
            ("group_flat_rate", "group phang", "{:.1%}", "canh bao §M4 60%")]
    width = max(len(r["tag"]) for r in rows) + 2
    print(f"\n{'':22} " + "".join(f"{r['tag']:>{width}}" for r in rows) + "   nguong")
    print("-" * (22 + width * len(rows) + 20))
    for key, label, fmt, gate in keys:
        cells = ""
        for r in rows:
            v = r.get(key)
            cells += f"{'--' if v is None else fmt.format(v):>{width}}"
        print(f"{label:22} {cells}   {gate}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    env = load_config(args.env_config)
    records = load_records(str(PROJECT_ROOT / f"{env['rl']['out_prefix']}_{args.split}.jsonl"))
    if args.limit:
        records = records[: args.limit]
    print(f"{args.split}: {len(records)} user x {args.n_samples} mau/user")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["policy"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows, generations_dump = [], {}
    t0 = time.time()
    for tag, ckpt in (("base", None), ("SFT", args.checkpoint)):
        if tag == "SFT" and not ckpt:
            continue
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=getattr(torch, cfg["policy"]["dtype"]), device_map="cuda")
        if ckpt:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ckpt)
        model.eval()
        print(f"\n[{tag}] sinh...")
        gens = generate(model, tokenizer, records, cfg, args)
        generations_dump[tag] = {str(k): v for k, v in gens.items()}
        del model
        torch.cuda.empty_cache()

        # Ranker loaded after the policy is freed: two 4B models will not share
        # a 24 GB card, and the failure would land at the end of a long run.
        ranker = FrozenRanker(mode="hf", device="cuda")
        print(f"[{tag}] cham...")
        rows.append(evaluate(tag, gens, records, ranker, args))
        del ranker
        torch.cuda.empty_cache()

    _print(rows)
    out = PROJECT_ROOT / args.out
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"split": args.split, "n_samples": args.n_samples,
                            "temperature": args.temperature,
                            "checkpoint": args.checkpoint,
                            "wall_seconds": round(time.time() - t0, 1)},
                   "results": rows, "generations": generations_dump}, f, ensure_ascii=False)
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()
