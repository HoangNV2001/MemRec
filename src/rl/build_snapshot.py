"""
M1 step 1: warm up memory for the RL user set and freeze it into a graph snapshot.

    python -m src.rl.build_snapshot --config configs/rl/m1_env_books.yaml

This is the only expensive step of M1 (CPU + gpt-4o-mini, no GPU). It is
idempotent at the file level: rerunning overwrites the snapshot, but it *does*
re-spend the API budget, so it takes an explicit --force to clobber an existing
snapshot.
"""
from pathlib import Path

import argparse
import json
import sys
import time

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.data import RecDataset                     # noqa: E402
from src.rl import env as rl_env                    # noqa: E402
from src.rl import splits as rl_splits              # noqa: E402
from src.rl.warmup import run_warmup, summarize     # noqa: E402
from src.train import MemRecTrainer                 # noqa: E402
from src.utils import load_config, set_seed         # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Build the frozen Stage-R graph snapshot (M1)")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default=None, help="overrides config.dataset")
    p.add_argument("--parallel_workers", type=int, default=None)
    p.add_argument("--output_dir", default="results/m1_warmup",
                   help="where warm-up LLM conversations + stats are written")
    p.add_argument("--limit_users", type=int, default=None,
                   help="smoke-test escape hatch: warm up only the first N users of each split")
    p.add_argument("--force", action="store_true", help="overwrite an existing snapshot")
    p.add_argument("--memory_file", default=None,
                   help="re-materialise the snapshot from an already warmed memory dump "
                        "(memory_warmup_only.json) instead of calling the API again. "
                        "Free and offline — use this whenever only the packing or prompt "
                        "changed, so the warm-up budget is spent exactly once (§11.4②).")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_name = args.dataset or config["dataset"]
    rl_cfg = config.get("rl", {})
    seed = config.get("seed", 42)
    n_candidates = config.get("n_eval_candidates", 10)
    n_workers = args.parallel_workers or config.get("parallel_workers", 16)

    snapshot_path = PROJECT_ROOT / rl_cfg["snapshot_file"]
    if snapshot_path.exists() and not args.force:
        sys.exit(
            f"{snapshot_path} already exists. Rebuilding re-spends the API budget; "
            f"pass --force if that is what you want."
        )

    set_seed(seed)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- dataset -----------------------------------------------------------
    data_file = PROJECT_ROOT / "data" / "processed" / dataset_name / f"{dataset_name}.inter"
    if not data_file.exists():
        sys.exit(f"dataset not found: {data_file}")
    print(f"Loading dataset from {data_file}")
    dataset = RecDataset(str(data_file), seed=seed)
    print(dataset)

    # ---- splits ------------------------------------------------------------
    with open(PROJECT_ROOT / rl_cfg["test_user_list"], "r", encoding="utf-8") as f:
        pinned_test = json.load(f)["user_ids"]

    splits = rl_splits.build_splits(
        dataset=dataset,
        test_user_ids=pinned_test,
        n_train=rl_cfg.get("n_train", 1200),
        n_val=rl_cfg.get("n_val", 150),
        seed=seed,
    )
    if args.limit_users:
        splits = {k: v[: args.limit_users] for k, v in splits.items()}
        print(f"⚠ --limit_users={args.limit_users}: smoke mode, snapshot is NOT the real one")

    for name, users in splits.items():
        print(f"  split {name}: {len(users)} users")
    rl_splits.assert_disjoint(splits)

    all_users = sorted(set(splits["train"]) | set(splits["val"]) | set(splits["test"]))
    print(f"  total distinct users to warm up: {len(all_users)}")

    # ---- agent -------------------------------------------------------------
    config = dict(config)
    config["dataset"] = dataset_name
    config["save_llm_conversations"] = True
    config["conversation_file"] = str(output_dir / "llm_conversations.jsonl")

    trainer = MemRecTrainer(None, dataset, config, torch.device("cpu"))
    agent = trainer.agent

    # ---- warm-up (or reload an already-warmed memory) -----------------------
    prior_stats = None
    if args.memory_file:
        print(f"\nReloading warmed memory from {args.memory_file} (no API calls)...")
        agent.storage.load(args.memory_file)
        results, wall = [], 0.0
        stats = {"reloaded_from": args.memory_file}
        # Carry the original run's cost record forward: the snapshot metadata is
        # what the M7b cost table reads, and re-materialising must not erase it.
        prior_path = Path(args.memory_file).parent / "warmup_stats.json"
        if prior_path.exists():
            with open(prior_path, "r", encoding="utf-8") as f:
                prior_stats = json.load(f)
            stats["original_warmup"] = prior_stats.get("stats")
            wall = prior_stats.get("wall_seconds", 0.0)
    else:
        print(f"\nWarming up {len(all_users)} users with {n_workers} workers "
              f"(target = valid item; test item never touched)...")
        t0 = time.time()
        results = run_warmup(
            agent=agent,
            user_ids=all_users,
            n_candidates=n_candidates,
            seed=seed,
            n_workers=n_workers,
        )
        wall = time.time() - t0
        stats = summarize(results)
        print(f"\nWarm-up done in {wall / 60:.1f} min: {stats}")

    main_tokens = trainer.llm_client.get_token_stats()
    rr_tokens = trainer.reranker_llm_client.get_token_stats() if trainer.reranker_llm_client else {}
    print(f"  Stage-R/W tokens : {main_tokens}")
    print(f"  Stage-ReRank     : {rr_tokens}")

    # ---- snapshot ----------------------------------------------------------
    candidates = rl_splits.build_candidates_for_users(
        dataset, all_users, n_candidates=n_candidates, seed=seed
    )
    candidate_items = {i for cands in candidates.values() for i in cands}
    print(f"\nFreezing snapshot ({len(all_users)} users, {len(candidate_items)} candidate items)...")

    snapshot = rl_env.build_snapshot(
        agent=agent,
        user_ids=all_users,
        extra_item_ids=candidate_items,
        meta={
            "dataset": dataset_name,
            "seed": seed,
            "k": config["memrec"]["k"],
            "n_candidates": n_candidates,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_sha": _git_sha(),
            "warmup_stats": stats,
            "warmup_wall_seconds": round(wall, 1),
            "token_stats": (
                prior_stats["token_stats"] if prior_stats
                else {"stage_rw": main_tokens, "stage_rerank": rr_tokens}
            ),
            "candidate_blind": True,
            "test_clean": "warm-up target is valid item (history[-2]); no eval-loop writes",
        },
    )
    snapshot.save(str(snapshot_path))
    size_mb = snapshot_path.stat().st_size / 1e6
    print(f"  snapshot -> {snapshot_path} ({size_mb:.1f} MB, "
          f"{len(snapshot)} users, {len(snapshot.item_memories)} item memories)")

    rl_splits.save_splits(
        splits,
        str(PROJECT_ROOT / rl_cfg["splits_file"]),
        meta={"dataset": dataset_name, "seed": seed, "test_user_list": rl_cfg["test_user_list"]},
    )
    print(f"  splits   -> {PROJECT_ROOT / rl_cfg['splits_file']}")

    if not args.memory_file:
        with open(output_dir / "warmup_stats.json", "w", encoding="utf-8") as f:
            json.dump(
                {"stats": stats, "wall_seconds": wall,
                 "token_stats": {"stage_rw": main_tokens, "stage_rerank": rr_tokens},
                 "failures": [r for r in results if not r.get("ok")]},
                f, indent=2,
            )
        # Durable copy of the only expensive artefact in M1 (§11.4②): the snapshot
        # can be re-materialised from this for free with --memory_file.
        agent.storage.save(str(output_dir / "memory_warmup_only.json"))
    print(f"\n✓ M1 snapshot complete.")


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    main()
