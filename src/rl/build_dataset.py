"""
M1 step 2: turn the frozen snapshot into train/val/test jsonl for TRL.

    python -m src.rl.build_dataset --config configs/rl/m1_env_books.yaml

Pure CPU, no API, no GPU -- cheap to rerun whenever the prompt template changes.

Each line (RL_PLAN.md §7 M1):

    {"user_id", "prompt", "candidates", "gold_item_id", "M_u", "neighbors",
     "r_null", "baseline_h1", ...}

``r_null`` and ``baseline_h1`` are written as null and back-filled by a single
batch job after M2, exactly as the plan specifies.

Fields beyond the plan's list are all *outside* ``prompt`` and exist for the
reward function and the M5 ablations:
  ``instruction``      - InstructRec text; goes to the frozen ranker only, never
                         to the policy (it paraphrases the target book).
  ``candidate_titles`` - so the reward ranker does not need the 600MB .meta file.
  ``candidate_memories``
  ``neighbor_ids``     - whitelist for the grounding reward (§5.2).
  ``n_train_items``    - |H_u|, for the sparsity-quartile breakdown (§M5).
"""
from pathlib import Path

import argparse
import json
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.data import RecDataset                 # noqa: E402
from src.rl import policy as rl_policy          # noqa: E402
from src.rl import splits as rl_splits          # noqa: E402
from src.rl.env import GraphSnapshot, parse_neighbor_snippets  # noqa: E402
from src.rl.leakage import gold_leak_reason     # noqa: E402
from src.utils import load_config               # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Build Stage-R RL jsonl datasets (M1)")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--out_prefix", default=None, help="overrides config.rl.out_prefix")
    return p.parse_args()


def build_record(
    user_id: int,
    snapshot: GraphSnapshot,
    dataset: RecDataset,
    candidates,
    n_facets: int,
) -> dict:
    state = snapshot.state(user_id)
    gold = int(dataset.test_data[user_id])

    instruction = ""
    if dataset.instructions and user_id in dataset.instructions:
        instruction = dataset.instructions[user_id].get("instruction", "") or ""

    titles = {}
    memories = {}
    for cid in candidates:
        meta = (dataset.item_metadata or {}).get(cid, {})
        titles[str(cid)] = meta.get("title", f"Item-{cid}")
        mem = snapshot.item_memory(cid)
        if mem:
            memories[str(cid)] = mem

    return {
        "user_id": int(user_id),
        # candidate-blind state; the only thing the policy ever sees
        "prompt": rl_policy.build_prompt(
            user_id=user_id,
            user_memory=state.user_memory,
            neighbors_text=state.neighbors_text,
            n_facets=n_facets,
        ),
        "candidates": [int(c) for c in candidates],
        "gold_item_id": gold,
        "M_u": state.user_memory,
        "neighbors": state.neighbor_ids,
        "r_null": None,        # back-filled after M2
        "baseline_h1": None,   # back-filled after M2
        # --- reward-side context, never shown to the policy ---
        "instruction": instruction,
        "candidate_titles": titles,
        "candidate_memories": memories,
        "neighbor_ids": state.neighbor_ids,
        # {node_id: snippet} exactly as rendered in the prompt. The grounding
        # reward (§5.2) compares each facet against the text the policy actually
        # read, not against storage-side M_v -- see src/rl/env.py.
        "neighbor_snippets": parse_neighbor_snippets(state.neighbors_text),
        "n_train_items": state.n_train_items,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_name = args.dataset or config["dataset"]
    rl_cfg = config.get("rl", {})
    seed = config.get("seed", 42)
    n_candidates = config.get("n_eval_candidates", 10)
    n_facets = config.get("memrec", {}).get("n_facets", 7)
    out_prefix = args.out_prefix or rl_cfg["out_prefix"]

    print("Loading dataset...")
    data_file = PROJECT_ROOT / "data" / "processed" / dataset_name / f"{dataset_name}.inter"
    dataset = RecDataset(str(data_file), seed=seed)
    dataset.load_item_metadata()
    dataset.load_instructions()

    print("Loading snapshot + splits...")
    snapshot = GraphSnapshot.load(str(PROJECT_ROOT / rl_cfg["snapshot_file"]))
    splits = rl_splits.load_splits(str(PROJECT_ROOT / rl_cfg["splits_file"]))
    rl_splits.assert_disjoint(splits)
    print(f"  snapshot: {len(snapshot)} users, {len(snapshot.item_memories)} item memories")

    all_users = sorted(set().union(*splits.values()))
    candidates = rl_splits.build_candidates_for_users(
        dataset, all_users, n_candidates=n_candidates, seed=seed
    )

    counts = {}
    dropped_log = {}
    for split_name, users in splits.items():
        out_path = PROJECT_ROOT / f"{out_prefix}_{split_name}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = missing = 0
        leaked = []
        with open(out_path, "w", encoding="utf-8") as f:
            for user_id in users:
                if user_id not in snapshot.users or user_id not in candidates:
                    missing += 1
                    continue
                record = build_record(user_id, snapshot, dataset, candidates[user_id], n_facets)
                # Drop users whose prompt names the answer. See src/rl/leakage.py:
                # the Books catalogue has duplicate entries for the same book, so a
                # different item_id with the gold title can sit in the neighbour table.
                reason = gold_leak_reason(record)
                if reason:
                    leaked.append({"user_id": user_id, "reason": reason})
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
        counts[split_name] = written
        dropped_log[split_name] = leaked
        note = []
        if missing:
            note.append(f"{missing} not in snapshot")
        if leaked:
            note.append(f"{len(leaked)} dropped for gold leakage")
        print(f"  {split_name}: {written} records -> {out_path}"
              + (f"  ({', '.join(note)})" if note else ""))

    drop_path = PROJECT_ROOT / f"{out_prefix}_dropped_users.json"
    with open(drop_path, "w", encoding="utf-8") as f:
        json.dump(dropped_log, f, indent=2)
    print(f"  dropped-user log -> {drop_path}")

    print(f"\n✓ dataset built: {counts}")


if __name__ == "__main__":
    main()
