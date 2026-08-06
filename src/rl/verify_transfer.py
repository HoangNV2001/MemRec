"""
Verify that the RL data bundles landed correctly on a new machine.

Run this immediately after unpacking, before anything else:

    python -m src.rl.verify_transfer

Checks content, not just presence -- a truncated scp leaves a file that exists,
opens, and is wrong. Every invariant here is one that M2/M3/M4 silently depend on,
so a failure now is worth far more than the same failure three hours into a
rented GPU session.

Exits non-zero if any REQUIRED check fails. Optional artefacts are reported but
never fail the run.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import argparse
import hashlib
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (path, expected line count or None, sha256, required)
EXPECTED = [
    ("data/rl/stager_books_train.jsonl", 1185,
     "5b76f77c4986cf6964c02dded127ae0747a1f217a326e726a846c21e094732b4", True),
    ("data/rl/stager_books_test.jsonl", 993,
     "8e62f6899ce328c99c70d53133eed1cd0e8841071d1e7acce94b0293b318c94e", True),
    ("data/rl/stager_books_val.jsonl", 149,
     "584e5251ce6da73a40fbf3d3e444dc97595a31f25c3b7a7a61f260b1d76a042d", True),
    ("data/rl/m2_val_reference_books.json", None,
     "3e8e287bbe309c7052a8d2ee7e2f85180fe5ed0291e588ceec0ae8b32e4d14cc", True),
    ("data/rl/user_splits_books.json", None, None, True),          # tracked in git
    ("data/rl/graph_snapshot_books.json", None,
     "baa83950968699af9e26b0695a8cc5dca82265c43047643643e6b9646a2d5c3d", False),
    ("results/m1_warmup_2350/memory_warmup_only.json", None,
     "8ca8c9d848b8ac64160944183182cd1337d721688d4710bba0690ebdf256bf09", False),
]

# Columns the reward function reads. A missing one shows up at training time as
# "every rollout is malformed", which is a miserable thing to debug on a clock.
REQUIRED_COLUMNS = {
    "user_id", "prompt", "candidates", "gold_item_id", "M_u", "neighbors",
    "neighbor_snippets", "candidate_titles", "candidate_memories", "instruction",
    "r_null", "baseline_h1",
}

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Report:
    def __init__(self):
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def ok(self, msg: str):
        print(f"  {GREEN}PASS{RESET}  {msg}")

    def fail(self, msg: str):
        print(f"  {RED}FAIL{RESET}  {msg}")
        self.failures.append(msg)

    def warn(self, msg: str):
        print(f"  {YELLOW}SKIP{RESET}  {msg}")
        self.warnings.append(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_files(rep: Report, verify_hashes: bool) -> None:
    print("\n[1/4] Files present, complete, and uncorrupted")
    for rel, n_lines, digest, required in EXPECTED:
        path = PROJECT_ROOT / rel
        if not path.exists():
            (rep.fail if required else rep.warn)(
                f"{rel} missing" + ("" if required else " (optional)")
            )
            continue

        size_mb = path.stat().st_size / 1e6
        if n_lines is not None:
            actual = sum(1 for _ in open(path, "r", encoding="utf-8"))
            if actual != n_lines:
                rep.fail(f"{rel}: {actual} lines, expected {n_lines} "
                         f"(truncated transfer?)")
                continue

        if verify_hashes and digest:
            actual_digest = sha256(path)
            if actual_digest != digest:
                rep.fail(f"{rel}: sha256 mismatch\n"
                         f"           got      {actual_digest}\n"
                         f"           expected {digest}")
                continue

        rep.ok(f"{rel}  ({size_mb:.1f} MB"
               + (f", {n_lines} lines" if n_lines else "") + ")")


def check_schema(rep: Report) -> None:
    print("\n[2/4] Records carry every column the reward reads")
    from src.rl.dataset import load_records

    for split in ("train", "val", "test"):
        path = PROJECT_ROOT / f"data/rl/stager_books_{split}.jsonl"
        if not path.exists():
            continue
        try:
            records = load_records(str(path))
        except Exception as exc:  # noqa: BLE001
            rep.fail(f"{split}: cannot parse jsonl — {type(exc).__name__}: {exc}")
            continue

        missing = REQUIRED_COLUMNS - set(records[0])
        if missing:
            rep.fail(f"{split}: records missing columns {sorted(missing)}")
            continue

        bad = [r["user_id"] for r in records if r["gold_item_id"] not in r["candidates"]]
        if bad:
            rep.fail(f"{split}: {len(bad)} records whose gold is not among candidates")
            continue

        empty_snippets = sum(1 for r in records if not r.get("neighbor_snippets"))
        note = f", {empty_snippets} with no neighbour snippets" if empty_snippets else ""
        rep.ok(f"{split}: {len(records)} records, all columns present{note}")


def check_splits(rep: Report) -> None:
    print("\n[3/4] Splits still user-disjoint after transfer")
    from src.rl.dataset import load_records
    from src.rl.splits import assert_disjoint

    ids = {}
    for split in ("train", "val", "test"):
        path = PROJECT_ROOT / f"data/rl/stager_books_{split}.jsonl"
        if path.exists():
            ids[split] = sorted({r["user_id"] for r in load_records(str(path))})
    try:
        assert_disjoint(ids)
        rep.ok(f"train/val/test disjoint ({'/'.join(str(len(v)) for v in ids.values())} users)")
    except AssertionError as exc:
        rep.fail(str(exc))


def check_reference(rep: Report) -> None:
    print("\n[4/4] Cached gpt-4o-mini reference is usable")
    path = PROJECT_ROOT / "data/rl/m2_val_reference_books.json"
    if not path.exists():
        rep.fail("m2_val_reference_books.json missing — M2 Part B would have to "
                 "re-spend ~$0.5 and 6 min of API to rebuild it")
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            ref = json.load(f)
    except Exception as exc:  # noqa: BLE001
        rep.fail(f"reference cache will not parse — {type(exc).__name__}: {exc}")
        return

    arms = set(ref["meta"]["arms"])
    expected_arms = {"sample1", "sample2", "shuffled", "lorem", "empty"}
    if arms != expected_arms:
        rep.fail(f"reference arms {sorted(arms)} != {sorted(expected_arms)}")
        return

    n_users = len(ref["scores"])
    n_scored = sum(1 for u in ref["scores"] for a in ref["scores"][u]
                   if "ndcg_at_5" in ref["scores"][u][a])
    if n_scored < n_users * len(expected_arms):
        rep.warn(f"{n_scored}/{n_users * len(expected_arms)} (user, arm) pairs scored "
                 f"— some arms errored during generation")
    else:
        rep.ok(f"{n_users} users x {len(arms)} arms = {n_scored} scored pairs")

    means = {
        a: sum(ref["scores"][u][a]["ndcg_at_5"] for u in ref["scores"]
               if "ndcg_at_5" in ref["scores"][u].get(a, {}))
           / max(1, sum(1 for u in ref["scores"] if "ndcg_at_5" in ref["scores"][u].get(a, {})))
        for a in expected_arms
    }
    print(f"  {DIM}NDCG@5 by arm: "
          + "  ".join(f"{a}={means[a]:.4f}" for a in ["sample1", "shuffled", "empty"])
          + RESET)
    if means["sample1"] <= means["empty"]:
        rep.fail("reference looks wrong: real memory does not beat empty memory")


def main():
    ap = argparse.ArgumentParser(description="Verify transferred RL data (see docs/DEPLOY_GPU.md)")
    ap.add_argument("--skip_hashes", action="store_true",
                    help="skip sha256 (faster; line counts still catch truncation)")
    args = ap.parse_args()

    print(f"Verifying RL data under {PROJECT_ROOT}")
    rep = Report()
    check_files(rep, verify_hashes=not args.skip_hashes)
    check_schema(rep)
    check_splits(rep)
    check_reference(rep)

    print("\n" + "=" * 66)
    if rep.failures:
        print(f"{RED}{len(rep.failures)} REQUIRED CHECK(S) FAILED{RESET} — do not start a GPU run.")
        for f in rep.failures:
            print(f"  - {f.splitlines()[0]}")
        print("\nSee docs/DEPLOY_GPU.md §1 for what each bundle should contain.")
        sys.exit(1)

    print(f"{GREEN}All required checks passed.{RESET}")
    if rep.warnings:
        print(f"{YELLOW}{len(rep.warnings)} optional artefact(s) absent{RESET} — fine for "
              f"M2-B/M3/M4, but see docs/DEPLOY_GPU.md §1.4:")
        for w in rep.warnings:
            print(f"  - {w}")
    print("\nNext:  pytest tests/rl/ -q            # ~30s, CPU only")
    print("       bash scripts/rl/02_validate_reward.sh hf")


if __name__ == "__main__":
    main()
