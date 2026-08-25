"""
MH0 — freeze and validate the selective multi-hop experiment inputs.

This command is deliberately CPU-only and makes no LLM calls. It turns the
existing frozen memory snapshot plus pre-target interaction history into:

* an immutable topology artifact;
* one materialized, candidate-blind 1-hop control per split; and
* a manifest with every input/output hash and validation outcome.

Run:

    python -m src.multihop.mh0 --config configs/multihop/mh0_books.yaml

Existing output is never overwritten unless --force is supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.env import GraphSnapshot, UserState, parse_neighbor_snippets
from src.rl.leakage import gold_leak_reason
from src.rl.policy import build_prompt
from src.rl.splits import assert_disjoint


SCHEMA_VERSION = 1

# These profiles document the two known local RL-data states. A match is evidence
# that data is internally consistent; it does not make the old GRPO plan active.
KNOWN_INPUT_PROFILES: Mapping[str, Mapping[str, str]] = {
    "legacy_pre_m2_backfill": {
        "data/rl/stager_books_train.jsonl": "5b76f77c4986cf6964c02dded127ae0747a1f217a326e726a846c21e094732b4",
        "data/rl/stager_books_val.jsonl": "584e5251ce6da73a40fbf3d3e444dc97595a31f25c3b7a7a61f260b1d76a042d",
        "data/rl/stager_books_test.jsonl": "8e62f6899ce328c99c70d53133eed1cd0e8841071d1e7acce94b0293b318c94e",
        "data/rl/m2_val_reference_books.json": "3e8e287bbe309c7052a8d2ee7e2f85180fe5ed0291e588ceec0ae8b32e4d14cc",
    },
    "m2_backfill_bundle": {
        "data/rl/stager_books_train.jsonl": "2b51ced25e5f0886f6facd135134602256723b6114901464aeeed28f8333a6da",
        "data/rl/stager_books_val.jsonl": "a1488c8176a29faff7994c4e1b0d3c34f84926b220a659bca78ea0af35a353ed",
        "data/rl/stager_books_test.jsonl": "964a7a3076658eb7924e65b77981f66aa0a70d4998199b8dc64ef40b364ce1bf",
        "data/rl/m2_val_reference_books.json": "7c8725826c5b628208ca6584a8a5eb811ec413db96eb83cf6a8b67189fcfdadf",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze MH0 topology and 1-hop controls")
    parser.add_argument(
        "--config",
        default="configs/multihop/mh0_books.yaml",
        help="YAML containing the multihop input paths",
    )
    parser.add_argument("--output-dir", default=None, help="override multihop.output_dir")
    parser.add_argument("--force", action="store_true", help="replace existing MH0 artifacts")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            line = encoded + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def read_pre_target_history(path: Path) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], int]:
    """Recreate RecDataset.train_data without importing torch-dependent modules."""
    events: MutableMapping[int, List[Tuple[float, int, int]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"user_id", "item_id", "timestamp"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"{path} must have TSV columns {sorted(required)}")
        for order, row in enumerate(reader):
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            timestamp = float(row["timestamp"])
            events[user_id].append((timestamp, order, item_id))

    user_items: Dict[int, List[int]] = {}
    item_users: MutableMapping[int, List[int]] = defaultdict(list)
    for user_id, values in events.items():
        values.sort(key=lambda event: (event[0], event[1]))
        if len(values) < 3:
            continue
        train_items = [item_id for _, _, item_id in values[:-2]]
        user_items[user_id] = train_items
        for item_id in train_items:
            item_users[item_id].append(user_id)

    return user_items, dict(item_users), sum(len(items) for items in user_items.values())


def topology_payload(
    interaction_path: Path,
    user_items: Mapping[int, Sequence[int]],
    item_users: Mapping[int, Sequence[int]],
    edge_count: int,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "interaction_file": str(interaction_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(interaction_path),
            "split_rule": "timestamp sort, then all but each user's final two interactions",
        },
        "stats": {
            "n_users": len(user_items),
            "n_items": len(item_users),
            "n_edges": edge_count,
        },
        "user_items": {str(user_id): list(items) for user_id, items in sorted(user_items.items())},
        "item_users": {str(item_id): list(users) for item_id, users in sorted(item_users.items())},
    }


def input_profile(input_hashes: Mapping[str, str]) -> str:
    for name, expected in KNOWN_INPUT_PROFILES.items():
        if all(input_hashes.get(path) == digest for path, digest in expected.items()):
            return name
    return "unknown"


def control_row(split: str, record: Mapping[str, Any], snapshot: GraphSnapshot) -> Dict[str, Any]:
    user_id = int(record["user_id"])
    state = snapshot.state(user_id)
    prompt = build_prompt(
        user_id=user_id,
        user_memory=state.user_memory,
        neighbors_text=state.neighbors_text,
        n_facets=7,
    )
    if record.get("prompt") != prompt:
        raise ValueError(f"{split}/user {user_id}: source prompt differs from frozen snapshot")
    leak = gold_leak_reason(dict(record))
    if leak:
        raise ValueError(f"{split}/user {user_id}: source prompt leakage: {leak}")

    snippets = parse_neighbor_snippets(state.neighbors_text)
    selected_ids = list(snippets)
    if not selected_ids:
        raise ValueError(f"{split}/user {user_id}: no packed 1-hop snippets")
    if not set(selected_ids) <= set(state.neighbor_ids):
        raise ValueError(f"{split}/user {user_id}: packed snippets are not in the pruned neighbor set")

    candidates = [int(candidate) for candidate in record["candidates"]]
    if len(candidates) != 10 or int(record["gold_item_id"]) not in candidates:
        raise ValueError(f"{split}/user {user_id}: malformed fixed candidate list")

    # Stage-R context is intentionally candidate-blind. Ranking-only values stay
    # under ranking_context, so later selector code can enforce its input contract.
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "user_id": user_id,
        "stage_r_context": {
            "prompt": prompt,
            "prompt_sha256": canonical_hash(prompt),
            "user_memory": state.user_memory,
            "neighbors_text": state.neighbors_text,
            "selected_node_ids": selected_ids,
            "neighbor_snippets": snippets,
            "K_u": len(selected_ids),
            "T_u": len(state.neighbors_text) // 4,
        },
        "ranking_context": {
            "candidates": candidates,
            "candidate_order_sha256": canonical_hash(candidates),
            "gold_item_id": int(record["gold_item_id"]),
            "instruction": record.get("instruction", ""),
            "candidate_titles": record.get("candidate_titles", {}),
            "candidate_memories": record.get("candidate_memories", {}),
        },
    }


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    mh_cfg = config.get("multihop", {})
    snapshot_path = project_path(mh_cfg.get("snapshot_file", "data/rl/graph_snapshot_books.json"))
    records_prefix = project_path(mh_cfg.get("records_prefix", "data/rl/stager_books"))
    interaction_path = project_path(
        mh_cfg.get(
            "interaction_file",
            "data/processed/instructrec-books/instructrec-books.inter",
        )
    )
    output_dir = project_path(args.output_dir or mh_cfg.get("output_dir", "data/multihop"))
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_path = output_dir / "mh0_topology_books.json"
    manifest_path = output_dir / "mh0_manifest.json"
    control_paths = {
        split: output_dir / f"mh0_control_{split}.jsonl"
        for split in ("train", "val", "test")
    }
    targets = [topology_path, manifest_path, *control_paths.values()]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        names = ", ".join(safe_relative(path) for path in existing)
        raise FileExistsError(f"MH0 outputs already exist: {names}. Re-run with --force to replace them.")

    required_paths = [
        snapshot_path,
        interaction_path,
        Path(f"{records_prefix}_train.jsonl"),
        Path(f"{records_prefix}_val.jsonl"),
        Path(f"{records_prefix}_test.jsonl"),
        PROJECT_ROOT / "data/rl/m2_val_reference_books.json",
    ]
    missing = [safe_relative(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"MH0 required input(s) missing: {', '.join(missing)}")

    print("MH0: loading frozen memory snapshot and fixed records")
    snapshot = GraphSnapshot.load(str(snapshot_path))
    records = {
        split: load_jsonl(Path(f"{records_prefix}_{split}.jsonl"))
        for split in ("train", "val", "test")
    }
    split_ids = {
        split: [int(record["user_id"]) for record in rows]
        for split, rows in records.items()
    }
    assert_disjoint(split_ids)

    input_hashes = {safe_relative(path): sha256(path) for path in required_paths}
    profile = input_profile(input_hashes)
    print(f"  input profile: {profile}")
    if profile == "unknown":
        print("  warning: hashes do not match a known RL-data profile; manifest will preserve exact values.")

    print("MH0: rebuilding immutable pre-target topology")
    user_items, item_users, edge_count = read_pre_target_history(interaction_path)
    topology = topology_payload(interaction_path, user_items, item_users, edge_count)
    with topology_path.open("w", encoding="utf-8") as handle:
        json.dump(topology, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    topology_hash = sha256(topology_path)
    print(
        f"  topology: {len(user_items)} users, {len(item_users)} items, "
        f"{edge_count} edges -> {safe_relative(topology_path)}"
    )

    print("MH0: materializing candidate-blind 1-hop controls")
    control_summary: Dict[str, Dict[str, Any]] = {}
    for split, rows in records.items():
        materialized = [control_row(split, row, snapshot) for row in rows]
        count, digest = write_jsonl(control_paths[split], materialized)
        if count != len(rows):
            raise AssertionError(f"{split}: wrote {count}, expected {len(rows)} controls")
        k_values = [row["stage_r_context"]["K_u"] for row in materialized]
        t_values = [row["stage_r_context"]["T_u"] for row in materialized]
        control_summary[split] = {
            "records": count,
            "sha256": digest,
            "K_u": {"min": min(k_values), "max": max(k_values), "mean": sum(k_values) / count},
            "T_u": {"min": min(t_values), "max": max(t_values), "mean": sum(t_values) / count},
            "path": safe_relative(control_paths[split]),
        }
        print(f"  {split}: {count} controls -> {safe_relative(control_paths[split])}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config": {
            "path": safe_relative(config_path),
            "sha256": sha256(config_path),
            "dataset": config.get("dataset"),
            "seed": config.get("seed"),
            "llm": {
                "provider": config.get("provider", {}).get("name"),
                "model_env": "LLM__MODEL_NAME",
                "api_version_env": "LLM__API_VERSION",
            },
        },
        "input_profile": profile,
        "input_hashes": input_hashes,
        "snapshot": {
            "path": safe_relative(snapshot_path),
            "users": len(snapshot),
            "item_memories": len(snapshot.item_memories),
        },
        "topology": {
            "path": safe_relative(topology_path),
            "sha256": topology_hash,
            **topology["stats"],
        },
        "control": control_summary,
        "execution": {
            "llm_calls": 0,
            "api_cost_usd": 0.0,
            "note": "MH0 is an offline CPU-only validation/materialization step.",
        },
        "validation": {
            "split_counts": {split: len(rows) for split, rows in records.items()},
            "split_disjoint": True,
            "stage_r_candidate_blind": True,
            "source_prompt_leakage": 0,
            "candidate_count": 10,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"MH0 complete: {safe_relative(manifest_path)}")


if __name__ == "__main__":
    main()
