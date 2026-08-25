"""MH1 — build a bounded, candidate-blind 2-hop pool and validation bundles.

This command consumes only MH0's Stage-R control context and the immutable
pre-target topology.  It never reads the ranking instruction, candidate list or
gold item while constructing a pool or selecting a bundle.  Those fields are
used only afterwards by a leakage audit that can make a user ineligible for all
arms of a later comparison.

Run::

    python -m src.multihop.mh1 --config configs/multihop/mh0_books.yaml

The command is CPU-only and makes no LLM/API calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.packer import SnippetPacker
from src.rl.leakage import gold_leak_reason
from src.rl.policy import build_prompt

from .mh0 import SCHEMA_VERSION, canonical_hash, project_path, safe_relative, sha256


MH1_SCHEMA_VERSION = 1
_NODE_TYPES = ("item", "user")
_FORBIDDEN_STAGE_R_FIELDS = {"candidates", "candidate_titles", "candidate_memories", "gold_item_id", "instruction"}
_RENDERED_NODE = re.compile(r"(?m)^\d+\.\s+\[(Item|User)-(\d+)\]")


@dataclass(frozen=True)
class Topology:
    """Immutable pre-target bipartite graph produced by MH0."""

    user_items: Mapping[int, Tuple[int, ...]]
    item_users: Mapping[int, Tuple[int, ...]]

    @classmethod
    def load(cls, path: Path) -> "Topology":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unexpected MH0 topology schema: {path}")
        return cls(
            user_items={int(uid): tuple(map(int, items)) for uid, items in payload["user_items"].items()},
            item_users={int(iid): tuple(map(int, users)) for iid, users in payload["item_users"].items()},
        )

    def user_recency(self, user_id: int, item_id: int) -> float:
        """Same positional recency proxy used by ``UserItemGraph``."""
        history = self.user_items.get(int(user_id), ())
        if not history:
            return 0.0
        try:
            position = history.index(int(item_id))
        except ValueError:
            return 0.0
        return (position + 1) / len(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MH1 2-hop pools and validation bundles")
    parser.add_argument("--config", default="configs/multihop/mh0_books.yaml")
    parser.add_argument("--output-dir", default=None, help="override multihop.output_dir")
    parser.add_argument("--force", action="store_true", help="replace existing MH1 outputs")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_row(handle, digest: "hashlib._Hash", row: Mapping[str, Any]) -> None:
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    handle.write(line)
    digest.update(line.encode("utf-8"))


def node_id(node_type: str, value: int) -> str:
    if node_type not in _NODE_TYPES:
        raise ValueError(f"unknown node type: {node_type}")
    return f"{node_type.capitalize()}-{int(value)}"


def node_type_and_value(identifier: str) -> Tuple[str, int]:
    kind, separator, value = str(identifier).partition("-")
    if not separator or kind.lower() not in _NODE_TYPES or not value.isdigit():
        raise ValueError(f"malformed node id: {identifier!r}")
    return kind.lower(), int(value)


def _is_better_path(candidate: Mapping[str, Any], current: Mapping[str, Any] | None) -> bool:
    """Prefer a stronger path, then a lexicographically stable witness path."""
    if current is None:
        return True
    candidate_key = (float(candidate["path_strength"]), tuple(candidate["witness_path"]))
    current_key = (float(current["path_strength"]), tuple(current["witness_path"]))
    return candidate_key > current_key


def build_raw_pool(
    stage_r_context: Mapping[str, Any],
    topology: Topology,
    *,
    user_id: int | None = None,
    max_remote_items: int,
    max_remote_users: int,
    max_remote_users_per_item: int,
) -> List[Dict[str, Any]]:
    """Build C2(u) using only candidate-blind Stage-R context plus topology.

    A remote item follows ``u -> anchor item -> selected peer -> remote item``.
    A remote user follows that witness one edge farther through the remote item.
    The deterministic caps deliberately bound expansion before any LLM sees it.
    """
    user_id = int(user_id if user_id is not None else stage_r_context["user_id"])
    selected_ids = list(stage_r_context["selected_node_ids"])
    selected_set = set(selected_ids)
    user_history = set(topology.user_items.get(user_id, ()))
    if not user_history:
        return []

    peer_ids = [
        value
        for identifier in selected_ids
        for kind, value in [node_type_and_value(identifier)]
        if kind == "user" and value != user_id
    ]
    item_nodes: Dict[str, Dict[str, Any]] = {}
    history_length = len(user_history)

    for peer_id in peer_ids:
        peer_history = topology.user_items.get(peer_id, ())
        shared = sorted(user_history.intersection(peer_history))
        if not shared:
            continue
        overlap = len(shared) / history_length
        # The strongest first-hop witness is the user's most recent shared item.
        anchor = max(shared, key=lambda item: (topology.user_recency(user_id, item), -item))
        first_hop_strength = topology.user_recency(user_id, anchor) * overlap
        for remote_item in peer_history:
            identifier = node_id("item", remote_item)
            if remote_item in user_history or identifier in selected_set:
                continue
            strength = first_hop_strength * topology.user_recency(peer_id, remote_item)
            candidate = {
                "node_id": identifier,
                "node_type": "item",
                "id": int(remote_item),
                "path_strength": strength,
                "witness_path": [
                    node_id("user", user_id),
                    node_id("item", anchor),
                    node_id("user", peer_id),
                    identifier,
                ],
                "source_peer_id": peer_id,
                "anchor_item_id": anchor,
            }
            if _is_better_path(candidate, item_nodes.get(identifier)):
                item_nodes[identifier] = candidate

    ordered_items = sorted(
        item_nodes.values(), key=lambda node: (-float(node["path_strength"]), int(node["id"]))
    )[:max_remote_items]

    user_nodes: Dict[str, Dict[str, Any]] = {}
    for item_node in ordered_items:
        remote_item = int(item_node["id"])
        possible_users = [
            other_user
            for other_user in topology.item_users.get(remote_item, ())
            if other_user != user_id and node_id("user", other_user) not in selected_set
        ]
        possible_users.sort(
            key=lambda other_user: (-topology.user_recency(other_user, remote_item), other_user)
        )
        for remote_user in possible_users[:max_remote_users_per_item]:
            identifier = node_id("user", remote_user)
            candidate = {
                "node_id": identifier,
                "node_type": "user",
                "id": int(remote_user),
                "path_strength": float(item_node["path_strength"])
                * topology.user_recency(remote_user, remote_item),
                "witness_path": [*item_node["witness_path"], identifier],
                "source_peer_id": int(item_node["source_peer_id"]),
                "anchor_item_id": int(item_node["anchor_item_id"]),
                "via_item_id": remote_item,
            }
            if _is_better_path(candidate, user_nodes.get(identifier)):
                user_nodes[identifier] = candidate

    ordered_users = sorted(
        user_nodes.values(), key=lambda node: (-float(node["path_strength"]), int(node["id"]))
    )[:max_remote_users]
    return [*ordered_items, *ordered_users]


def needed_metadata_ids(pool: Sequence[Mapping[str, Any]], topology: Topology) -> set[int]:
    """Return only fields ``SnippetPacker`` needs for static remote snippets."""
    wanted: set[int] = set()
    for node in pool:
        if node["node_type"] == "item":
            wanted.add(int(node["id"]))
        else:
            wanted.update(topology.user_items.get(int(node["id"]), ())[-3:])
    return wanted


def load_metadata_subset(path: Path, wanted_ids: set[int]) -> Dict[int, Dict[str, str]]:
    """Scan metadata once, retaining only title/description prefixes used by packer."""
    metadata: Dict[int, Dict[str, str]] = {}
    # Amazon book descriptions can exceed the csv module's conservative 128 KiB
    # default even though the packer subsequently retains only 120 characters.
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"item_id", "title", "description"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"metadata TSV missing columns {sorted(required)}: {path}")
        for row in reader:
            item_id = int(row["item_id"])
            if item_id not in wanted_ids:
                continue
            metadata[item_id] = {
                "title": (row.get("title") or "")[:80],
                "description": (row.get("description") or "")[:120],
            }
    return metadata


def render_remote_pool(
    raw_pool: Sequence[Mapping[str, Any]], topology: Topology, metadata: Mapping[int, Mapping[str, str]]
) -> List[Dict[str, Any]]:
    """Render remote snippets with the repository's own ``SnippetPacker`` rule."""
    dataset = SimpleNamespace(item_metadata=dict(metadata), train_data=topology.user_items)
    packer = SnippetPacker()
    rendered = []
    for node in raw_pool:
        snippet = packer.build_neighbor_snippet(
            {"type": node["node_type"], "id": int(node["id"]), "score": float(node["path_strength"])},
            dataset,
        )
        rendered.append({**node, "snippet": snippet})
    return rendered


def baseline_nodes(stage_r_context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    neighbors_text = str(stage_r_context["neighbors_text"])
    matches = list(_RENDERED_NODE.finditer(neighbors_text))
    nodes = []
    for index, match in enumerate(matches):
        kind = match.group(1).lower()
        value = int(match.group(2))
        identifier = node_id(kind, value)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(neighbors_text)
        # Keep embedded newlines from metadata descriptions. ``parse_neighbor_snippets``
        # is deliberately line-oriented for grounding rewards and cannot restore them.
        snippet = f"[{identifier}]" + neighbors_text[match.end() : end].removesuffix("\n")
        nodes.append(
            {
                "node_id": identifier,
                "node_type": kind,
                "id": value,
                "snippet": snippet,
                "source": "one_hop",
            }
        )
    selected = list(stage_r_context["selected_node_ids"])
    if [node["node_id"] for node in nodes] != selected:
        raise ValueError("MH0 selected node IDs do not match rendered one-hop neighbor text")
    rendered = render_neighbors(nodes)
    if rendered != neighbors_text:
        raise ValueError("MH0 selected snippets no longer reconstruct the exact one-hop neighbor text")
    return nodes


def render_neighbors(nodes: Sequence[Mapping[str, Any]]) -> str:
    if not nodes:
        return "**Collaborative Neighbors:** (none available)"
    return "**Collaborative Neighbors:**\n" + "\n".join(
        f"{index}. {node['snippet']}" for index, node in enumerate(nodes, start=1)
    )


def estimate_tokens(text: str) -> int:
    """Match ``SnippetPacker.estimate_tokens`` exactly."""
    return len(text) // 4


def type_quota(quota: int, base_nodes: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(node["node_type"] for node in base_nodes)
    total = len(base_nodes)
    if not total:
        return {kind: 0 for kind in _NODE_TYPES}
    ideal = {kind: quota * counts[kind] / total for kind in _NODE_TYPES}
    target = {kind: min(counts[kind], int(ideal[kind])) for kind in _NODE_TYPES}
    remaining = quota - sum(target.values())
    for kind in sorted(_NODE_TYPES, key=lambda value: (-(ideal[value] - target[value]), value)):
        take = min(remaining, counts[kind] - target[kind])
        target[kind] += take
        remaining -= take
    return target


def _take_by_type(
    pool: Sequence[Mapping[str, Any]], targets: Mapping[str, int], *, randomizer: random.Random | None
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for kind in _NODE_TYPES:
        candidates = [dict(node) for node in pool if node["node_type"] == kind]
        if randomizer is not None:
            randomizer.shuffle(candidates)
        selected.extend(candidates[: int(targets[kind])])
    return selected


def _with_one_hop_fallback(
    base_nodes: Sequence[Mapping[str, Any]], remote_nodes: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Replace the lowest displayed control nodes type-for-type, keeping K fixed."""
    removal_counts = Counter(node["node_type"] for node in remote_nodes)
    remove_indices: set[int] = set()
    for kind, count in removal_counts.items():
        candidates = [index for index, node in enumerate(base_nodes) if node["node_type"] == kind]
        remove_indices.update(candidates[-count:])
    fallback = [dict(node) for index, node in enumerate(base_nodes) if index not in remove_indices]
    remote = [{**node, "source": "two_hop"} for node in remote_nodes]
    return [*fallback, *remote]


def materialize_bundle(
    *,
    user_id: int,
    arm: str,
    quota: int,
    variant: int | None,
    stage_r_context: Mapping[str, Any],
    base_nodes: Sequence[Mapping[str, Any]],
    requested_remote: Sequence[Mapping[str, Any]],
    n_facets: int,
) -> Dict[str, Any]:
    """Create a budget-valid Stage-R input, falling back to one-hop as needed."""
    k_target = int(stage_r_context["K_u"])
    token_target = int(stage_r_context["T_u"])
    remote = [dict(node) for node in requested_remote]
    budget_shortfall = 0
    while True:
        nodes = _with_one_hop_fallback(base_nodes, remote)
        neighbors_text = render_neighbors(nodes)
        token_actual = estimate_tokens(neighbors_text)
        if len(nodes) != k_target:
            raise AssertionError(f"{arm}: K mismatch {len(nodes)} != {k_target}")
        if token_actual <= token_target or not remote:
            break
        # Drop the most expensive remote snippet first. The replacement is the
        # corresponding 1-hop fallback, so this never exceeds the control budget.
        remove_index = max(
            range(len(remote)),
            key=lambda index: (len(str(remote[index]["snippet"])), str(remote[index]["node_id"])),
        )
        remote.pop(remove_index)
        budget_shortfall += 1

    prompt = build_prompt(
        user_id=int(user_id),
        user_memory=str(stage_r_context["user_memory"]),
        neighbors_text=neighbors_text,
        n_facets=n_facets,
    )
    stage_r = {
        "prompt": prompt,
        "prompt_sha256": canonical_hash(prompt),
        "neighbors_text": neighbors_text,
        "selected_node_ids": [node["node_id"] for node in nodes],
        "K_u": k_target,
        "T_u": token_target,
        "T_actual": token_actual,
    }
    if _FORBIDDEN_STAGE_R_FIELDS.intersection(stage_r):
        raise AssertionError("ranking fields leaked into stage_r payload")
    return {
        "arm": arm,
        "quota_requested": quota,
        "variant": variant,
        "remote_requested": len(requested_remote),
        "remote_actual": len(remote),
        "remote_shortfall": quota - len(remote),
        "remote_shortfall_budget": budget_shortfall,
        "remote_node_ids": [node["node_id"] for node in remote],
        "fallback_node_ids": [node["node_id"] for node in nodes if node["source"] == "one_hop"],
        "remote_paths": [
            {
                "node_id": node["node_id"],
                "path_strength": node["path_strength"],
                "witness_path": node["witness_path"],
            }
            for node in remote
        ],
        "type_mix": dict(Counter(node["node_type"] for node in nodes)),
        "stage_r": stage_r,
    }


def audit_bundle_for_ranking_leak(bundle: Mapping[str, Any], ranking_context: Mapping[str, Any]) -> str | None:
    """Audit *after* construction; this must never select/filter a bundle."""
    record = {
        "prompt": bundle["stage_r"]["prompt"],
        "candidates": ranking_context["candidates"],
        "gold_item_id": ranking_context["gold_item_id"],
        "candidate_titles": ranking_context.get("candidate_titles", {}),
    }
    return gold_leak_reason(record)


def audit_pool_for_ranking_leak(pool: Sequence[Mapping[str, Any]], ranking_context: Mapping[str, Any]) -> str | None:
    """Screen a completed C2 pool without using ranking data to alter it.

    If an unselected C2 node names a candidate/gold today, a later sampler could
    expose it.  The safe response is to exclude that user from *every* arm, not
    to remove the node with target-aware logic.
    """
    record = {
        "prompt": render_neighbors(pool),
        "candidates": ranking_context["candidates"],
        "gold_item_id": ranking_context["gold_item_id"],
        "candidate_titles": ranking_context.get("candidate_titles", {}),
    }
    return gold_leak_reason(record)


def build_validation_bundles(
    control: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    quotas: Sequence[int],
    oracle_bundles_per_quota: int,
    n_facets: int,
) -> Dict[str, Any]:
    """Materialize naive and 12 bounded-oracle Stage-R contexts for one user."""
    stage_r_context = control["stage_r_context"]
    user_id = int(control["user_id"]) if "user_id" in control else int(stage_r_context["user_id"])
    base = baseline_nodes(stage_r_context)
    if len(base) != int(stage_r_context["K_u"]):
        raise AssertionError("control K_u does not match selected node list")

    naive: Dict[str, Dict[str, Any]] = {}
    oracle: List[Dict[str, Any]] = []
    for quota in quotas:
        targets = type_quota(int(quota), base)
        naive_selected = _take_by_type(pool, targets, randomizer=None)
        naive[str(quota)] = materialize_bundle(
            user_id=user_id,
            arm="naive_two_hop",
            quota=int(quota),
            variant=None,
            stage_r_context=stage_r_context,
            base_nodes=base,
            requested_remote=naive_selected,
            n_facets=n_facets,
        )
        for variant in range(oracle_bundles_per_quota):
            randomizer = random.Random(f"{seed}:val:{user_id}:{quota}:{variant}")
            selected = _take_by_type(pool, targets, randomizer=randomizer)
            oracle.append(
                materialize_bundle(
                    user_id=user_id,
                    arm="oracle_two_hop",
                    quota=int(quota),
                    variant=variant,
                    stage_r_context=stage_r_context,
                    base_nodes=base,
                    requested_remote=selected,
                    n_facets=n_facets,
                )
            )

    all_bundles = [*naive.values(), *oracle]
    pool_reason = audit_pool_for_ranking_leak(pool, control["ranking_context"])
    reasons = ([f"C2 pool: {pool_reason}"] if pool_reason else []) + [
        f"{bundle['arm']}/q{bundle['quota_requested']}/v{bundle['variant']}: {reason}"
        for bundle in all_bundles
        if (reason := audit_bundle_for_ranking_leak(bundle, control["ranking_context"]))
    ]
    return {
        "schema_version": MH1_SCHEMA_VERSION,
        "split": "val",
        "user_id": user_id,
        # This is a hash-only evaluation contract. The actual candidates,
        # instruction and gold never enter a Stage-R payload or pool.
        "evaluation_contract": {
            "candidate_order_sha256": control["ranking_context"]["candidate_order_sha256"],
            "candidate_count": len(control["ranking_context"]["candidates"]),
            "candidate_blind_construction": True,
        },
        "one_hop": {
            "prompt_sha256": stage_r_context["prompt_sha256"],
            "selected_node_ids": list(stage_r_context["selected_node_ids"]),
            "K_u": int(stage_r_context["K_u"]),
            "T_u": int(stage_r_context["T_u"]),
        },
        "naive_two_hop": naive,
        "oracle_two_hop": oracle,
        "eligibility": {"eligible_all_arms": not reasons, "reasons": reasons},
    }


def pool_record(
    control: Mapping[str, Any], pool: Sequence[Mapping[str, Any]], *, split: str
) -> Dict[str, Any]:
    stage_r_context = control["stage_r_context"]
    user_id = int(control["user_id"]) if "user_id" in control else int(stage_r_context["user_id"])
    if any(field in stage_r_context for field in _FORBIDDEN_STAGE_R_FIELDS):
        raise AssertionError("MH0 Stage-R context contains ranking fields")
    return {
        "schema_version": MH1_SCHEMA_VERSION,
        "split": split,
        "user_id": user_id,
        "control_contract": {
            "stage_r_context_sha256": canonical_hash(stage_r_context),
            "one_hop_prompt_sha256": stage_r_context["prompt_sha256"],
            "one_hop_selected_node_ids": list(stage_r_context["selected_node_ids"]),
            "K_u": int(stage_r_context["K_u"]),
            "T_u": int(stage_r_context["T_u"]),
            "candidate_blind_construction": True,
        },
        "pool": {"nodes": list(pool), "n_items": sum(node["node_type"] == "item" for node in pool), "n_users": sum(node["node_type"] == "user" for node in pool)},
    }


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _config_int(config: Mapping[str, Any], key: str, minimum: int = 1) -> int:
    value = int(config[key])
    if value < minimum:
        raise ValueError(f"multihop.mh1.{key} must be >= {minimum}")
    return value


def _verify_mh0_contract(mh0_manifest: Mapping[str, Any], topology_path: Path, controls: Mapping[str, Path]) -> None:
    expected_topology = mh0_manifest["topology"]["sha256"]
    if sha256(topology_path) != expected_topology:
        raise ValueError("MH0 topology hash differs from its manifest")
    for split, path in controls.items():
        expected = mh0_manifest["control"][split]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"MH0 {split} control hash differs from its manifest")


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    mh_cfg = config.get("multihop", {})
    mh1_cfg = mh_cfg.get("mh1", {})
    output_dir = project_path(args.output_dir or mh_cfg.get("output_dir", "data/multihop"))
    topology_path = project_path(mh_cfg.get("topology_file", "data/multihop/mh0_topology_books.json"))
    metadata_path = project_path(
        mh_cfg.get("metadata_file", "data/processed/instructrec-books/instructrec-books.meta")
    )
    mh0_manifest_path = project_path(mh_cfg.get("mh0_manifest", "data/multihop/mh0_manifest.json"))
    controls = {
        split: project_path(mh_cfg.get(f"mh0_control_{split}", f"data/multihop/mh0_control_{split}.jsonl"))
        for split in ("train", "val", "test")
    }
    outputs = {
        **{split: output_dir / f"mh1_pools_{split}.jsonl" for split in controls},
        "bundles_val": output_dir / "mh1_bundles_val.jsonl",
        "manifest": output_dir / "mh1_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "MH1 outputs already exist: " + ", ".join(safe_relative(path) for path in existing)
            + ". Re-run with --force to replace them."
        )
    inputs = [config_path, topology_path, metadata_path, mh0_manifest_path, *controls.values()]
    missing = [safe_relative(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("MH1 required input(s) missing: " + ", ".join(missing))

    max_remote_items = _config_int(mh1_cfg, "max_remote_items")
    max_remote_users = _config_int(mh1_cfg, "max_remote_users")
    max_users_per_item = _config_int(mh1_cfg, "max_remote_users_per_item")
    oracle_per_quota = _config_int(mh1_cfg, "oracle_bundles_per_quota")
    quotas = tuple(int(value) for value in mh1_cfg["remote_quotas"])
    if not quotas or min(quotas) < 1 or len(set(quotas)) != len(quotas):
        raise ValueError("multihop.mh1.remote_quotas must be unique positive integers")
    n_facets = int(config.get("memrec", {}).get("n_facets", 7))
    seed = int(config.get("seed", 42))

    mh0_manifest = json.loads(mh0_manifest_path.read_text(encoding="utf-8"))
    _verify_mh0_contract(mh0_manifest, topology_path, controls)
    topology = Topology.load(topology_path)
    pool_kwargs = {
        "max_remote_items": max_remote_items,
        "max_remote_users": max_remote_users,
        "max_remote_users_per_item": max_users_per_item,
    }

    print("MH1: collecting item metadata required by bounded C2 pools")
    wanted_ids: set[int] = set()
    input_counts: Dict[str, int] = {}
    for split, path in controls.items():
        count = 0
        for control in iter_jsonl(path):
            raw_pool = build_raw_pool(
                control["stage_r_context"], topology, user_id=int(control["user_id"]), **pool_kwargs
            )
            wanted_ids.update(needed_metadata_ids(raw_pool, topology))
            count += 1
        input_counts[split] = count
    metadata = load_metadata_subset(metadata_path, wanted_ids)
    print(f"  controls: {input_counts}; metadata: {len(metadata)}/{len(wanted_ids)} requested items")

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_hashes: Dict[str, str] = {}
    pool_stats: Dict[str, Dict[str, Any]] = {}
    bundle_hash = ""
    bundle_count = 0
    eligible_count = 0
    ineligible_reasons: Counter[str] = Counter()
    print("MH1: materializing bounded pools and validation bundles")
    with outputs["bundles_val"].open("w", encoding="utf-8") as bundle_handle:
        bundle_digest = hashlib.sha256()
        for split, input_path in controls.items():
            count = total_items = total_users = 0
            short_pools = 0
            with outputs[split].open("w", encoding="utf-8") as pool_handle:
                digest = hashlib.sha256()
                for control in iter_jsonl(input_path):
                    raw_pool = build_raw_pool(
                        control["stage_r_context"], topology, user_id=int(control["user_id"]), **pool_kwargs
                    )
                    pool = render_remote_pool(raw_pool, topology, metadata)
                    record = pool_record(control, pool, split=split)
                    write_jsonl_row(pool_handle, digest, record)
                    count += 1
                    total_items += record["pool"]["n_items"]
                    total_users += record["pool"]["n_users"]
                    short_pools += int(len(pool) < max(quotas))
                    if split == "val":
                        bundle = build_validation_bundles(
                            control,
                            pool,
                            seed=seed,
                            quotas=quotas,
                            oracle_bundles_per_quota=oracle_per_quota,
                            n_facets=n_facets,
                        )
                        write_jsonl_row(bundle_handle, bundle_digest, bundle)
                        bundle_count += 1
                        if bundle["eligibility"]["eligible_all_arms"]:
                            eligible_count += 1
                        else:
                            for reason in bundle["eligibility"]["reasons"]:
                                ineligible_reasons[reason.split(": ", 1)[-1]] += 1
            pool_hashes[split] = digest.hexdigest()
            pool_stats[split] = {
                "records": count,
                "sha256": pool_hashes[split],
                "mean_remote_items": total_items / count if count else 0.0,
                "mean_remote_users": total_users / count if count else 0.0,
                "pool_shorter_than_max_quota": short_pools,
                "path": safe_relative(outputs[split]),
            }
            print(f"  {split}: {count} pools -> {safe_relative(outputs[split])}")
        bundle_hash = bundle_digest.hexdigest()
    expected_bundles = len(quotas) * oracle_per_quota
    if bundle_count != input_counts["val"]:
        raise AssertionError(f"wrote {bundle_count} validation bundle records, expected {input_counts['val']}")

    manifest = {
        "schema_version": MH1_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config": {"path": safe_relative(config_path), "sha256": sha256(config_path), "seed": seed},
        "input": {
            "mh0_manifest": {"path": safe_relative(mh0_manifest_path), "sha256": sha256(mh0_manifest_path)},
            "topology": {"path": safe_relative(topology_path), "sha256": sha256(topology_path)},
            "controls": {split: {"path": safe_relative(path), "sha256": sha256(path)} for split, path in controls.items()},
            "metadata": {"path": safe_relative(metadata_path), "sha256": sha256(metadata_path), "loaded_items": len(metadata)},
        },
        "pool_policy": {
            "candidate_blind": True,
            "selected_peer_source": "MH0 stage_r_context.selected_node_ids of type User",
            "max_remote_items": max_remote_items,
            "max_remote_users": max_remote_users,
            "max_remote_users_per_item": max_users_per_item,
            "path_strength": "recency(u, anchor) * overlap(u, peer) * recency(peer, remote); remote-user paths multiply recency(remote-user, remote-item)",
        },
        "pools": pool_stats,
        "validation_bundles": {
            "path": safe_relative(outputs["bundles_val"]),
            "sha256": bundle_hash,
            "records": bundle_count,
            "naive_bundles_per_user": len(quotas),
            "oracle_bundles_per_user": expected_bundles,
            "quotas": list(quotas),
            "eligible_all_arms": eligible_count,
            "ineligible_all_arms": bundle_count - eligible_count,
            "ineligible_reason_counts": dict(sorted(ineligible_reasons.items())),
        },
        "execution": {"llm_calls": 0, "api_cost_usd": 0.0, "note": "MH1 is an offline CPU-only artifact build."},
    }
    outputs["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  val bundles: {bundle_count} records, {expected_bundles} oracle/user -> {safe_relative(outputs['bundles_val'])}")
    print(f"MH1 complete: {safe_relative(outputs['manifest'])}")


if __name__ == "__main__":
    main()
