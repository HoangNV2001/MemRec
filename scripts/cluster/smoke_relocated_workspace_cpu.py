#!/usr/bin/env python3
"""Twenty locked policy users in the relocated workspace; fake LLM, no CUDA."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from src.cluster_runtime import CLUSTER_ROOT, require_allocation
    from src.cmirank.candidate_artifacts import load_candidate_samplers, smoke_user_ids
    from src.cmirank.memory_smoke import load_memory_contract, run_memory_smoke
    from src.cmirank.policy_inputs import load_locked_policy_inputs
    from src.cmirank.provenance import artifact_json_dumps, file_sha256
    from src.cmirank.shortcut_audit import verify_candidate_run
    from src.data.dataset_base import RecDataset
    from src.models.memrec_agent import MemRecAgent
    from scripts.smoke_full_memrec_cpu import FakeJSONClient
    import torch
    require_allocation()
    if args.repo.resolve() != CLUSTER_ROOT / "repo/MemRec-hnv" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Unexpected migrated source or CUDA visibility")
    args.output_dir.mkdir()
    inputs = load_locked_policy_inputs(args.repo)
    config, _ = load_memory_contract(args.repo, 4)
    recipes = load_candidate_samplers(args.repo, CLUSTER_ROOT / "runs/cmirank-minilm-candidate-index-v2-20261001-hnv",
                                       inputs["snapshot"], version=2)
    rows = verify_candidate_run(CLUSTER_ROOT / "runs" / config["candidate_run_id"], config, inputs, recipes)
    users = smoke_user_ids(sorted(inputs["warmups"]), recipes["episode_config"])
    metadata = SimpleNamespace(data_path=args.repo / "data/processed/instructrec-books/instructrec-books.inter", item_metadata=None)
    RecDataset.load_item_metadata(metadata)
    if file_sha256(metadata.data_path.with_suffix(".meta")) != recipes["config"]["metadata_sha256"]:
        raise RuntimeError("Migrated metadata differs from the frozen source")
    snapshot = replace(inputs["snapshot"], item_metadata=metadata.item_metadata)
    client = FakeJSONClient()
    agent = MemRecAgent(snapshot, client, temperature=config["temperature"], max_tokens=config["max_tokens"], **config["agent"])
    def emit(name, row):
        with (args.output_dir / f"{name}.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result = run_memory_smoke(agent, inputs=inputs, rows=rows, user_ids=users, emit=emit, candidate_context_warning=True)
    if client.total_physical_requests != 100 or torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke must use exactly 100 fake calls and no CUDA")
    result.update(status="RELOCATED_WORKSPACE_CPU_SMOKE_PASS_NOT_REAL_LM_OR_PPO",
        graph_snapshot_sha256=inputs["snapshot_sha256"], fake_logical_calls=100, real_llm_requests=0, gpu_requested=False)
    (args.output_dir / "report.json").write_text(artifact_json_dumps(result))
    print(artifact_json_dumps(result))


if __name__ == "__main__":
    main()
