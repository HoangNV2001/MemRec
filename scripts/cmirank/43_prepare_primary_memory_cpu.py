#!/usr/bin/env python3
"""Fresh primary20 train users, real candidates/metadata, fake API wiring only."""

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cluster_runtime import require_allocation, require_project_root
from src.cmirank.primary_memory import primary_user_ids, primary_code_hashes, reviewed_ppo_mechanics
from src.cmirank.memory_smoke import load_memory_contract, run_memory_smoke
from src.cmirank.policy_inputs import load_locked_policy_inputs
from src.cmirank.candidate_artifacts import load_candidate_samplers
from src.cmirank.shortcut_audit import verify_candidate_run
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.optimizer_smoke import allocation_memory
from src.data.dataset_base import RecDataset
from src.models.memrec_agent import MemRecAgent
from src.models.llm_client import validate_json_shape
from scripts.smoke_full_memrec_cpu import FakeJSONClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    private = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    job = require_allocation()
    c, contract_sha = load_memory_contract(ROOT, 8)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = args.run_dir.resolve()
    if (run != private/"runs"/c["prepare_run_id"] or run.exists()
            or commit != os.environ["MEMREC_EXPECTED_COMMIT"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(sys.prefix).resolve() != private/"envs/llm-hnv"
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Fresh exact-source CUDA-hidden baseline-environment CPU smoke required")
    began = time.monotonic()
    memory = allocation_memory(job)
    if memory["maximum_bytes"] != 64*2**30:
        raise ValueError("Requalify a changed host cap; never modify the allocation/cgroup")
    reviewed_ppo_mechanics(private, c)
    inputs = load_locked_policy_inputs(ROOT)
    users = primary_user_ids(inputs, c)
    recipes = load_candidate_samplers(ROOT, private/"runs/cmirank-minilm-candidate-index-v2-20261001-hnv",
        inputs["snapshot"], version=2)
    rows = verify_candidate_run(private/"runs"/c["candidate_run_id"], c, inputs, recipes)
    spec = importlib.util.spec_from_file_location("memrec_memory_cpu_runtime", ROOT/"scripts/cmirank/10_smoke_real_memory.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.verify_audit(private/"runs"/c["audit_run_id"], c, inputs, recipes)
    metadata = ROOT/"data/processed/instructrec-books/instructrec-books.meta"
    if file_sha256(metadata) != recipes["config"]["metadata_sha256"]:
        raise ValueError("Static metadata changed")
    holder = SimpleNamespace(data_path=metadata.with_suffix(".inter"), item_metadata=None)
    RecDataset.load_item_metadata(holder)  # Never initialize the suffix-aware dataset.
    snapshot = replace(inputs["snapshot"], item_metadata=holder.item_metadata)
    import torch
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    torch.set_num_threads(4)
    model = private/"models/Qwen3-30B-A3B-Instruct-2507-FP8-hnv"
    marker = json.loads((model/"download-complete-hnv.json").read_text())
    if marker["revision"] != c["model_revision"] or marker["model"] != c["model_id"]:
        raise ValueError("Primary baseline teacher changed")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    payloads, token_lengths = [], []
    class SchemaFake(FakeJSONClient):
        def generate_json(self, messages, properties, **kwargs):
            payload = {"model": c["model_id"], "messages": messages,
                "response_format": {"type": "json_schema", "json_schema": {"name": "response", "strict": True,
                    "schema": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}},
                "temperature": kwargs["temperature"], "max_tokens": kwargs["max_tokens"]}
            ChatCompletionRequest(**payload)
            raw = super().generate_json(messages, properties, **kwargs)
            validate_json_shape(raw, payload["response_format"]["json_schema"]["schema"])
            n = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            if n+c["max_tokens"] > c["max_model_len"]:
                raise ValueError("Complete primary prompt exceeds original context cap")
            payloads.append(payload); token_lengths.append(n)
            self.last_request = payload
            return raw
    client = SchemaFake()
    agent = MemRecAgent(snapshot, client, temperature=c["temperature"], max_tokens=c["max_tokens"], **c["agent"])
    journals = {}
    result = run_memory_smoke(agent, inputs=inputs, rows=rows, user_ids=users,
        emit=lambda name, row: journals.setdefault(name, []).append(row), candidate_context_warning=True)
    if (client.total_requests != 100 or len(journals["policy-inputs"]) != 20
            or torch.cuda.is_initialized() or not set(users).isdisjoint(inputs["groups"]["policy_val"])):
        raise ValueError("CPU wiring/target-blind train-only cohort changed")
    run.mkdir()
    for name, values in {**journals, "fake-api-requests": payloads}.items():
        (run/f"{name}.jsonl").write_text("".join(json.dumps(v, sort_keys=True, ensure_ascii=False)+"\n" for v in values))
    report = {"status": "PRIMARY_TRAIN20_CPU_WIRING_PASS_REAL_MEMORY_GATE_REMAINS", "source_commit": commit,
        "config_sha256": contract_sha, "code_sha256": primary_code_hashes(ROOT), "slurm_job_id": job,
        "allocation_memory": memory,
        "query_user_ids": users, "graph_snapshot_sha256": inputs["snapshot_sha256"],
        "split_sha256": inputs["manifest_sha256"], "fake_logical_requests": 100, "real_llm_requests": 0,
        "gpu_requested": False, "training_ready": False, "full_1797_memory_cache": False,
        "train_users": 1497, "validation_users": 300, "selected_validation_users": 0,
        "max_prompt_tokens": max(token_lengths), "min_prompt_tokens": min(token_lengths),
        "teacher_marker_sha256": file_sha256(model/"download-complete-hnv.json"),
        "fake_wiring_summary": result,
        "artifact_sha256": {p.name: file_sha256(p) for p in run.glob("*.jsonl")},
        "elapsed_seconds": time.monotonic()-began}
    (run/"report.json").write_text(artifact_json_dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "query_user_ids", "max_prompt_tokens", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
