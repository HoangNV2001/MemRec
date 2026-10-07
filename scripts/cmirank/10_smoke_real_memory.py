#!/usr/bin/env python3
"""20-user real LM_Mem/LM_Rec smoke inside the authorized shared Slurm step.

CPU provenance/metadata preparation precedes GPU selection and model loading.
One owned model process group, bounded lifetime, unconditional release audit.
This is neither a full memory cache nor SLM/PPO training or held-out evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cmirank.candidate_artifacts import load_candidate_samplers, smoke_user_ids
from src.cmirank.gpu_resources import (
    canonical_gpu_uuid, compute_gpu_processes, parse_gpu_snapshot, select_reserved_gpu1, verified_numeric_cuda_binding,
)
from src.cmirank.memory_smoke import load_memory_contract, object_sha256, run_memory_smoke
from src.cmirank.policy_inputs import load_locked_policy_inputs
from src.cmirank.provenance import artifact_json_dumps, file_sha256
from src.cmirank.shortcut_audit import verify_candidate_run
from src.data.dataset_base import RecDataset
from src.models.llm_client import DurableRequestBudget, LLMClient
from src.models.memrec_agent import MemRecAgent
from src.cluster_runtime import require_allocation, require_project_root


def append_row(run: Path, name: str, row: dict) -> None:
    with (run / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class ObservedClient(LLMClient):
    """Observe the exact physical kwargs without changing baseline prompts."""

    def __init__(self, run: Path, config: dict, contract_sha: str, port: int):
        super().__init__(api_endpoint=f"http://127.0.0.1:{port}/v1", api_key="local-placeholder",
                         model=config["model_id"], provider_name="openai", sdk_max_retries=0)
        self.client.timeout = config["request_timeout_seconds"]
        self.fail_fast = self.strict_schema_validation = True
        self.request_budget = DurableRequestBudget(config["physical_request_cap"],
            str(run / "request-budget.sqlite"), contract_sha)
        self.last_request = None
        self.episode_id = None
        self.run, self.id_control = run, "constrained_decoding_delta" in config
        original = self.client.chat.completions.create

        def observed(**kwargs):
            self.last_request = deepcopy(kwargs)
            request_sha = object_sha256(kwargs)
            base = {"episode_id": self.episode_id, "request_sha256": request_sha,
                    "physical_attempt": self.request_budget.used}
            append_row(run, "physical-requests", {**base, "event": "request", "kwargs": kwargs})
            response = original(**kwargs)
            choice = response.choices[0]
            append_row(run, "physical-requests", {**base, "event": "response",
                "content": choice.message.content, "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else None})
            if choice.finish_reason != "stop" or not isinstance(choice.message.content, str):
                raise ValueError("Truncated/empty model response; no repair permitted")
            return response

        self.client.chat.completions.create = observed

    def set_episode(self, episode_id: str) -> None:
        self.episode_id = episode_id

    def generate_json(self, messages, properties, **kwargs):
        if not self.id_control:
            return super().generate_json(messages, properties, **kwargs)
        from src.cmirank.constrained_decoding import constrain_id_properties, validate_constrained_output
        constrained, audit = constrain_id_properties(messages, properties)
        append_row(self.run, "decoding-contracts", {"episode_id": self.episode_id,
            "messages_sha256": object_sha256(messages), "original_properties_sha256": object_sha256(properties),
            "constrained_properties_sha256": object_sha256(constrained), **audit})
        result = super().generate_json(messages, constrained, **kwargs)
        validate_constrained_output(result, {"type": "object", "properties": constrained,
            "required": list(constrained), "additionalProperties": False})
        return result


def verify_audit(audit_dir: Path, config: dict, inputs: dict, recipes: dict) -> None:
    if (audit_dir.name != config["audit_run_id"]
            or file_sha256(audit_dir / "full-hnv/report.json") != config["audit_report_sha256"]
            or file_sha256(audit_dir / "full-hnv/features.jsonl") != config["audit_features_sha256"]
            or (audit_dir / "source-commit.txt").read_text().strip() != config["candidate_source_commit"]):
        raise ValueError("Shortcut audit artifacts differ from reviewed completed v2 run")
    cleanup = json.loads((audit_dir / "cleanup.json").read_text())
    report = json.loads((audit_dir / "full-hnv/report.json").read_text())
    expected = {"full_candidates_sha256": config["full_candidates_sha256"],
                "full_report_sha256": config["full_report_sha256"], "recipe_version": 2,
                "snapshot_sha256": inputs["snapshot_sha256"],
                "split_sha256": inputs["manifest_sha256"],
                "episode_contract_sha256": recipes["episode_contract_sha256"]}
    if (cleanup != {"process_exit_code": 0, "device": "cpu", "gpu_requested": False, "child_exited": True}
            or report["status"] != "CPU_SHORTCUT_AUDIT_COMPLETE_REVIEW_REQUIRED"
            or report["gate_decision"] != "NO_FLAG_IN_FIXED_PROBES_REAL_MEMORY_AND_PPO_GATES_REMAIN"
            or report["diagnostics"]["risk_flags"]
            or report["diagnostics"]["fit_users"] != 1497
            or report["diagnostics"]["validation_users"] != 300
            or any(report["provenance"].get(k) != v for k, v in expected.items())):
        raise ValueError("V2 audit does not permit real-memory smoke")


def capture_gpu(run: Path, label: str) -> tuple[list, dict]:
    inventory = subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                     "--format=csv,noheader,nounits"], text=True)
    (run / f"gpus-{label}.csv").write_text(inventory)
    (run / f"apps-{label}.csv").write_text(apps)
    return parse_gpu_snapshot(inventory), compute_gpu_processes(apps)


def stop_owned_server(server, model_path: Path) -> None:
    """Only the process group created by this Popen; never an allocation kill."""
    if server is None:
        return
    if server.poll() is not None:
        return
    command = subprocess.check_output(["ps", "-p", str(server.pid), "-o", "args="], text=True)
    if str(model_path) not in command or os.getpgid(server.pid) != server.pid:
        raise RuntimeError("Server ownership/command changed; refusing an unknown target")
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--contract-version", choices=(1, 2, 3, 4, 5, 6, 7), type=int, default=4)
    parser.add_argument("--owner-confirmed-generator-step")
    args = parser.parse_args()
    config, contract_sha = load_memory_contract(ROOT, args.contract_version)
    run = args.run_dir.resolve()
    private_root = require_project_root(Path(os.environ["MEMREC_ROOT"]))
    if (run != private_root / "runs" / config["run_id"] or run.exists()
            or config["users"] != 20 or config["training_ready"] or config["output_repair"]
            or config["tensor_parallel_size"] != 1 or config["physical_request_cap"] != 110
            or config["nominal_physical_requests"] != 100):
        raise ValueError("Not a fresh locked 20-user smoke")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (commit != os.environ["MEMREC_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)):
        raise ValueError("Exact clean deployed source required")
    job = require_allocation()
    if args.owner_confirmed_generator_step and args.contract_version not in (4, 5, 7):
        raise ValueError("Owner-confirmed handoff is limited to v4/v5 or the approved v7 control")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU preparation never selects CUDA.
    run.mkdir()
    server, card, client, server_log = None, None, None, None
    exit_code = 1
    model_path = private_root / "models/Qwen3-30B-A3B-Instruct-2507-FP8-hnv"
    started = time.monotonic()
    # TERM from the outer hard timeout follows the same owned-server cleanup.
    def interrupted(signum, frame):
        raise InterruptedError(f"Smoke interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        inputs = load_locked_policy_inputs(ROOT)
        recipes = load_candidate_samplers(ROOT, args.index_dir, inputs["snapshot"], version=2)
        if args.candidate_run_dir.name != config["candidate_run_id"]:
            raise ValueError("Wrong candidate run")
        rows = verify_candidate_run(args.candidate_run_dir, config, inputs, recipes)
        verify_audit(args.audit_dir, config, inputs, recipes)
        if args.contract_version == 7:
            review_path = private_root / "runs/cmirank-real-memory-review-v6-20261007-hnv/review.json"
            if file_sha256(review_path) != config["constrained_decoding_delta"]["offline_v6_review_sha256"]:
                raise ValueError("V7 requires the independent completed v6 review")
            cpu_dir = private_root / "runs/cmirank-constrained-decoding-cpu-v7-20261007-hnv"
            cpu_report = json.loads((cpu_dir / "report.json").read_text())
            if (cpu_report["status"] != "CPU_INPUT_ID_GRAMMAR_SMOKE_PASS_GPU_SMOKE_AND_REVIEW_STILL_REQUIRED"
                    or cpu_report["source_commit"] != commit or cpu_report["config_sha256"] != contract_sha
                    or cpu_report["decoder_code_sha256"] != file_sha256(ROOT / "src/cmirank/constrained_decoding.py")
                    or cpu_report["schemas_compiled"] != 36 or not cpu_report["exact_structured_input_domain_parity"]
                    or cpu_report["new_llm_requests"] != 0 or cpu_report["gpu_requested"]
                    or file_sha256(cpu_dir / "compiled-input-domains.jsonl") != cpu_report["schema_domains_sha256"]):
                raise ValueError("V7 requires exact-source completed CPU input-domain/grammar smoke")
        user_ids = smoke_user_ids(sorted(inputs["warmups"]), recipes["episode_config"])
        metadata_path = ROOT / "data/processed/instructrec-books/instructrec-books.meta"
        if file_sha256(metadata_path) != recipes["config"]["metadata_sha256"]:
            raise ValueError("Static metadata changed")
        # Invoke only the unchanged static-metadata loader. RecDataset.__init__
        # would parse original suffix outcomes; it is deliberately never called.
        holder = SimpleNamespace(data_path=metadata_path.with_suffix(".inter"), item_metadata=None)
        RecDataset.load_item_metadata(holder)
        snapshot = replace(inputs["snapshot"], item_metadata=holder.item_metadata)
        agent = MemRecAgent(snapshot, None, temperature=config["temperature"],
                            max_tokens=config["max_tokens"], **config["agent"])
        marker = json.loads((model_path / "download-complete-hnv.json").read_text())
        if (marker.get("revision") != config["model_revision"] or marker.get("model") != config["model_id"]
                or not marker.get("weight_shards") or marker.get("weight_bytes", 0) < 25_000_000_000
                or len(list(model_path.glob("*.safetensors"))) != marker["weight_shards"]
                or any(file_sha256(model_path / name) != digest
                       for name, digest in marker["small_file_sha256"].items())):
            raise ValueError("Wrong completed checkpoint revision")
        if importlib.metadata.version("vllm") != "0.10.2":
            raise ValueError("Baseline vLLM environment changed")
        manifest = {"config": config, "config_sha256": contract_sha, "source_commit": commit,
                    "graph_snapshot_sha256": inputs["snapshot_sha256"],
                    "policy_split_manifest_sha256": inputs["manifest_sha256"],
                    "metadata_sha256": file_sha256(metadata_path), "query_user_ids": user_ids,
                    "original_instruction_accessed": False, "original_test_candidates_accessed": False,
                    "original_suffix_item_ids_parsed_or_used": False,
                    "checkpoint_marker_sha256": file_sha256(model_path / "download-complete-hnv.json"),
                    "checkpoint_config_sha256": file_sha256(model_path / "config.json"),
                    "checkpoint_tokenizer_sha256": file_sha256(model_path / "tokenizer_config.json"),
                    "compiler_cache_paths": {k: os.environ.get(k) for k in
                        ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH")},
                    "python_version": sys.version,
                    "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "vllm", "openai")}}
        if args.contract_version == 7:
            manifest["cpu_decoding_smoke_report_sha256"] = file_sha256(cpu_dir / "report.json")
            manifest["decoder_code_sha256"] = file_sha256(ROOT / "src/cmirank/constrained_decoding.py")
        (run / "manifest.json").write_text(artifact_json_dumps(manifest))
        if args.owner_confirmed_generator_step:
            # All data/checkpoint/metadata preparation is complete: only now
            # stop the exact current generator confirmed by the researcher.
            from src.cmirank.reserved_handoff import handoff
            manifest["gpu1_handoff"] = handoff(run, args.owner_confirmed_generator_step)
            (run / "manifest.json").write_text(artifact_json_dumps(manifest))
        cards, before_apps = capture_gpu(run, "before")
        card = select_reserved_gpu1(cards, set(before_apps))
        # Recheck the selected card immediately before CUDA/model initialization.
        cards_now, apps_now = capture_gpu(run, "load-time")
        now = next(c for c in cards_now if c.uuid == card.uuid)
        if select_reserved_gpu1([now], set(apps_now)).uuid != card.uuid:
            raise ValueError("Selected GPU no longer idle")
        card = now
        if args.contract_version >= 2:
            pci = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id",
                                             "--format=csv,noheader,nounits"], text=True)
            (run / "gpu-pci-load-time.csv").write_text(pci)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = verified_numeric_cuda_binding(cards_now, card, pci)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = card.uuid
        cuda_identity = subprocess.check_output([sys.executable, "-c",
            "import json, torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; "
            "print(json.dumps({'uuid': str(torch.cuda.get_device_properties(0).uuid), 'devices': 1}))"], text=True)
        identity = json.loads(cuda_identity)
        (run / "cuda-identity.json").write_text(artifact_json_dumps({
            "observed_cuda": identity, "selected_nvml_uuid": card.uuid,
            "numeric_visible_device": os.environ["CUDA_VISIBLE_DEVICES"]}))
        matches = (canonical_gpu_uuid(identity["uuid"]) == canonical_gpu_uuid(card.uuid)
                   if args.contract_version >= 3 else identity["uuid"].lower() == card.uuid.lower())
        if not matches:
            raise ValueError("Numeric CUDA binding does not point to the selected physical UUID")
        # The visibility-check child exits immediately; the CPU coordinator never
        # initializes a CUDA context or holds a second set of weights.
        manifest["physical_gpu_index"] = card.index
        manifest["gpu_uuid"] = card.uuid
        manifest["slurm_job_id"] = job
        manifest["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
        manifest["cuda_device_order"] = os.environ.get("CUDA_DEVICE_ORDER")
        manifest["cuda_identity_check"] = identity
        manifest["cpu_preparation_seconds"] = time.monotonic() - started
        (run / "manifest.json").write_text(artifact_json_dumps(manifest))
        port = 18105
        os.environ["MEMREC_LLM_CACHE_DB"] = str(run / "responses.sqlite")
        os.environ["MEMREC_LLM_CACHE_NAMESPACE"] = f"{config['model_revision']}:{contract_sha}"
        os.environ["MEMREC_LLM_CACHE_READ"] = "0"
        server_log = (run / "vllm-server.log").open("w")
        server_args = [str(private_root / "envs/llm-hnv/bin/vllm"), "serve", str(model_path),
            "--served-model-name", config["model_id"], "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", "1", "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--max-model-len", str(config["max_model_len"]), "--max-num-seqs", "1", "--seed", "42",
            "--dtype", "auto", "--generation-config", "vllm", "--disable-log-requests"]
        # Binding test prevents accidentally calling another project's service.
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
        server = subprocess.Popen(server_args, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        (run / "server.pid").write_text(str(server.pid) + "\n")
        ready_deadline = time.monotonic() + config["server_startup_timeout_seconds"]
        while True:
            if server.poll() is not None or time.monotonic() >= ready_deadline:
                raise RuntimeError("Owned model server failed readiness or timed out")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(5)  # Bounded service readiness only, not result polling.
        client = ObservedClient(run, config, object_sha256(manifest), port)
        agent.llm_client = agent.manager.llm = agent.reranker.llm = client
        peak_used_mib = 0
        def emit(name, row):
            nonlocal peak_used_mib
            if name == "progress":
                memory = int(subprocess.check_output(["nvidia-smi", "-i", card.uuid,
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
                peak_used_mib = max(peak_used_mib, memory)
                row = {**row, "gpu_used_mib": memory}
            append_row(run, name, row)
            if name == "progress":
                print(json.dumps({**row, "elapsed_seconds": time.monotonic() - started}), flush=True)
        result = run_memory_smoke(agent, inputs=inputs, rows=rows, user_ids=user_ids, emit=emit,
                                  candidate_context_warning=args.contract_version >= 4)
        stats = client.get_token_stats()
        if (stats["total_requests"] != 100 or stats["total_cache_hits"] != 0
                or stats["total_physical_requests"] != client.request_budget.used
                or not 100 <= client.request_budget.used <= 110):
            raise ValueError("Physical request/token/cache accounting differs")
        result.update({"status": "REAL_MEMORY_SMOKE_PASS_CLEANUP_AND_SEMANTIC_REVIEW_REQUIRED",
                       "source_commit": commit, "config_sha256": contract_sha,
                       "elapsed_seconds": time.monotonic() - started, "token_stats": stats,
                       "peak_observed_gpu_used_mib": peak_used_mib,
                       "gpu_telemetry_scope": "physical_memory_sampled_after_each_user_not_allocator_peak",
                       "artifact_sha256": {p.name: file_sha256(p) for p in run.glob("*.jsonl")}})
        if result["candidate_context_citation_occurrences"]:
            result["status"] = "REAL_MEMORY_SMOKE_PASS_WITH_ROLE_WARNINGS_CLEANUP_AND_SEMANTIC_REVIEW_REQUIRED"
        if args.contract_version == 7:
            decoding_rows = [json.loads(line) for line in (run / "decoding-contracts.jsonl").read_text().splitlines()]
            if len(decoding_rows) != 100:
                raise ValueError("Not one input-only decoding contract per logical request")
            result.update({"status": "SECONDARY_CONSTRAINED_MEMORY_SMOKE_PASS_CLEANUP_AND_SEMANTIC_REVIEW_REQUIRED",
                "comparison_role": "secondary_control_not_primary_replacement", "primary_provider_replaced": False,
                "controlled_decoding_logical_requests": len(decoding_rows),
                "candidate_role_warnings_present": bool(result["candidate_context_citation_occurrences"])})
        (run / "report.json").write_text(artifact_json_dumps(result))
        exit_code = 0
    except BaseException as error:
        (run / "failure.json").write_text(artifact_json_dumps({"error_type": type(error).__name__,
                                            "elapsed_seconds": time.monotonic() - started}))
        raise
    finally:
        # Unload the expensive model BEFORE artifact/client housekeeping.
        stop_owned_server(server, model_path)
        if client is not None:
            client.client.close()
            client.request_budget.connection.close()
            if client.response_cache:
                client.response_cache.connection.close()
        if server_log:
            server_log.close()
        # Coordinator never loaded weights, but drop its visibility-check context
        # by checking release after this process exits (launcher below).
        (run / "child-cleanup.json").write_text(artifact_json_dumps({
            "process_exit_code": exit_code, "owned_server_exited": server is None or server.poll() is not None,
            "gpu_uuid": card.uuid if card else None, "training_ready": False}))


if __name__ == "__main__":
    main()
