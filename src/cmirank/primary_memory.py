"""Primary MemRec memory on a predeclared train-only smoke cohort, no CUDA."""

import hashlib
import json
from pathlib import Path

from .provenance import file_sha256


def primary_user_ids(inputs, c):
    train, val = inputs["groups"]["policy_train"], inputs["groups"]["policy_val"]
    if (len(train) != 1497 or len(set(train)) != 1497 or len(val) != 300 or len(set(val)) != 300
            or set(train) & set(val) or c["cohort"] != "locked_policy_train_only"):
        raise ValueError("The original locked1497/300 split must remain unchanged")
    ordered = sorted(train, key=lambda uid: (hashlib.sha256(
        f"{c['selection_seed']}:{uid}".encode()).digest(), uid))
    return sorted(ordered[:20])


def primary_code_hashes(root):
    names = ("configs/cmirank/real_memory_smoke_v8.json", "src/cmirank/primary_memory.py",
        "scripts/cmirank/43_prepare_primary_memory_cpu.py", "scripts/cmirank/10_smoke_real_memory.py",
        "src/cmirank/memory_smoke.py", "src/cmirank/policy_inputs.py", "src/cmirank/shortcut_audit.py",
        "src/cmirank/candidate_artifacts.py", "src/models/memrec_agent.py", "src/memory/manager.py",
        "src/models/reranker_llm.py", "scripts/cmirank/run_real_policy_smoke.sh",
        "scripts/cmirank/run_real_policy_with_keeper.sh", "src/cmirank/t0_authority.py",
        "src/cmirank/reserved_handoff.py", "src/cmirank/gpu_resources.py")
    return {n: file_sha256(root/n) for n in names}


def reviewed_ppo_mechanics(private, c):
    run = private/"runs"/c["ppo_smoke_run_id"]
    if (file_sha256(run/"report.json") != c["ppo_smoke_report_sha256"]
            or file_sha256(run/"native-worker-report.json") != c["ppo_smoke_worker_sha256"]):
        raise ValueError("Reviewed both-role PPO smoke changed")
    r = json.loads((run/"report.json").read_text())
    worker = json.loads((run/"native-worker-report.json").read_text())
    cleanup = json.loads((run/"cleanup.json").read_text())
    handback = json.loads((run/"keeper-handback.json").read_text())
    if (r["status"] != "NATIVE_LORA_MULTITURN_VALID_AND_INVALID_PPO_SMOKE_PASS_REVIEW_REQUIRED"
            or r["optimizer_updates"] != 8 or r["samples"] != 20 or r["training_ready"]
            or worker["roles"]["actor"]["optimizer_updates"] != 4
            or worker["roles"]["critic"]["optimizer_updates"] != 4
            or not worker["native_episode_continuous_gae"] or worker["books_outcomes_accessed"]
            or cleanup["process_exit_code"] or not cleanup["gpu_released"]
            or handback["status"] != "KEEPER_GPU1_HANDBACK_VERIFIED"
            or handback["reserved_job_cancelled"] or handback["gpu0_signalled"]):
        raise ValueError("Completed/released mechanics smoke required, not a quality readiness claim")
    # The historical receipt stays attached to its original allocation.
    # A new current allocation needs its own resource grant/preflight.
    return r


def verify_primary_preparation(private, root, c, contract_sha, inputs, users, commit, job):
    directory = private/"runs"/c["prepare_run_id"]
    p = json.loads((directory/"report.json").read_text())
    if (p["status"] != "PRIMARY_TRAIN20_CPU_WIRING_PASS_REAL_MEMORY_GATE_REMAINS"
            or p["source_commit"] != commit or p["config_sha256"] != contract_sha
            or p["code_sha256"] != primary_code_hashes(root) or p["slurm_job_id"] != job
            or p["query_user_ids"] != users or p["graph_snapshot_sha256"] != inputs["snapshot_sha256"]
            or p["split_sha256"] != inputs["manifest_sha256"] or p["real_llm_requests"]
            or p["gpu_requested"] or p["training_ready"] or p["fake_logical_requests"] != 100):
        raise ValueError("Fresh current-allocation exact-source CPU20 preparation required")
    for name, sha in p["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(directory/name) != sha:
            raise ValueError("Primary CPU cohort/probes changed")
    reviewed_ppo_mechanics(private, c)
    return p
