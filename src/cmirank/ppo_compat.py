"""Full-role compatibility contracts; no models/GPU/data loader at import time."""

import json
from pathlib import Path

from .format_sft import validate_encoded_row
from .provenance import file_sha256


def load_compat_config(root: Path) -> tuple[dict, Path]:
    path = root / "configs/cmirank/ppo_gpu_compat_v1.json"
    config = json.loads(path.read_text())
    if (config["samples"] != 20 or config["gpu_index"] != 0
            or config["resource_policy"] != "gpu0_only_gpu1_all_workloads_read_only"
            or config["dtype"] != "bfloat16" or config["attn_implementation"] != "flash_attention_2"
            or config["gamma"] != 1 or config["gae_test_lambda"] != 1
            or config["optimizer_updates"] != 0 or config["generation_calls"] != 0
            or config["books_outcomes_accessed"] or config["ranking_metrics_computed"] or config["training_ready"]):
        raise ValueError("Not the frozen twenty-sample no-update full-role compatibility gate")
    return config, path


def compat_code_hashes(root: Path) -> dict:
    return {name: file_sha256(root / name) for name in (
        "src/cmirank/ppo_compat.py", "scripts/cmirank/20_prepare_ppo_compat_cpu.py",
        "scripts/cmirank/21_smoke_ppo_full_roles_gpu.py", "src/cmirank/format_sft.py",
        "src/cmirank/reserved_handoff.py", "src/cmirank/gpu_resources.py",
        "scripts/cmirank/run_real_policy_smoke.sh", "scripts/cmirank/run_real_policy_with_keeper.sh")}


def validate_compat_rows(rows: list[dict], config: dict) -> None:
    if len(rows) != 20 or len({r["user_id"] for r in rows}) != 20:
        raise ValueError("Need twenty disjoint synthetic holdout requests")
    for row in rows:
        if row["user_id"] >= 0 or row["snapshot_id"] != "synthetic-format-holdout-v1":
            raise ValueError("Books/seen training data forbidden in compatibility gate")
        validate_encoded_row(row, config)


def verify_preparation(directory: Path, *, commit: str, config_sha: str, code_sha: dict, config: dict) -> tuple[list, dict]:
    report = json.loads((directory / "report.json").read_text())
    if (report["status"] != "FULL_ROLE_COMPAT_CPU_PREPARATION_PASS_GPU_GATES_REMAIN"
            or report["source_commit"] != commit or report["config_sha256"] != config_sha
            or report["code_sha256"] != code_sha or report["gpu_requested"] or report["model_weights_loaded"]
            or report["training_ready"]):
        raise ValueError("Source-bound CPU preparation does not authorize this GPU source/config")
    for name, sha in report["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(directory / name) != sha:
            raise ValueError("Prepared compatibility artifacts changed")
    rows = [json.loads(line) for line in (directory / "tokens.jsonl").read_text().splitlines()]
    validate_compat_rows(rows, config)
    return rows, report
