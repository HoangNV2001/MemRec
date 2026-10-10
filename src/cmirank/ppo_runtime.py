"""CPU-only dependency and reward-placement guards; not a PPO implementation."""

from __future__ import annotations

import re
import json
from pathlib import Path
from urllib.parse import urlparse


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_runtime_config(root: Path, version: int = 2) -> tuple[dict, Path]:
    if version not in (1, 2):
        raise ValueError("Unknown runtime profile")
    base = json.loads((root / "configs/cmirank/ppo_runtime_v1.json").read_text())
    path = root / f"configs/cmirank/ppo_runtime_v{version}.json"
    if version == 2:
        retry = json.loads(path.read_text())
        allowed = {"schema_version", "inherits", "run_id", "environment_name", "allowed_cuda_toolkit_version",
                   "additional_versions", "reason"}
        if (set(retry) != allowed or retry["inherits"] != "configs/cmirank/ppo_runtime_v1.json"
                or retry["additional_versions"] != {"cuda-toolkit": "12.9.1", "cuda-bindings": "12.9.9", "cuda-tile": "1.6.0"}
                or retry["allowed_cuda_toolkit_version"] != "12.9.1"):
            raise ValueError("Resource retry cannot modify model/algorithm/core stack")
        base = {**base, **{k: v for k, v in retry.items() if k != "additional_versions"},
                "versions": {**base["versions"], **retry["additional_versions"]}}
    return base, path


def verify_resolution(report: dict, config: dict) -> dict:
    """Reject incompatible CUDA families and unpinned primary backend components."""
    versions = {}
    for row in report["install"]:
        name = canonical_name(row["metadata"]["name"])
        version = row["metadata"]["version"]
        url = row["download_info"]["url"]
        if name in versions:
            raise ValueError("Duplicate dependency in resolver receipt")
        versions[name] = version
        if ("-cu13" in name or "+cu13" in version or "cu130" in url
                or (name == "cuda-toolkit" and version != config.get("allowed_cuda_toolkit_version"))
                or (name.startswith("cuda-") and version.startswith("13."))):
            raise ValueError("CUDA13/toolkit dependency is not authorized")
        if urlparse(url).scheme != "https":
            raise ValueError("Dependencies must originate from HTTPS sources")
        if name in config["wheels"]:
            expected = config["wheels"][name]
            observed = row["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")
            if url != expected["url"] or observed != expected["sha256"]:
                raise ValueError("Official CUDA12 wheel URL/hash differs")
        if name == "verl":
            vcs = row["download_info"].get("vcs_info", {})
            if (url != "https://github.com/verl-project/verl.git" or vcs.get("vcs") != "git"
                    or vcs.get("commit_id") != config["verl_commit"]):
                raise ValueError("VeRL source differs from approved runtime profile")
    if "verl" not in versions:
        raise ValueError("VeRL missing from resolver receipt")
    if any(versions.get(k) != v for k, v in config["versions"].items()):
        raise ValueError("Primary dependency version differs from runtime profile")
    return versions


def locked_requirements(report: dict) -> list[str]:
    """Install the exact resolved artifact URLs, including immutable VeRL source."""
    lines = []
    for row in report["install"]:
        name = canonical_name(row["metadata"]["name"])
        info = row["download_info"]
        if "vcs_info" in info:
            if name != "verl":
                raise ValueError("Unexpected VCS dependency")
            lines.append(f"verl @ git+{info['url']}@{info['vcs_info']['commit_id']}")
        else:
            sha = info.get("archive_info", {}).get("hashes", {}).get("sha256")
            if not sha or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("Dependency artifact missing SHA256")
            lines.append(f"{name} @ {info['url'].split('#')[0]}#sha256={sha}")
    return sorted(lines)


def response_reward_layout(lengths: list[int], rewards: list[float]) -> tuple[list[float], list[int]]:
    """Place each action reward at its EOS; concatenate *only* response tokens.

    Intervening next-state prompts are observations, not generated actions.
    Production integration must maintain this episode continuity for GAE.
    """
    if len(lengths) != 9 or len(rewards) != 9 or any(n < 1 for n in lengths):
        raise ValueError("Need all nine nonempty responses of a ten-candidate episode")
    token_rewards, final_indices = [], []
    for n, reward in zip(lengths, rewards):
        token_rewards.extend([0.0] * (n - 1) + [float(reward)])
        final_indices.append(len(token_rewards) - 1)
    return token_rewards, final_indices


def require_new_environment(private: Path, name: str) -> Path:
    if not re.fullmatch(r"cmirank-ppo-cu129-v[0-9]+-hnv", name):
        raise ValueError("Not an isolated PPO environment")
    env = private / "envs" / name
    if env.exists() or env.is_symlink():
        raise ValueError("Refusing to overwrite an existing/partial environment")
    return env


def load_kernel_config(root: Path, version: int = 2) -> tuple[dict, Path]:
    if version not in (1, 2):
        raise ValueError("Unknown kernel build profile")
    base = json.loads((root / "configs/cmirank/ppo_kernel_build_v1.json").read_text())
    path = root / f"configs/cmirank/ppo_kernel_build_v{version}.json"
    if version == 2:
        retry = json.loads(path.read_text())
        allowed = {"schema_version", "inherits", "run_id", "overlay_name", "timeout_minutes", "compile_timeout_seconds",
                   "resume_source_relative_path", "resume_failure_sha256", "resume_submodule_commits", "reason"}
        relative = Path(retry.get("resume_source_relative_path", ""))
        if (set(retry) != allowed or retry["inherits"] != "configs/cmirank/ppo_kernel_build_v1.json"
                or relative.is_absolute() or ".." in relative.parts
                or relative.parts[:3] != ("runs", base["run_id"], "tmp")
                or len(relative.parts) != 4 or not relative.name.startswith("pip-req-build-")
                or retry["timeout_minutes"] != 95 or retry["compile_timeout_seconds"] != 5400):
            raise ValueError("Kernel retry changes scope/source/build contract")
        base.update(retry)
    else:
        base["compile_timeout_seconds"] = 2100
    return base, path
