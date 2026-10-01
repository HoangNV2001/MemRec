"""Runbook GPU preflight; no CUDA import and no process mutation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO


@dataclass(frozen=True)
class GPUCard:
    index: int
    uuid: str
    name: str
    utilization: int
    used_mib: int
    total_mib: int


def parse_gpu_snapshot(raw: str) -> list[GPUCard]:
    cards = []
    for row in csv.reader(StringIO(raw)):
        if not row:
            continue
        if len(row) != 6:
            raise ValueError("Unexpected GPU snapshot columns")
        index, uuid, name, util, used, total = (value.strip() for value in row)
        cards.append(GPUCard(int(index), uuid, name, int(util), int(used), int(total)))
    if not 1 <= len(cards) <= 4 or len({card.index for card in cards}) != len(cards):
        raise ValueError("Unexpected or duplicate visible GPU inventory")
    return cards


def compute_gpu_processes(raw: str) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for row in csv.reader(StringIO(raw)):
        if not row:
            continue
        if len(row) != 2 or not row[0].strip().startswith("GPU-"):
            raise ValueError("Cannot establish compute-process ownership by GPU")
        result.setdefault(row[0].strip(), set()).add(int(row[1].strip()))
    return result


def compute_gpu_uuids(raw: str) -> set[str]:
    return set(compute_gpu_processes(raw))


def select_idle_h100(cards: list[GPUCard], occupied: set[str]) -> GPUCard:
    """Choose a genuinely empty H100, descending physical index as requested."""
    for card in sorted(cards, key=lambda value: value.index, reverse=True):
        if (0 <= card.index <= 3 and card.uuid.startswith("GPU-")
                and "H100" in card.name and card.total_mib >= 80000
                and 0 <= card.utilization < 20 and 0 <= card.used_mib < 512
                and card.uuid not in occupied):
            return card
    raise ValueError("No empty H100: require <20% utilization, <512 MiB and no process")


def select_qwen_smoke_gpu(cards: list[GPUCard], occupied: set[str], config: dict) -> GPUCard:
    """One researcher-approved shared-card exception, never a default policy."""
    policy = config.get("gpu_policy", "idle_only")
    if policy == "idle_only":
        return select_idle_h100(cards, occupied)
    if (policy != "shared_gpu1_single_smoke_20261001"
            or config.get("run_id") != "cmirank-qwen35-g0-smoke-v3-sharedgpu1-20261001-hnv"
            or config.get("scope") != "synthetic_infrastructure_only"
            or config.get("examples") != 20
            or config.get("cuda_memory_fraction") != 0.60
            or config.get("timeout_minutes") != 10):
        raise ValueError("Shared-GPU exception is limited to the approved single smoke")
    for card in cards:
        if (card.index == 1 and card.uuid.startswith("GPU-") and "H100" in card.name
                and card.total_mib >= 80000 and card.utilization == 0
                and 0 <= card.used_mib <= 4096):
            return card
    raise ValueError("Shared smoke requires GPU 1 at 0% utilization and baseline <=4 GiB")


def gpu_release_verified(card: GPUCard, baseline: GPUCard, before: set[int],
                         after: set[int], *, shared: bool = False) -> bool:
    """After our child exits, check release without requiring others to exit."""
    if card.uuid != baseline.uuid:
        raise ValueError("Release snapshot differs from the selected GPU")
    if shared:
        return not (after - before) and card.used_mib <= baseline.used_mib + 128
    return card.used_mib < 512 and not after
