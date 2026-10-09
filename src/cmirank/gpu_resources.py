"""Runbook GPU preflight; no CUDA import and no process mutation."""

from __future__ import annotations

import csv
import re
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


def select_reserved_gpu1(cards: list[GPUCard], occupied: set[str]) -> GPUCard:
    """New permission: GPU 1 only, after a proven generator-step handoff.

    This function NEVER stops a step/job and never shares/falls back to GPU 0.
    A separate, verified handoff must make GPU 1 idle before a model launcher.
    """
    return select_reserved_gpu(cards, occupied, gpu_index=1)


def select_reserved_gpu(cards: list[GPUCard], occupied: set[str], *, gpu_index: int) -> GPUCard:
    """Explicitly authorized single card only; never fall back to another GPU."""
    if type(gpu_index) is not int or gpu_index not in (0, 1):
        raise ValueError("Reserved handoff supports only an explicitly authorized GPU0 or GPU1")
    return select_idle_h100([card for card in cards if card.index == gpu_index], occupied)


def generator_step_target(job: str, step: str, *, name: str, owner: str, state: str,
                          gpu_pids: set[int], step_pids: set[int]) -> str:
    """Validate ONLY a numeric omni-gen-1 step target, never a job/batch/step0.

    Caller must additionally verify PID UID/cgroups/command and capture the
    exact allocation/GPU snapshots. Returns a target, performs no cancellation.
    """
    if (not job.isdigit() or not step.isdigit() or int(step) == 0 or name != "omni-gen-1"
            or owner != "hoangnv242" or state != "RUNNING"
            or not gpu_pids or not gpu_pids <= step_pids):
        raise ValueError("Cannot prove exclusive ownership of the authorized GPU1 generator step")
    return f"{job}.{step}"


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


def verified_numeric_cuda_binding(cards: list[GPUCard], selected: GPUCard, pci_snapshot: str) -> str:
    """vLLM 0.10.2 requires integer CVD; prove NVML and PCI CUDA ordering agree.

    Return no fallback if the full inventory is incomplete/reordered. The model
    launcher must ALSO assert the UUID seen at logical CUDA device zero before
    loading weights. Never infer a device index from a previous run.
    """
    pci = {}
    for row in csv.reader(StringIO(pci_snapshot)):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("Malformed GPU PCI snapshot")
        index, uuid, bus = [value.strip() for value in row]
        match = re.fullmatch(r"([0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", bus)
        if not match or int(index) in pci:
            raise ValueError("Malformed/duplicate GPU PCI identity")
        pci[int(index)] = (uuid, tuple(int(part, 16) for part in match.groups()))
    if (set(pci) != {card.index for card in cards}
            or set(pci) != set(range(len(cards)))
            or any(pci[card.index][0] != card.uuid for card in cards)
            or len({bus for _, bus in pci.values()}) != len(cards)
            or sorted(pci, key=lambda index: pci[index][1]) != list(range(len(cards)))
            or selected not in cards):
        raise ValueError("NVML index and CUDA PCI ordering/UUID do not agree")
    return str(selected.index)


def canonical_gpu_uuid(value: str) -> str:
    """Same complete 128-bit identity: NVML has GPU-, Torch's CUuuid does not.

    Accept only the exact hex UUID grammar (optional GPU- namespace); never
    accept a prefix/subsequence, another GPU, a MIG identifier or extra text.
    """
    if not isinstance(value, str):
        raise ValueError("GPU UUID must be text")
    match = re.fullmatch(r"(?:gpu-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", value.lower())
    if match is None:
        raise ValueError("Malformed full GPU UUID")
    return match.group(1)
