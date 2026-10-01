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


def compute_gpu_uuids(raw: str) -> set[str]:
    result = set()
    for row in csv.reader(StringIO(raw)):
        if not row:
            continue
        if len(row) != 2 or not row[0].strip().startswith("GPU-"):
            raise ValueError("Cannot establish compute-process ownership by GPU")
        int(row[1].strip())
        result.add(row[0].strip())
    return result


def select_idle_h100(cards: list[GPUCard], occupied: set[str]) -> GPUCard:
    """Choose a genuinely empty H100, descending physical index as requested."""
    for card in sorted(cards, key=lambda value: value.index, reverse=True):
        if (0 <= card.index <= 3 and card.uuid.startswith("GPU-")
                and "H100" in card.name and card.total_mib >= 80000
                and 0 <= card.utilization < 20 and 0 <= card.used_mib < 512
                and card.uuid not in occupied):
            return card
    raise ValueError("No empty H100: require <20% utilization, <512 MiB and no process")
