"""Read-only CPU-only cgroup headroom estimate, never a memory reservation.

memory.current includes filesystem cache. Credit only a bounded conservative
estimate of clean inactive file pages; never trigger reclamation, change limits
or reuse this CPU/meta check as proof of full GPU-model loading feasibility.
"""

import os
from pathlib import Path

from .optimizer_smoke import allocation_memory


def cpu_headroom(memory, stats, node_available_bytes, margin_bytes):
    keys = ("anon", "file", "inactive_file", "file_mapped", "file_dirty",
            "file_writeback", "shmem", "unevictable")
    values = [memory["maximum_bytes"], memory["current_bytes"], node_available_bytes, margin_bytes]
    values.extend(stats[k] for k in keys)
    if (any(type(x) is not int or x < 0 for x in values)
            or min(memory["maximum_bytes"], margin_bytes) <= 0
            or memory["current_bytes"] > memory["maximum_bytes"]):
        raise ValueError("Valid current cgroup/node counters and positive unchanged margin required")
    raw = memory["maximum_bytes"]-memory["current_bytes"]
    # Over-subtract all mapped/dirty/writeback/shared/unevictable pages even
    # though some are not on the inactive list. Credit neither active cache
    # nor reclaimable slab. Account at most one margin, never all file cache.
    clean = max(0, min(stats["inactive_file"], stats["file"])
        -sum(stats[k] for k in ("file_mapped", "file_dirty", "file_writeback", "shmem", "unevictable")))
    credit = min(clean, margin_bytes)
    estimate = min(raw+credit, node_available_bytes, memory["maximum_bytes"])
    return {"method": "cpu_only_clean_inactive_file_estimate_not_reservation",
        "raw_headroom_bytes": raw, "clean_inactive_file_estimate_bytes": clean,
        "cache_credit_bytes": credit, "node_available_bytes": node_available_bytes,
        "estimated_headroom_bytes": estimate, "required_margin_bytes": margin_bytes,
        "within_estimated_budget": estimate >= margin_bytes,
        "full_model_loading_fit_proven": False, "forced_reclaim": False}


def cpu_memory_budget(job, margin_bytes):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Cache-aware headroom is limited to a CUDA-hidden CPU audit")
    memory = allocation_memory(job)
    line = next(x for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    p = Path("/sys/fs/cgroup")/line.split("::", 1)[1].lstrip("/")
    while p.name != f"job_{job}":
        if p == p.parent or p == Path("/sys/fs/cgroup"):
            raise ValueError("Own-allocation cgroup unavailable")
        p = p.parent
    stats = {k: int(v) for k, v in (x.split() for x in (p/"memory.stat").read_text().splitlines())}
    events = {k: int(v) for k, v in (x.split() for x in (p/"memory.events").read_text().splitlines())}
    available = next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines()
                     if x.startswith("MemAvailable:"))
    return {"allocation_memory": memory, "stats_bytes": stats, "memory_events": events,
        "assessment": cpu_headroom(memory, stats, available, margin_bytes)}
