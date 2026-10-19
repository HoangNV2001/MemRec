"""Read-only clean-file LRU accounting for the measured sequential LoRA layout.

Active/inactive describe LRU age, not whether file pages are pinned. The old
inactive-only profile remains unchanged. This opt-in estimate is bounded by
one unchanged margin; dirty/mapped/shared/unevictable pages get no credit.
Neither kernel reclamation nor allocation/cgroup changes are requested here.
"""

from .host_memory import cpu_headroom, cpu_memory_budget


def clean_file_headroom(memory, stats, node_available_bytes, margin_bytes):
    old = cpu_headroom(memory, stats, node_available_bytes, margin_bytes)
    active = stats["active_file"]
    if type(active) is not int or active < 0:
        raise ValueError("Valid active-file LRU counter required")
    excluded = sum(stats[k] for k in ("file_mapped", "file_dirty", "file_writeback", "shmem", "unevictable"))
    clean = max(0, min(stats["file"], active+stats["inactive_file"])-excluded)
    credit = min(clean, margin_bytes)
    available = min(old["raw_headroom_bytes"]+credit, node_available_bytes, memory["maximum_bytes"])
    return {"method": "clean_file_lru_estimate_measured_native_layout_required_not_reservation",
        "raw_headroom_bytes": old["raw_headroom_bytes"], "clean_file_estimate_bytes": clean,
        "cache_credit_bytes": credit, "estimated_headroom_bytes": available,
        "required_margin_bytes": margin_bytes, "within_estimated_budget": available >= margin_bytes,
        "inactive_only_assessment": old, "full_model_loading_fit_proven": False, "forced_reclaim": False}


def clean_file_memory_budget(job, margin_bytes):
    # The legacy reader validates CUDA-hidden context, ownership, the actual
    # cgroup cap/counters/events and node MemAvailable. Do not duplicate it.
    b = cpu_memory_budget(job, margin_bytes)
    b["assessment"] = clean_file_headroom(b["allocation_memory"], b["stats_bytes"],
        b["assessment"]["node_available_bytes"], margin_bytes)
    return b
