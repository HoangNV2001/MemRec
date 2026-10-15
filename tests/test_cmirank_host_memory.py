from copy import deepcopy
import importlib
import os

import pytest

from src.cmirank.host_memory import cpu_headroom, cpu_memory_budget

G = 2**30


def counters():
    return {"maximum_bytes": 64*G, "current_bytes": 62*G}, {"anon": 4*G, "file": 56*G,
        "inactive_file": 30*G, "file_mapped": G//2, "file_dirty": 0, "file_writeback": 0,
        "shmem": G//16, "unevictable": 0}


def test_file_cache_is_not_confused_with_anonymous_state_or_a_reservation():
    memory, stats = counters()
    r = cpu_headroom(memory, stats, 1024*G, 20*G)
    assert r["raw_headroom_bytes"] == 2*G
    assert r["cache_credit_bytes"] == 20*G
    assert r["estimated_headroom_bytes"] == 22*G and r["within_estimated_budget"]
    assert not r["forced_reclaim"] and not r["full_model_loading_fit_proven"]


@pytest.mark.parametrize("field", ["file_mapped", "file_dirty", "file_writeback", "shmem", "unevictable"])
def test_no_credit_for_dirty_mapped_shared_or_unevictable_pages(field):
    memory, stats = counters()
    stats[field] = 30*G
    r = cpu_headroom(memory, stats, 1024*G, 20*G)
    assert r["cache_credit_bytes"] == 0 and not r["within_estimated_budget"]


def test_no_credit_for_active_cache_or_slab_and_respect_node_memory():
    memory, stats = counters()
    stats.update(inactive_file=0, active_file=56*G, slab_reclaimable=20*G)
    assert not cpu_headroom(memory, stats, 1024*G, 20*G)["within_estimated_budget"]
    _, stats = counters()
    r = cpu_headroom(memory, stats, 10*G, 20*G)
    assert r["estimated_headroom_bytes"] == 10*G and not r["within_estimated_budget"]


def test_bad_or_missing_counters_fail_closed():
    memory, stats = counters()
    with pytest.raises(KeyError):
        cpu_headroom(memory, {}, 1024*G, 20*G)
    for key in ("inactive_file", "file_dirty"):
        bad = deepcopy(stats)
        bad[key] = -1
        with pytest.raises(ValueError):
            cpu_headroom(memory, bad, 1024*G, 20*G)
    memory["current_bytes"] = 65*G
    with pytest.raises(ValueError):
        cpu_headroom(memory, stats, 1024*G, 20*G)


def test_cpu_accounting_cannot_be_used_as_a_visible_gpu_guard(monkeypatch):
    for mask in ("0", "1", "0,1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        with pytest.raises(ValueError):
            cpu_memory_budget("21820", 20*G)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValueError):
        cpu_memory_budget("21820", 20*G)


def test_import_is_lightweight_and_has_no_writes_reclaim_or_signalling():
    from pathlib import Path
    module = importlib.import_module("src.cmirank.host_memory")
    source = Path(module.__file__).read_text()
    for forbidden in ("import torch", ".write_text(", 'open("w")', "os.kill", "subprocess.run", "drop_caches", "memory.reclaim"):
        assert forbidden not in source
