from copy import deepcopy
import os
from pathlib import Path

import pytest

from src.cmirank.host_admission import clean_file_headroom, clean_file_memory_budget
from src.cmirank.lora_rollout import load_lora_rollout_config

G = 2**30
ROOT = Path(__file__).resolve().parents[1]


def counters():
    return {"maximum_bytes": 64*G, "current_bytes": 64*G-16*2**20}, {
        "anon": 4*G, "file": 58*G, "inactive_file": 8*G, "active_file": 50*G,
        "file_mapped": G//2, "file_dirty": 0, "file_writeback": 0,
        "shmem": G//16, "unevictable": 0}


def test_clean_lru_accounting_keeps_cap_and_margin_and_legacy_false_rejection():
    memory, stats = counters()
    r = clean_file_headroom(memory, stats, 1000*G, 20*G)
    assert not r["inactive_only_assessment"]["within_estimated_budget"]
    assert r["within_estimated_budget"] and r["cache_credit_bytes"] == 20*G
    assert r["required_margin_bytes"] == 20*G and not r["forced_reclaim"]
    assert not r["full_model_loading_fit_proven"]
    assert r["estimated_headroom_bytes"] <= memory["maximum_bytes"]


@pytest.mark.parametrize("field", ["file_dirty", "file_writeback", "file_mapped", "shmem", "unevictable"])
def test_lru_does_not_credit_dirty_mapped_shared_or_unevictable_memory(field):
    memory, stats = counters()
    stats[field] = 58*G
    r = clean_file_headroom(memory, stats, 1000*G, 20*G)
    assert r["cache_credit_bytes"] == 0 and not r["within_estimated_budget"]


def test_no_anon_or_slab_credit_and_node_pressure_is_a_hard_bound():
    memory, stats = counters()
    stats.update(file=0, inactive_file=0, active_file=0, anon=58*G, slab_reclaimable=58*G)
    assert not clean_file_headroom(memory, stats, 1000*G, 20*G)["within_estimated_budget"]
    _, stats = counters()
    assert not clean_file_headroom(memory, stats, 10*G, 20*G)["within_estimated_budget"]


def test_all_file_credit_is_bounded_by_both_lru_lists_and_type_counter():
    memory, stats = counters()
    stats.update(file=10*G, active_file=100*G, inactive_file=100*G)
    r = clean_file_headroom(memory, stats, 1000*G, 20*G)
    assert r["cache_credit_bytes"] < 10*G and not r["within_estimated_budget"]
    stats.update(file=58*G, active_file=0, inactive_file=0)
    assert not clean_file_headroom(memory, stats, 1000*G, 20*G)["within_estimated_budget"]


def test_invalid_lru_or_visible_cuda_fails_closed(monkeypatch):
    memory, stats = counters()
    for value in (-1, True, 1.0):
        bad = deepcopy(stats)
        bad["active_file"] = value
        with pytest.raises(ValueError):
            clean_file_headroom(memory, bad, 1000*G, 20*G)
    for mask in ("0", "1", "0,1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        with pytest.raises(ValueError):
            clean_file_memory_budget("21820", 20*G)


def test_opt_in_profile_does_not_change_any_scientific_precision_or_resource_limit():
    old, _ = load_lora_rollout_config(ROOT, 1)
    c, _ = load_lora_rollout_config(ROOT)
    allowed = {"schema_version", "inherits", "prepare_run_id", "run_id", "review_run_id", "reason"}
    assert all(c[k] == v for k, v in old.items() if k not in allowed)
    assert c["host_load_reserve_bytes"] == 20*G and c["gpu_index"] == 1 and c["physical_request_cap"] == 60
    source = (ROOT/"src/cmirank/host_admission.py").read_text()
    for bad in (".write_text(", "os.kill", "subprocess.run", "import torch", "drop_caches", "memory.reclaim"):
        assert bad not in source
