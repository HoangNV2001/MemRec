"""CPU-only checks of the shared-cluster selection and infrastructure fixtures."""

from dataclasses import replace

import pytest

from src.cmirank.gpu_resources import (
    compute_gpu_uuids, compute_gpu_processes, parse_gpu_snapshot, select_idle_h100,
    select_qwen_smoke_gpu, gpu_release_verified,
)
from src.cmirank.labels import make_labels
from src.cmirank.prompts import render_step_prompt
from src.cmirank.smoke_fixtures import synthetic_rank_requests


def test_selector_requires_empty_device_and_prefers_largest_index():
    cards = parse_gpu_snapshot(
        "0, GPU-zero, NVIDIA H100 80GB HBM3, 0, 1, 81559\n"
        "1, GPU-one, NVIDIA H100 80GB HBM3, 0, 1, 81559\n"
        "2, GPU-two, NVIDIA H100 80GB HBM3, 0, 1, 81559\n"
        "3, GPU-three, NVIDIA H100 80GB HBM3, 0, 2000, 81559\n"
    )
    assert select_idle_h100(cards, set()).index == 2
    assert select_idle_h100(cards, {"GPU-two"}).index == 1
    assert select_idle_h100(cards, {"GPU-one", "GPU-two"}).index == 0
    with pytest.raises(ValueError, match="No empty"):
        select_idle_h100(cards, {"GPU-zero", "GPU-one", "GPU-two"})


def test_selector_fails_closed_on_unknown_inventory_or_processes():
    with pytest.raises(ValueError):
        parse_gpu_snapshot("0, GPU-a, NVIDIA H100, N/A, 1, 81559\n")
    with pytest.raises(ValueError, match="duplicate"):
        parse_gpu_snapshot("0, GPU-a, NVIDIA H100, 0, 1, 81559\n" * 2)
    with pytest.raises(ValueError):
        compute_gpu_uuids("Unknown, N/A\n")
    assert compute_gpu_uuids("GPU-a, 1234\n") == {"GPU-a"}


def test_synthetic_fixtures_are_reproducible_and_never_use_outcome_fields():
    first = synthetic_rank_requests()
    second = synthetic_rank_requests()
    assert [request.sha256() for request in first] == [request.sha256() for request in second]
    assert len({request.sha256() for request in first}) == 20
    for request in first:
        assert "target_item_id" not in request.to_dict()
        assert len(request.candidates) == 10
        assert "<answer>" in render_step_prompt(request, make_labels(range(10)))


def shared_smoke_config():
    return {"gpu_policy": "shared_gpu1_single_smoke_20261001",
            "run_id": "cmirank-qwen35-g0-smoke-v3-sharedgpu1-20261001-hnv",
            "scope": "synthetic_infrastructure_only", "examples": 20,
            "cuda_memory_fraction": 0.60, "timeout_minutes": 10}


def test_shared_permission_does_not_relax_default_idle_policy():
    cards = parse_gpu_snapshot("1, GPU-one, NVIDIA H100, 0, 2488, 81559\n")
    assert select_qwen_smoke_gpu(cards, {"GPU-one"}, shared_smoke_config()).index == 1
    with pytest.raises(ValueError, match="No empty"):
        select_qwen_smoke_gpu(cards, {"GPU-one"}, {})


@pytest.mark.parametrize("field,value", [
    ("run_id", "future-full-run-hnv"), ("scope", "books_full"),
    ("examples", 200), ("cuda_memory_fraction", 0.90), ("timeout_minutes", 45),
])
def test_oneoff_shared_permission_cannot_promote_or_expand(field, value):
    config = shared_smoke_config()
    config[field] = value
    with pytest.raises(ValueError, match="single smoke"):
        select_qwen_smoke_gpu([], set(), config)


def test_shared_smoke_refuses_busy_or_large_baseline_and_other_card():
    card = parse_gpu_snapshot("1, GPU-one, NVIDIA H100, 0, 2488, 81559\n")[0]
    for changed in (replace(card, utilization=1), replace(card, used_mib=4097),
                    replace(card, index=0)):
        with pytest.raises(ValueError, match="GPU 1"):
            select_qwen_smoke_gpu([changed], {changed.uuid}, shared_smoke_config())


def test_shared_cleanup_checks_our_release_not_whole_card_emptiness():
    card = parse_gpu_snapshot("1, GPU-one, NVIDIA H100, 0, 2488, 81559\n")[0]
    assert compute_gpu_processes("GPU-one, 1234\nGPU-one, 5678\n") == {
        "GPU-one": {1234, 5678}}
    assert gpu_release_verified(card, card, {1234}, {1234}, shared=True)
    assert not gpu_release_verified(card, card, {1234}, {1234})
    assert not gpu_release_verified(card, card, {1234}, {1234, 5678}, shared=True)
    assert not gpu_release_verified(replace(card, used_mib=4096), card,
                                    {1234}, {1234}, shared=True)
