"""CPU-only checks of the shared-cluster selection and infrastructure fixtures."""

import pytest

from src.cmirank.gpu_resources import compute_gpu_uuids, parse_gpu_snapshot, select_idle_h100
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
