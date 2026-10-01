"""Regression tests for Transformers' set-valued checkpoint loading reports."""

import json

import pytest

from src.cmirank.provenance import artifact_json_dumps


def test_loading_report_sets_are_deterministic_arrays_without_mutation():
    report = {"missing_keys": set(), "unexpected_keys": {"mtp.z", "mtp.a"},
              "error_msgs": [], "nested": {"names": frozenset({"b", "a"})}}
    encoded = artifact_json_dumps(report)
    assert json.loads(encoded) == {
        "missing_keys": [], "unexpected_keys": ["mtp.a", "mtp.z"],
        "error_msgs": [], "nested": {"names": ["a", "b"]},
    }
    assert encoded == artifact_json_dumps(report)
    assert encoded.endswith("\n")
    assert isinstance(report["missing_keys"], set)
    assert report["unexpected_keys"] == {"mtp.z", "mtp.a"}


def test_diagnostic_serializer_does_not_silently_stringify_unknown_types():
    with pytest.raises(TypeError, match="not JSON serializable"):
        artifact_json_dumps({"unexpected": object()})


def test_diagnostic_serializer_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        artifact_json_dumps({"loss": float("nan")})
