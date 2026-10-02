"""Small synthetic promotion tests; never read real user outcome labels."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.cmirank.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "candidate_audit", ROOT / "scripts/cmirank/08_audit_candidates_cpu.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def promotion_fixture(tmp_path):
    config = {"smoke_include_user_ids": [18, 19], "smoke_users": 20}
    recipes = {"episode_config": config, "episode_contract_sha256": "episode-sha",
               "candidate_contract_sha256": "candidate-sha", "index_manifest_sha256": "index-sha"}
    inputs = {"snapshot_sha256": "snapshot-sha", "manifest_sha256": "split-sha",
              "warmups": dict.fromkeys(range(22), 100)}
    artifact = tmp_path / "candidate-rows.jsonl"
    artifact.write_text('{}\n')
    report = {"status": "CANDIDATE_INTEGRITY_SMOKE_PASS_NOT_G0_OR_RANKING_RESULT",
              "source_commit": "source-sha", "users": 20, "candidate_sets": 40,
              "graph_snapshot_sha256": "snapshot-sha", "policy_split_manifest_sha256": "split-sha",
              "episode_contract_sha256": "episode-sha", "candidate_contract_sha256": "candidate-sha",
              "index_manifest_sha256": "index-sha", "query_user_ids": list(range(20)),
              "candidate_manifest_sha256": file_sha256(artifact)}
    (tmp_path / "report.json").write_text(json.dumps(report))
    return recipes, inputs, report


def test_full_gate_requires_exact_source_scope_contract_and_integrity(tmp_path):
    recipes, inputs, report = promotion_fixture(tmp_path)
    assert audit.verify_full_gate(tmp_path, source_commit="source-sha", recipes=recipes,
                                  inputs=inputs) == report
    (tmp_path / "candidate-rows.jsonl").write_text('{"tampered":true}\n')
    with pytest.raises(ValueError, match="no matching"):
        audit.verify_full_gate(tmp_path, source_commit="source-sha", recipes=recipes, inputs=inputs)


@pytest.mark.parametrize("field,value", [
    ("status", "FAILED"), ("users", 19), ("source_commit", "other-source"),
    ("episode_contract_sha256", "other-episode"), ("graph_snapshot_sha256", "other-graph"),
    ("query_user_ids", list(range(1, 21))),
])
def test_full_gate_does_not_promote_a_different_smoke(tmp_path, field, value):
    recipes, inputs, report = promotion_fixture(tmp_path)
    report[field] = value
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="no matching"):
        audit.verify_full_gate(tmp_path, source_commit="source-sha", recipes=recipes, inputs=inputs)
