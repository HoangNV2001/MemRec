from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.cmirank.memory_smoke import load_memory_contract
from src.cmirank.primary_memory import primary_user_ids
from src.cmirank.provenance import file_sha256
from src.cmirank.t0_authority import gpu1_keeper_authorization

ROOT = Path(__file__).resolve().parents[1]


def test_primary_train20_keeps_original_unconstrained_provider_and_budget():
    old, _ = load_memory_contract(ROOT, 6)
    new, _ = load_memory_contract(ROOT, 8)
    for key in ("model_id", "model_revision", "backend", "agent", "temperature", "max_tokens",
            "max_model_len", "gpu_memory_utilization", "physical_request_cap", "nominal_physical_requests",
            "candidate_run_id", "full_candidates_sha256", "audit_report_sha256", "output_repair"):
        assert old[key] == new[key]
    assert "constrained_decoding_delta" not in new and new["decoding_changes"] is False
    assert new["comparison_role"].startswith("primary_upstream_aligned") and not new["training_ready"]


def test_train20_selection_never_reads_targets_or_uses_validation_users():
    c, _ = load_memory_contract(ROOT, 8)
    inputs = {"groups": {"policy_train": list(range(1497)), "policy_val": list(range(2000, 2300))}}
    users = primary_user_ids(inputs, c)
    inputs["groups"]["policy_train"].reverse()
    assert primary_user_ids(inputs, c) == users
    assert users == sorted(set(users)) and len(users) == 20
    assert set(users).issubset(range(1497)) and set(users).isdisjoint(range(2000, 2300))


@pytest.mark.parametrize("bad", ["overlap", "drop_train", "duplicate_val"])
def test_no_cohort_change_or_automatic_replacement(bad):
    c, _ = load_memory_contract(ROOT, 8)
    inputs = {"groups": {"policy_train": list(range(1497)), "policy_val": list(range(2000, 2300))}}
    if bad == "overlap": inputs["groups"]["policy_val"][0] = 1
    elif bad == "drop_train": inputs["groups"]["policy_train"].pop()
    else: inputs["groups"]["policy_val"][0] = inputs["groups"]["policy_val"][1]
    with pytest.raises(ValueError): primary_user_ids(inputs, c)


@pytest.mark.parametrize("key,value", [("gpu_index", 0), ("decoding_changes", True), ("cohort", "policy_val"),
    ("selection_seed", "choose_another_seed"), ("output_repair", True)])
def test_primary_profile_cannot_sneak_in_control_or_outcome_tuning(tmp_path, key, value):
    folder = tmp_path/"configs/cmirank"; folder.mkdir(parents=True)
    for p in (ROOT/"configs/cmirank").glob("real_memory_smoke_v*.json"):
        (folder/p.name).write_bytes(p.read_bytes())
    path = folder/"real_memory_smoke_v8.json"; c = json.loads(path.read_text())
    c[key] = value; path.write_text(json.dumps(c))
    with pytest.raises(ValueError):load_memory_contract(tmp_path, 8)


def test_new_allocation_needs_fresh_job_bound_authority_not_edited_old_receipt(tmp_path):
    folder = tmp_path/"approvals"; folder.mkdir()
    grant = {"schema_version": 1, "authority_id": "previous", "job_id": "21820", "gpu_index": 1,
        "gpu_uuid": "GPU-selected", "received_at_utc": "2026-10-12T06:36:49Z",
        "scope": "MemRec_single_GPU1_only_omni_gen_1_handoff", "valid_until": "researcher_supersedes",
        "gpu0_read_only": True, "no_per_step_confirmation_required": True,
        "reserved_job_cancellation_allowed": False, "host_container_pid_mapping_observed": False}
    old = folder/"old.json"; old.write_text(json.dumps(grant)); old_sha = file_sha256(old)
    proof = {"job": "23851", "gpu1_uuid": "GPU-selected", "target": "23851.3", "protect_other_workloads": True}
    with pytest.raises(ValueError):gpu1_keeper_authorization(tmp_path, old, old_sha, proof, gpu_index=1)
    new = folder/"new.json"; new.write_text(json.dumps({**grant, "job_id": "23851", "authority_id": "new-owner-update"}))
    assert gpu1_keeper_authorization(tmp_path, new, file_sha256(new), proof, gpu_index=1)["no_per_step_confirmation_required"]
    assert file_sha256(old) == old_sha


def test_primary_uses_original_protected_release_and_keeper_wrapper():
    for name in ("run_real_policy_smoke.sh", "run_real_policy_with_keeper.sh"):
        text = (ROOT/"scripts/cmirank"/name).read_text()
        assert '"$MEMREC_TASK_KIND" == primary_memory' in text and "scancel" not in text
    source = (ROOT/"scripts/cmirank/10_smoke_real_memory.py").read_text()
    assert "verify_primary_preparation" in source and "protect_other_workloads=True" in source
