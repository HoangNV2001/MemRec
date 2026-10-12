from datetime import datetime, timezone
import json

import pytest

from src.cmirank.provenance import file_sha256
from src.cmirank.t0_authority import gpu1_keeper_authorization, standing_authorization


NOW = datetime(2026, 10, 12, 3, 0, tzinfo=timezone.utc)


def setup(tmp_path):
    run = tmp_path / "runs/cmirank-ppo-parity-old-hnv"
    run.mkdir(parents=True)
    proof = {"job": "42", "target": "42.8", "gpu0_uuid": "GPU-approved", "protect_other_workloads": True,
             "members": [{"pid": 100, "start_ticks": "900", "uid": 123}]}
    handback = {"status": "KEEPER_GPU0_HANDBACK_VERIFIED", "job": "42", "generator_step": "42.8",
               "gpu_uuid": "GPU-approved", "members": proof["members"], "reserved_job_cancelled": False,
               "gpu1_signalled": False, "protect_other_workloads": True}
    cleanup = {"child_exited": True, "gpu_released": True, "allocation_still_running": True,
               "gpu_uuid": "GPU-approved", "process_exit_code": 1}
    (run / "keeper-handback.json").write_text(json.dumps(handback))
    (run / "cleanup.json").write_text(json.dumps(cleanup))
    approvals = tmp_path / "approvals"
    approvals.mkdir()
    grant = {"schema_version": 1, "authority_id": "test-owner-four-hours", "job_id": "42", "gpu_index": 0,
             "gpu_uuid": "GPU-approved", "scope": "T0_only_own_MemRec_restored_keeper0", "gpu1_read_only": True,
             "reserved_job_cancellation_allowed": False, "host_container_pid_mapping_observed": False,
             "received_at_utc": "2026-10-12T02:18:54Z", "expires_at_utc": "2026-10-12T06:13:54Z",
             "initial_handback_run_id": run.name, "initial_handback_sha256": file_sha256(run / "keeper-handback.json")}
    path = approvals / "grant.json"
    path.write_text(json.dumps(grant))
    return run, path, proof


def authorize(tmp_path, path, proof, run, **kwargs):
    return standing_authorization(tmp_path, path, file_sha256(path), run.name, proof,
                                  task_kind="ppo_parity", timeout_seconds=1500, now=NOW, **kwargs)


def test_owner_grant_records_truthful_authority_and_discarded_task_handback(tmp_path):
    run, path, proof = setup(tmp_path)
    result = authorize(tmp_path, path, proof, run)
    assert result["kind"] == "OWNER_STANDING_T0_GRANT"
    assert result["previous_handback_run_id"] == run.name
    assert not result["host_container_pid_mapping_observed"]
    assert result["task_timeout_plus_handback_seconds"] == 1740


@pytest.mark.parametrize("mutation", ["job", "uuid", "pid", "step", "startticks", "gpu1", "model_not_released", "anchor_changed", "lease_expired", "timeout_overruns_lease"])
def test_guards_are_not_replaced_by_a_standing_grant(tmp_path, mutation):
    run, path, proof = setup(tmp_path)
    now = NOW
    if mutation == "job": proof["job"] = "99"
    if mutation == "uuid": proof["gpu0_uuid"] = "GPU-foreign"
    if mutation == "pid": proof["members"][0]["pid"] = 101
    if mutation == "step": proof["target"] = "42.9"
    if mutation == "startticks": proof["members"][0]["start_ticks"] = "901"
    if mutation == "gpu1": proof["protect_other_workloads"] = False
    if mutation == "model_not_released":
        cleanup = json.loads((run / "cleanup.json").read_text())
        cleanup["gpu_released"] = False
        (run / "cleanup.json").write_text(json.dumps(cleanup))
    if mutation == "anchor_changed":
        with (run / "keeper-handback.json").open("a") as handle: handle.write("\n")
    if mutation == "lease_expired": now = datetime(2026, 10, 12, 6, 14, tzinfo=timezone.utc)
    if mutation == "timeout_overruns_lease": now = datetime(2026, 10, 12, 6, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        standing_authorization(tmp_path, path, file_sha256(path), run.name, proof,
                               task_kind="ppo_parity", timeout_seconds=1500, now=now)


def test_followup_keeper_must_link_back_to_same_grant_and_own_verified_handback(tmp_path):
    run, path, proof = setup(tmp_path)
    old_authority = authorize(tmp_path, path, proof, run)
    later = tmp_path / "runs/cmirank-ppo-parity-next-hnv"
    later.mkdir()
    handback = json.loads((run / "keeper-handback.json").read_text())
    handback["generator_step"] = "42.12"
    (later / "keeper-handback.json").write_text(json.dumps(handback))
    (later / "cleanup.json").write_bytes((run / "cleanup.json").read_bytes())
    (later / "gpu0-handoff.json").write_text(json.dumps({"authorization": old_authority}))
    proof["target"] = "42.12"
    assert authorize(tmp_path, path, proof, later)["previous_handback_run_id"] == later.name
    old_authority["grant_sha256"] = "different-grant"
    (later / "gpu0-handoff.json").write_text(json.dumps({"authorization": old_authority}))
    with pytest.raises(ValueError): authorize(tmp_path, path, proof, later)


def test_standing_grant_never_authorizes_training_or_unknown_resource_scope(tmp_path):
    run, path, proof = setup(tmp_path)
    with pytest.raises(ValueError):
        standing_authorization(tmp_path, path, file_sha256(path), run.name, proof,
                               task_kind="full_ppo_training", timeout_seconds=1500, now=NOW)


@pytest.mark.parametrize("mutation", [None, "gpu0", "other_job", "other_uuid", "protected_gpu_write", "changed_receipt"])
def test_persistent_gpu1_owner_attestation_is_not_fresh_confirmation_or_pid_mapping(tmp_path, mutation):
    _, path, _ = setup(tmp_path)
    grant = {"schema_version": 1, "authority_id": "current-gpu1", "received_at_utc": "2026-10-12T02:18:54Z",
             "scope": "MemRec_single_GPU1_only_omni_gen_1_handoff", "valid_until": "researcher_supersedes",
             "job_id": "42", "gpu_index": 1, "gpu_uuid": "GPU-approved-1", "gpu0_read_only": True,
             "no_per_step_confirmation_required": True, "reserved_job_cancellation_allowed": False,
             "host_container_pid_mapping_observed": False}
    path.write_text(json.dumps(grant))
    sha = file_sha256(path)
    proof = {"job": "42", "gpu1_uuid": "GPU-approved-1", "target": "42.777", "protect_other_workloads": True}
    index = 1
    if mutation == "gpu0": index = 0
    if mutation == "other_job": proof["job"] = "43"
    if mutation == "other_uuid": proof["gpu1_uuid"] = "GPU-unknown"
    if mutation == "protected_gpu_write": proof["protect_other_workloads"] = False
    if mutation == "changed_receipt": path.write_text(json.dumps({**grant, "job_id": "43"}))
    if mutation is not None:
        with pytest.raises(ValueError):
            gpu1_keeper_authorization(tmp_path, path, sha, proof, gpu_index=index, now=NOW)
    else:
        result = gpu1_keeper_authorization(tmp_path, path, sha, proof, gpu_index=index, now=NOW)
        assert result["kind"] == "OWNER_STANDING_GPU1_KEEPER_EXCLUSIVITY"
        assert result["no_per_step_confirmation_required"] and not result["host_container_pid_mapping_observed"]
        assert result["current_generator_step"] == "42.777"
