"""Owner grants: historical bounded T0/GPU0 lease and current GPU1 keeper policy.

This is an attestation plus visible guards, never a host/container PID mapping.
No scheduler mutations, CUDA initialization or implicit authority at import.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .provenance import file_sha256


def _utc(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _run(private, name):
    if not isinstance(name, str) or not re.fullmatch(r"cmirank-[a-z0-9-]+-hnv", name):
        raise ValueError("Only a scoped MemRec run receipt can identify our restored keeper")
    path = private / "runs" / name
    if path.resolve() != path or not path.is_dir():
        raise ValueError("Receipt must be a real nonsymlink run inside the private project")
    return path


def standing_authorization(private, grant_path, grant_sha, receipt_run, proof, *, task_kind,
                           timeout_seconds, now=None):
    private, grant_path = Path(private).resolve(), Path(grant_path).resolve()
    if (not grant_path.is_relative_to(private / "approvals") or file_sha256(grant_path) != grant_sha
            or task_kind not in ("ppo_parity", "ppo_compat") or type(timeout_seconds) is not int
            or not 0 < timeout_seconds <= 1500):
        raise ValueError("Standing authority is private, immutable and limited to bounded T0 gates")
    grant = json.loads(grant_path.read_text())
    now = now or datetime.now(timezone.utc)
    issued, expires = _utc(grant["received_at_utc"]), _utc(grant["expires_at_utc"])
    # Include timeout termination/drain and the original keeper's startup audit.
    required_seconds = timeout_seconds + 240
    if (grant["schema_version"] != 1 or grant["gpu_index"] != 0
            or grant["scope"] != "T0_only_own_MemRec_restored_keeper0"
            or grant["gpu1_read_only"] is not True or grant["reserved_job_cancellation_allowed"] is not False
            or grant["host_container_pid_mapping_observed"] is not False
            or proof["job"] != grant["job_id"] or proof["gpu0_uuid"] != grant["gpu_uuid"]
            or not proof["protect_other_workloads"] or now < issued or now >= expires
            or (expires - issued).total_seconds() > 4 * 3600
            or (expires - now).total_seconds() < required_seconds):
        raise ValueError("Expired, insufficient-time or out-of-scope standing resource authority")
    current = _run(private, receipt_run)
    handback = json.loads((current / "keeper-handback.json").read_text())
    if (handback["generator_step"] != proof["target"] or handback["gpu_uuid"] != proof["gpu0_uuid"]
            or handback["members"] != proof["members"] or handback["job"] != proof["job"]):
        raise ValueError("Current keeper step/PID/startticks/UUID differs from our own handback receipt")
    first_sha = file_sha256(current / "keeper-handback.json")
    seen = set()
    for _ in range(24):
        if current.name in seen:
            raise ValueError("Cyclic keeper authority provenance")
        seen.add(current.name)
        handback = json.loads((current / "keeper-handback.json").read_text())
        cleanup = json.loads((current / "cleanup.json").read_text())
        if (handback["status"] != "KEEPER_GPU0_HANDBACK_VERIFIED"
                or handback["job"] != grant["job_id"] or handback["gpu_uuid"] != grant["gpu_uuid"]
                or handback["reserved_job_cancelled"] or handback["gpu1_signalled"]
                or not handback["protect_other_workloads"]
                or not cleanup["child_exited"] or not cleanup["gpu_released"]
                or not cleanup["allocation_still_running"] or cleanup["gpu_uuid"] != grant["gpu_uuid"]):
            raise ValueError("Own predecessor must have released the model and verified keeper0 handback")
        if current.name == grant["initial_handback_run_id"]:
            if file_sha256(current / "keeper-handback.json") != grant["initial_handback_sha256"]:
                raise ValueError("Standing grant's immutable initial handback changed")
            break
        previous = json.loads((current / "gpu0-handoff.json").read_text())["authorization"]
        if (previous["kind"] != "OWNER_STANDING_T0_GRANT" or previous["grant_sha256"] != grant_sha
                or previous["authority_id"] != grant["authority_id"]):
            raise ValueError("A new keeper is not linked to a handoff under this same owner grant")
        current = _run(private, previous["previous_handback_run_id"])
        if file_sha256(current / "keeper-handback.json") != previous["previous_handback_sha256"]:
            raise ValueError("Predecessor handback provenance changed")
    else:
        raise ValueError("Keeper authority chain exceeds its bounded T0 scope")
    return {"kind": "OWNER_STANDING_T0_GRANT", "authority_id": grant["authority_id"],
            "grant_sha256": grant_sha, "received_at_utc": grant["received_at_utc"],
            "expires_at_utc": grant["expires_at_utc"], "checked_at_utc": now.isoformat(),
            "previous_handback_run_id": receipt_run, "previous_handback_sha256": first_sha,
            "task_kind": task_kind, "task_timeout_plus_handback_seconds": required_seconds,
            "host_container_pid_mapping_observed": False}


def gpu1_keeper_authorization(private, grant_path, grant_sha, proof, *, gpu_index, now=None):
    """Researcher explicitly made GPU1 keeper exclusivity persistent, no per-step reply.

    Caller still performs current two-pass UID/cgroup/command/CVD/NVML checks.
    Does not extend authority to another allocation/card or an unknown workload.
    """
    private, grant_path = Path(private).resolve(), Path(grant_path).resolve()
    if (gpu_index != 1 or not grant_path.is_relative_to(private / "approvals")
            or file_sha256(grant_path) != grant_sha):
        raise ValueError("GPU1 standing keeper authority must be scoped and immutable")
    grant = json.loads(grant_path.read_text())
    now = now or datetime.now(timezone.utc)
    if (grant["schema_version"] != 1 or grant["gpu_index"] != 1
            or grant["scope"] != "MemRec_single_GPU1_only_omni_gen_1_handoff"
            or grant["valid_until"] != "researcher_supersedes" or not grant["no_per_step_confirmation_required"]
            or not grant["gpu0_read_only"] or grant["reserved_job_cancellation_allowed"]
            or grant["host_container_pid_mapping_observed"]
            or proof["job"] != grant["job_id"] or proof["gpu1_uuid"] != grant["gpu_uuid"]
            or not proof["protect_other_workloads"] or now < _utc(grant["received_at_utc"])):
        raise ValueError("Standing GPU1 keeper authority does not cover this resource state")
    return {"kind": "OWNER_STANDING_GPU1_KEEPER_EXCLUSIVITY", "authority_id": grant["authority_id"],
            "grant_sha256": grant_sha, "received_at_utc": grant["received_at_utc"],
            "valid_until": grant["valid_until"], "checked_at_utc": now.isoformat(),
            "current_generator_step": proof["target"], "no_per_step_confirmation_required": True,
            "host_container_pid_mapping_observed": False}
