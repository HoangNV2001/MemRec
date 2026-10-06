"""No Slurm/NVML calls or signals; prove the single-step mutation boundary."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cmirank.gpu_resources import GPUCard
import src.cmirank.reserved_handoff as handoff


ROWS = [
    {"StepId": "21273.19", "Name": "omni-gen-1", "UserId": "1052", "State": "RUNNING"},
    {"StepId": "21273.20", "Name": "omni-gen-0", "UserId": "1052", "State": "RUNNING"},
]


@pytest.mark.parametrize("target", ["21273", "21273.batch", "21273.extern", "21273.0", "21273.1", "21273.20", "21274.19"])
def test_confirmation_rejects_job_infrastructure_stale_or_protected_step(target):
    with pytest.raises(ValueError):
        handoff.confirmed_target("21273", target, ROWS, 1052)


def test_current_confirmation_preserves_gpu0_and_rejects_wrong_owner_or_multiple_generators():
    assert handoff.confirmed_target("21273", "21273.19", ROWS, 1052) == "21273.20"
    with pytest.raises(ValueError):
        handoff.confirmed_target("21273", "21273.19", ROWS, 99)
    with pytest.raises(ValueError):
        handoff.confirmed_target("21273", "21273.19", ROWS + [dict(ROWS[0])], 1052)


def test_slurm_parser_is_bound_to_real_uid_state_and_name_fields():
    raw = "StepId=21273.19 UserId=1052 StartTime=2026-10-05T10:36:39 State=RUNNING Name=omni-gen-1\n"
    assert handoff.step_rows(raw)[0]["StepId"] == "21273.19"
    assert handoff.step_rows(raw)[0]["UserId"] == "1052"


def test_process_guard_requires_exact_step_uid_command_and_gpu1_env(monkeypatch):
    monkeypatch.setattr(handoff.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=1052))
    env = {"SLURM_JOB_ID": "21273", "SLURM_STEP_ID": "19", "CUDA_VISIBLE_DEVICES": "1", "GPU": "1"}
    group = "0::/system.slice/slurmstepd.scope/job_21273/step_19/user/task_0\n"
    argv = ["/mnt/data/users/hoangnv242/envs/omnidistill/bin/python", str(handoff.GENERATOR_ROOT / "omni_gen.py")]
    assert handoff.validate_task("21273", "19", 1052, argv, group, env) == "python"
    with pytest.raises(ValueError):
        handoff.validate_task("21273", "19", 1052, argv, group.replace("step_19/", "step_190/"), env)
    with pytest.raises(ValueError):
        handoff.validate_task("21273", "19", 1052, argv, group, {**env, "CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError):
        handoff.validate_task("21273", "19", 99, argv, group, env)
    with pytest.raises(ValueError):
        handoff.validate_task("21273", "19", 1052, ["python", "foreign.py"], group, env)


def setup_fake_handoff(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(handoff, "require_allocation", lambda: "21273")
    monkeypatch.setattr(handoff.os, "geteuid", lambda: 1052)
    proof = {"target": "21273.19", "protected_step": "21273.20", "gpu1_uuid": "GPU-one",
             "nvml_gpu1_pids": [999], "members": [{"pid": 123, "role": "python"}],
             "pid_mapping": "NOT_OBSERVED_OWNER_ATTESTED_CURRENT_STEP_EXCLUSIVITY"}
    monkeypatch.setattr(handoff, "inspect", lambda *a: dict(proof))
    calls = []
    monkeypatch.setattr(handoff.subprocess, "run", lambda args, **kw: calls.append(args))
    monkeypatch.setattr(handoff.subprocess, "check_output", lambda *a, **kw:
        "StepId=21273.20 UserId=1052 State=RUNNING Name=omni-gen-0\n")
    return calls


def test_handoff_signals_only_exact_approved_step_and_keeps_unmapped_pid_warning(monkeypatch, tmp_path):
    calls = setup_fake_handoff(monkeypatch)
    cards = [GPUCard(0, "GPU-zero", "H100", 100, 9000, 81559), GPUCard(1, "GPU-one", "H100", 0, 1, 81559)]
    monkeypatch.setattr(handoff, "gpu_snapshot", lambda: (cards, {"GPU-zero": {77}}, "cards", "apps"))
    result = handoff.handoff(tmp_path, "21273.19")
    assert calls == [["scancel", "21273.19"]]
    assert result["gpu1_idle"] and not result["reserved_job_cancelled"] and not result["gpu0_signalled"]
    assert result["pid_mapping"].startswith("NOT_OBSERVED")
    assert json.loads((tmp_path / "gpu1-handoff.json").read_text())["step_cancelled"]


def test_failed_drain_never_kills_unknown_pid_or_job(monkeypatch, tmp_path):
    calls = setup_fake_handoff(monkeypatch)
    cards = [GPUCard(1, "GPU-one", "H100", 90, 9000, 81559)]
    monkeypatch.setattr(handoff, "gpu_snapshot", lambda: (cards, {"GPU-one": {999}}, "cards", "apps"))
    ticks = iter([0, 0, 50])
    monkeypatch.setattr(handoff.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(handoff.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="did not drain"):
        handoff.handoff(tmp_path, "21273.19")
    assert calls == [["scancel", "21273.19"]]
    assert json.loads((tmp_path / "gpu1-handoff.json").read_text())["status"] == "HANDOFF_DRAIN_FAILED_NO_MODEL_LOAD"


def test_handoff_job_target_is_rejected_before_any_inspection_or_signal(monkeypatch, tmp_path):
    calls = setup_fake_handoff(monkeypatch)
    with pytest.raises(ValueError):
        handoff.handoff(tmp_path, "21273")
    assert calls == []
