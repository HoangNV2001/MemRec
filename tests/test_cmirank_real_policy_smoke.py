"""Real-input diagnostic wiring and handback gates, no model/CUDA/network."""

from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

import pytest

from src.cmirank.labels import make_labels
from src.cmirank.parser import parse_direct_ranking
from src.cmirank.policy_smoke import load_secondary_inputs, run_functional_smoke, verify_cpu_token_audit
from src.cmirank.prompts import render_direct_prompt, render_step_prompt
from src.cmirank.provenance import file_sha256
from src.cmirank.smoke_fixtures import synthetic_rank_requests

ROOT = Path(__file__).resolve().parents[1]


def generator(prompt):
    labels = re.findall(r"^\[(C\d{2})\] Title:", prompt, re.MULTILINE)
    answer = labels[-1] if "REMAINING CANDIDATES" in prompt else " ".join(reversed(labels))
    return {"raw_output": f"<answer>{answer}</answer>", "complete": True,
            "input_tokens": 100, "output_tokens": 10, "seconds": .01}


def test_twenty_user_complete_nminus1_and_direct_share_requests_without_scoring():
    emitted = defaultdict(list)
    requests = synthetic_rank_requests(20, 42)
    result = run_functional_smoke(requests, generate=generator, emit=lambda n, row: emitted[n].append(row))
    assert result["status"] == "FUNCTIONAL_SMOKE_PASS"
    assert (result["iterative_valid"], result["direct_valid"], result["iterative_steps"], result["generation_calls"]) == (20, 20, 180, 200)
    assert not result["training_ready"] and not result["ranking_metrics_computed"] and result["model_updates"] == 0
    for sample in range(20):
        rows = [row for row in emitted["episode"] if row["sample"] == sample]
        assert len(rows) == 2 and rows[0]["rank_request_sha256"] == rows[1]["rank_request_sha256"]
        expected = {row["id"] for row in requests[sample].candidates}
        assert all(set(row["ranked_candidate_ids"]) == expected for row in rows)
        assert len(rows[0]["trace"]) == 9
        assert [len(row["active_labels_before"]) for row in rows[0]["trace"]] == list(range(10, 1, -1))


@pytest.mark.parametrize("failure", ["inactive_label", "duplicate_direct", "truncated"])
def test_invalid_output_never_repaired_or_promoted(failure):
    calls, emitted = [], []

    def bad(prompt):
        calls.append(prompt)
        out = generator(prompt)
        if failure == "inactive_label" and "REMAINING CANDIDATES" in prompt:
            out["raw_output"] = "<answer>C99</answer>"
        elif failure == "duplicate_direct" and "REMAINING CANDIDATES" not in prompt:
            out["raw_output"] = "<answer>" + " ".join(["C00"] * 10) + "</answer>"
        elif failure == "truncated":
            out["complete"] = False
        return out

    result = run_functional_smoke(synthetic_rank_requests(20, 42), generate=bad,
                                 emit=lambda n, row: emitted.append((n, deepcopy(row))))
    assert result["status"] == "FUNCTIONAL_SMOKE_FAILED_NO_PROMOTION"
    assert result["generation_calls"] == len(calls) <= 200 and not result["output_repair"]
    failures = [row for n, row in emitted if n == "episode" and not row["valid"]]
    assert failures and all(row["ranked_candidate_ids"] == [] for row in failures)
    if failure != "duplicate_direct":
        assert len(calls) == 40  # Failed first iterative action, then one direct call per user.


def test_direct_prompt_matches_iterative_evidence_payload_and_150_char_cap():
    request = synthetic_rank_requests(20, 42)[0]
    labels = make_labels(range(10))
    iterative, direct = render_step_prompt(request, labels), render_direct_prompt(request)
    assert iterative.split("Target user:", 1)[1].split("\nREMAINING CANDIDATES", 1)[0] == direct.split("Target user:", 1)[1].split("\nCANDIDATES", 1)[0]
    def payload(prompt):
        return re.findall(r"\[C\d{2}\] Title:.*?\nItem memory:.*?\n", prompt)
    assert payload(iterative) == payload(direct) and len(payload(direct)) == 10


@pytest.mark.parametrize("raw", ["C00 C01", "<answer>C00,C01</answer>", "<answer>C00 C00</answer>",
                                  "<answer>C00 C99</answer>", "<answer>C00 C01</answer><answer>C00 C01</answer>"])
def test_direct_parser_is_a_complete_permutation_parser(raw):
    assert not parse_direct_ranking(raw, ["C00", "C01"]).valid
    assert parse_direct_ranking("<answer>C01 C00</answer>", ["C00", "C01"]).labels == ("C01", "C00")


def input_fixture(tmp_path):
    run, review = tmp_path / "secondary-run", tmp_path / "review.json"
    run.mkdir()
    requests = synthetic_rank_requests(20, 42)
    # Synthetic fixture user IDs are unique, and their order is deterministic.
    requests = sorted(requests, key=lambda request: request.user_id)
    users = [request.user_id for request in requests]
    rows = [{"episode_id": f"user-{request.user_id}-pseudo", "rank_request": request.to_dict(),
             "memory_state_sha256": "same-state", "warmup_scope_user_ids": users} for request in requests]
    source = run / "policy-inputs.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {"source_run_id": run.name, "status": "SECONDARY_CONTROL_REVIEW_PASS_LITERAL_VALIDITY_NOT_SEMANTIC_GROUNDING",
        "pseudo_memory_read_only_verified": True, "primary_provider_replaced": False, "unknown_context_occurrences": 0,
        "training_ready": False, "replayed_prefix_memory_sha256": "same-state",
        "source_artifact_sha256": {source.name: file_sha256(source)}}
    review.write_text(json.dumps(receipt))
    return run, review


def test_loader_uses_reviewed_secondary_policy_only_and_rejects_changed_file(tmp_path):
    run, review = input_fixture(tmp_path)
    requests, proof = load_secondary_inputs(run, review, expected_review_sha=file_sha256(review))
    assert len(requests) == 20 and not proof["training_ready"]
    with (run / "policy-inputs.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="Frozen policy inputs changed"):
        load_secondary_inputs(run, review, expected_review_sha=file_sha256(review))


def test_loader_rejects_label_injection_even_with_a_rehashed_fixture_receipt(tmp_path):
    run, review = input_fixture(tmp_path)
    source = run / "policy-inputs.jsonl"
    rows = [json.loads(row) for row in source.read_text().splitlines()]
    rows[0]["positive_item_id"] = 123
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = json.loads(review.read_text())
    receipt["source_artifact_sha256"][source.name] = file_sha256(source)
    review.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="label separation"):
        load_secondary_inputs(run, review, expected_review_sha=file_sha256(review))


def token_receipt_fixture(tmp_path):
    proof = {"user_ids": list(range(20)), "rank_request_hashes": [str(i) for i in range(20)],
             "training_ready": False, "memory_state_sha256": "same-state"}
    rows = [{"user_id": uid, "mode": mode, "tokens": 100}
            for uid in proof["user_ids"] for mode in ("iterative", "direct")]
    audit = tmp_path / "cpu-token-audit.json"
    audit.write_text(json.dumps(rows))
    args = {"commit": "source", "config_sha": "config", "prompt_sha": "prompt", "parser_sha": "parser",
            "marker_sha": "marker", "provenance": proof, "max_input": 4096}
    manifest = {"source_commit": "source", "config_sha256": "config", "prompt_code_sha256": "prompt",
                "parser_code_sha256": "parser", "checkpoint_marker_sha256": "marker", "cpu_audit_only": True}
    report = {"status": "CPU_REAL_POLICY_INPUT_TOKEN_AUDIT_PASS_NO_GPU", "source_commit": "source",
              "config_sha256": "config", "gpu_requested": False, "model_weights_loaded": False,
              "token_audit_sha256": file_sha256(audit), **proof}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "report.json").write_text(json.dumps(report))
    return args


def test_fresh_gpu_process_requires_matching_cpu_token_source_and_state(tmp_path):
    args = token_receipt_fixture(tmp_path)
    assert len(verify_cpu_token_audit(tmp_path, **args)) == 40
    with pytest.raises(ValueError, match="exact GPU source"):
        verify_cpu_token_audit(tmp_path, **{**args, "commit": "different-source"})


@pytest.mark.parametrize("tamper", ["too_long", "phase_order", "checkpoint"])
def test_cpu_token_guard_rejects_rehashed_bad_length_order_and_checkpoint(tmp_path, tamper):
    args = token_receipt_fixture(tmp_path)
    if tamper == "checkpoint":
        path = tmp_path / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["checkpoint_marker_sha256"] = "changed"
        path.write_text(json.dumps(manifest))
    else:
        path = tmp_path / "cpu-token-audit.json"
        rows = json.loads(path.read_text())
        if tamper == "too_long":
            rows[0]["tokens"] = 4097
        else:
            rows[0], rows[1] = rows[1], rows[0]
        path.write_text(json.dumps(rows))
        report_path = tmp_path / "report.json"
        report = json.loads(report_path.read_text())
        report["token_audit_sha256"] = file_sha256(path)
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_cpu_token_audit(tmp_path, **args)


def keeper_module():
    spec = importlib.util.spec_from_file_location("keeper_cpu", ROOT / "scripts/cmirank/14_keeper_handback_cpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handback_deduplicates_existing_keeper_and_only_starts_gpu1():
    keeper = keeper_module()
    rows = [{"StepId": "21820.35", "Name": "omni-gen-0", "State": "RUNNING", "UserId": "1052"}]
    assert keeper.handback_state("21820", rows, 1052) == ("START_GPU1_KEEPER", None, "21820.35")
    rows.append({"StepId": "21820.93", "Name": "omni-gen-1", "State": "RUNNING", "UserId": "1052"})
    assert keeper.handback_state("21820", rows, 1052) == ("ALREADY_RUNNING", "21820.93", "21820.35")
    with pytest.raises(ValueError):
        keeper.handback_state("21820", rows + [rows[-1]], 1052)
    with pytest.raises(ValueError):
        keeper.handback_state("21820", [dict(rows[0], UserId="999")], 1052)
