"""Primary MemRec memory on a predeclared train-only smoke cohort, no CUDA."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .provenance import file_sha256


def primary_user_ids(inputs, c):
    train, val = inputs["groups"]["policy_train"], inputs["groups"]["policy_val"]
    if (len(train) != 1497 or len(set(train)) != 1497 or len(val) != 300 or len(set(val)) != 300
            or set(train) & set(val) or c["cohort"] != "locked_policy_train_only"):
        raise ValueError("The original locked1497/300 split must remain unchanged")
    ordered = sorted(train, key=lambda uid: (hashlib.sha256(
        f"{c['selection_seed']}:{uid}".encode()).digest(), uid))
    return sorted(ordered[:20])


def primary_code_hashes(root, version=8):
    names = ("configs/cmirank/real_memory_smoke_v8.json", "src/cmirank/primary_memory.py",
        "scripts/cmirank/43_prepare_primary_memory_cpu.py", "scripts/cmirank/10_smoke_real_memory.py",
        "src/cmirank/memory_smoke.py", "src/cmirank/policy_inputs.py", "src/cmirank/shortcut_audit.py",
        "src/cmirank/candidate_artifacts.py", "src/models/memrec_agent.py", "src/memory/manager.py",
        "src/models/reranker_llm.py", "scripts/cmirank/run_real_policy_smoke.sh",
        "scripts/cmirank/run_real_policy_with_keeper.sh", "src/cmirank/t0_authority.py",
        "src/cmirank/reserved_handoff.py", "src/cmirank/gpu_resources.py")
    if version == 9:
        names += ("configs/cmirank/real_memory_smoke_v9.json", "src/cmirank/memory_review.py",
                  "scripts/cmirank/11_review_memory_smoke_cpu.py")
    elif version != 8:
        raise ValueError("Unknown primary memory profile")
    return {n: file_sha256(root/n) for n in names}


def reviewed_primary_failure(private, c):
    """The new warning protocol cannot rewrite/promote its strict predecessor."""
    delta = c["source_citation_delta"]
    path = private/"runs"/delta["failed_v8_review_run_id"]/"review.json"
    if file_sha256(path) != delta["failed_v8_review_sha256"]:
        raise ValueError("Independent immutable strict-v8 review required")
    report = json.loads(path.read_text())
    run = private/"runs"/delta["failed_v8_run_id"]
    if (report["status"] != "FAILED_MEMORY_SMOKE_OFFLINE_REVIEW_COMPLETE_NOT_PROMOTED"
            or report["literal_grounding_gate_passed"] or report["training_ready"]
            or not report["review_source_tracked_clean"] or not report["gpu_released"]
            or not report["keeper_restored"] or report["physical_requests"] != 23
            or report["completed_warmup_users"] != 7 or report["completed_pseudo_users"]
            or report["unknown_context_occurrences"] != 1):
        raise ValueError("Preserved strict failure/release review differs")
    for name, digest in report["source_artifact_sha256"].items():
        if Path(name).name != name or file_sha256(run/name) != digest:
            raise ValueError("Preserved failed primary run changed")
    return report


def primary_warning_probes(details, journal, uid, clicked, fanout_cap):
    """CPU-only warning/hard-gate probes on copies, never repair model output."""
    from .memory_smoke import object_sha256, validate_retrieval, validate_scores, validate_write
    from src.memory.manager import MemRecManager
    from src.models.llm_client import validate_json_shape
    schema = {"type": "object", "properties": MemRecManager(None).get_stage_r_schema()}
    before = object_sha256({"details": details, "journal": journal})
    def retrieval(d):
        validate_json_shape(d["retrieval_bundle"], schema)
        return validate_retrieval(d, uid, candidate_context_warning=True, unknown_source_warning=True)
    audit = retrieval(details)
    if not audit["unknown_context_citations"]:
        raise ValueError("CPU warning acceptance probe must contain a synthetic unknown source")
    rejected = []
    def reject(name, action):
        try:
            action()
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError(f"Preserved hard gate unexpectedly accepted {name}")
    reject("legacy_unknown_source_still_fails", lambda: validate_retrieval(details, uid, candidate_context_warning=True))
    for name, value in (("confidence_low", -0.1), ("confidence_high", 1.1), ("confidence_bool", True),
                        ("confidence_nan", float("nan"))):
        d = deepcopy(details); d["retrieval_bundle"]["facets"][0]["confidence"] = value
        reject(name, lambda: retrieval(d))
    for name in ("empty_facets", "blank_facet", "wrong_edge_target", "invalid_edge_weight"):
        d = deepcopy(details)
        if name == "empty_facets": d["retrieval_bundle"]["facets"] = []
        elif name == "blank_facet": d["retrieval_bundle"]["facets"][0]["facet"] = " "
        else:
            d["retrieval_bundle"]["support_edges"] = [{"from": audit["unknown_context_citations"][0]["identifier"],
                "to": f"User-{uid+1}" if name == "wrong_edge_target" else f"User-{uid}",
                "w": 1.1 if name == "invalid_edge_weight" else 0.5}]
        reject(name, lambda: retrieval(d))
    ids = audit["visible_candidate_ids"]
    scores = [{"item_id": item, "score": 0.5} for item in ids]
    validate_scores(scores, ids)
    for name in ("candidate_missing", "candidate_duplicate", "candidate_unknown", "score_bool"):
        s = deepcopy(scores)
        if name == "candidate_missing": s.pop()
        elif name == "candidate_duplicate": s[0]["item_id"] = s[1]["item_id"]
        elif name == "candidate_unknown": s[0]["item_id"] = max(ids)+1
        else: s[0]["score"] = True
        reject(name, lambda: validate_scores(s, ids))
    allowed = [f"{n['type'].capitalize()}-{n['id']}" for n in details["pruned_subgraph"]["neighbors"]]
    unknown = "Item-999999999999"
    if unknown in allowed or len(allowed) <= fanout_cap:
        raise ValueError("CPU write probes require an unknown ID and enough real neighbors")
    for name in ("stage_w_unknown_neighbor", "stage_w_duplicate_neighbor", "stage_w_fanout", "stage_w_unrequested_mutation"):
        result, mutation = deepcopy(journal["result"]), deepcopy(journal["mutation"])
        identifiers = ([unknown] if name == "stage_w_unknown_neighbor" else
                       [allowed[0]]*2 if name == "stage_w_duplicate_neighbor" else
                       allowed[:fanout_cap+1] if name == "stage_w_fanout" else [])
        result["raw_response"]["neighbor_updates"] = [{"neighbor_id": n, "memory_update": "probe", "rationale": "probe"} for n in identifiers]
        if name == "stage_w_unrequested_mutation": mutation["users"][uid+1] = "probe"
        reject(name, lambda: validate_write(result, mutation, details, uid, clicked, fanout_cap))
    if object_sha256({"details": details, "journal": journal}) != before:
        raise ValueError("CPU probes mutated their raw inputs")
    return {"warning_acceptances": 1, "hard_rejections": rejected, "raw_probe_inputs_unchanged": True}


def reviewed_ppo_mechanics(private, c):
    run = private/"runs"/c["ppo_smoke_run_id"]
    if (file_sha256(run/"report.json") != c["ppo_smoke_report_sha256"]
            or file_sha256(run/"native-worker-report.json") != c["ppo_smoke_worker_sha256"]):
        raise ValueError("Reviewed both-role PPO smoke changed")
    r = json.loads((run/"report.json").read_text())
    worker = json.loads((run/"native-worker-report.json").read_text())
    cleanup = json.loads((run/"cleanup.json").read_text())
    handback = json.loads((run/"keeper-handback.json").read_text())
    if (r["status"] != "NATIVE_LORA_MULTITURN_VALID_AND_INVALID_PPO_SMOKE_PASS_REVIEW_REQUIRED"
            or r["optimizer_updates"] != 8 or r["samples"] != 20 or r["training_ready"]
            or worker["roles"]["actor"]["optimizer_updates"] != 4
            or worker["roles"]["critic"]["optimizer_updates"] != 4
            or not worker["native_episode_continuous_gae"] or worker["books_outcomes_accessed"]
            or cleanup["process_exit_code"] or not cleanup["gpu_released"]
            or handback["status"] != "KEEPER_GPU1_HANDBACK_VERIFIED"
            or handback["reserved_job_cancelled"] or handback["gpu0_signalled"]):
        raise ValueError("Completed/released mechanics smoke required, not a quality readiness claim")
    # The historical receipt stays attached to its original allocation.
    # A new current allocation needs its own resource grant/preflight.
    return r


def verify_primary_preparation(private, root, c, contract_sha, inputs, users, commit, job):
    directory = private/"runs"/c["prepare_run_id"]
    p = json.loads((directory/"report.json").read_text())
    if (p["status"] != "PRIMARY_TRAIN20_CPU_WIRING_PASS_REAL_MEMORY_GATE_REMAINS"
            or p["source_commit"] != commit or p["config_sha256"] != contract_sha
            or p["code_sha256"] != primary_code_hashes(root, c["schema_version"]) or p["slurm_job_id"] != job
            or p["query_user_ids"] != users or p["graph_snapshot_sha256"] != inputs["snapshot_sha256"]
            or p["split_sha256"] != inputs["manifest_sha256"] or p["real_llm_requests"]
            or p["gpu_requested"] or p["training_ready"] or p["fake_logical_requests"] != 100):
        raise ValueError("Fresh current-allocation exact-source CPU20 preparation required")
    for name, sha in p["artifact_sha256"].items():
        if Path(name).name != name or file_sha256(directory/name) != sha:
            raise ValueError("Primary CPU cohort/probes changed")
    reviewed_ppo_mechanics(private, c)
    if "source_citation_delta" in c:
        reviewed_primary_failure(private, c)
        probes = p["source_citation_warning_probes"]
        if (probes["warning_acceptances"] != 1 or len(set(probes["hard_rejections"])) != 17
                or not probes["raw_probe_inputs_unchanged"]
                or p["fake_wiring_summary"]["unknown_context_citation_occurrences"] != 40
                or p["fake_wiring_summary"]["literal_grounding_gate_passed"]):
            raise ValueError("Fresh raw-citation warning and preserved-hard-gate probes required")
    return p
