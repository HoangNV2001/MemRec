"""Real-memory smoke gates; unchanged MemRec prompts and no policy training.

V7 is an explicit secondary decoding control, not unchanged upstream decoding.

The common graph covers 1,797 query users, but this diagnostic warms ONLY the
20 predeclared smoke users. Its final requests cannot stand in for a full
1,797-user warmed cache. Semantic grounding is not proven by these guards.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path

from .policy_data import NEUTRAL_PSEUDO_INSTRUCTION
from .request import RankRequest
from .provenance import file_sha256


def load_memory_contract(root: Path, version: int) -> tuple[dict, str]:
    base_path = root / "configs/cmirank/real_memory_smoke_v1.json"
    config = json.loads(base_path.read_text())
    validate_memory_contract(config)
    base_sha = file_sha256(base_path)
    if version == 1:
        return config, base_sha
    if version not in (2, 3, 4, 5, 6, 7):
        raise ValueError("Unsupported memory smoke version")
    delta_path = root / "configs/cmirank/real_memory_smoke_v2.json"
    delta = json.loads(delta_path.read_text())
    expected = {"schema_version": 2, "predecessor_contract_sha256": base_sha,
        "run_id": "cmirank-real-memory-smoke-v2-20261005-hnv",
        "change_only": "vllm_0102_cuda_device_binding", "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": "numeric_nvml_index_after_full_pci_order_check_and_selected_uuid_assertion",
        "training_ready": False}
    if delta != expected:
        raise ValueError("V2 may change only vLLM device binding, not the memory experiment")
    config["run_id"] = delta["run_id"]
    config["device_binding_delta"] = delta
    hashes = {"base_sha256": base_sha, "delta_sha256": file_sha256(delta_path)}
    if version >= 3:
        identity_path = root / "configs/cmirank/real_memory_smoke_v3.json"
        identity_delta = json.loads(identity_path.read_text())
        if identity_delta != {"schema_version": 3, "predecessor_contract_sha256": hashes["delta_sha256"],
            "run_id": "cmirank-real-memory-smoke-v3-20261005-hnv",
            "change_only": "canonical_uuid_representation_for_device_identity_assertion",
            "uuid_comparison": "full_128_bit_uuid_optional_gpu_prefix_canonical_lowercase",
            "training_ready": False}:
            raise ValueError("V3 may fix only canonical GPU UUID comparison")
        config["run_id"] = identity_delta["run_id"]
        config["device_identity_delta"] = identity_delta
        hashes["identity_delta_sha256"] = file_sha256(identity_path)
    if version >= 4:
        role_path = root / "configs/cmirank/real_memory_smoke_v4.json"
        role_delta = json.loads(role_path.read_text())
        expected = {"schema_version": 4,
            "approval_status": "researcher_continue_after_explicit_evidence_role_proposal_2026-10-05",
            "predecessor_contract_sha256": hashes["identity_delta_sha256"],
            "run_id": "cmirank-real-memory-smoke-v4-20261005-hnv",
            "change_only": "stage_r_visible_candidate_citations_are_role_warnings_not_invented_ids",
            "candidate_context_citations": "retain_raw_upstream_output_record_role_warning_no_semantic_safety_claim",
            "unknown_context_ids": "hard_fail", "schema_labels_provenance_stage_w_gates": "unchanged_hard_fail",
            "model_prompt_data_decoding_changes": False, "output_repair": False, "training_ready": False}
        if role_delta != expected:
            raise ValueError("V4 may classify only visible candidate citations; other hard gates unchanged")
        config["run_id"] = role_delta["run_id"]
        config["evidence_role_delta"] = role_delta
        hashes["evidence_role_delta_sha256"] = file_sha256(role_path)
    if version >= 5:
        resource_path = root / "configs/cmirank/real_memory_smoke_v5.json"
        resource_delta = json.loads(resource_path.read_text())
        expected = {"schema_version": 5,
            "predecessor_contract_sha256": hashes["evidence_role_delta_sha256"],
            "run_id": "cmirank-real-memory-smoke-v5-20261006-hnv",
            "change_only": "generator_wrapper_initial_reservation_mask_not_compute_child_mask",
            "wrapper_initial_cuda_masks": ["1", "0,1"], "python_cuda_mask": "1",
            "gpu1_own_uid_cgroup_command_and_owner_confirmation": "unchanged_required",
            "model_prompt_data_decoding_changes": False, "output_repair": False, "training_ready": False}
        if resource_delta != expected:
            raise ValueError("V5 changes only the generator wrapper initial mask check")
        config["run_id"] = resource_delta["run_id"]
        config["generator_mask_delta"] = resource_delta
        hashes["generator_mask_delta_sha256"] = file_sha256(resource_path)
    if version >= 6:
        cache_path = root / "configs/cmirank/real_memory_smoke_v6.json"
        cache_delta = json.loads(cache_path.read_text())
        expected = {"schema_version": 6,
            "predecessor_contract_sha256": hashes["generator_mask_delta_sha256"],
            "run_id": "cmirank-real-memory-smoke-v6-20261006-hnv",
            "change_only": "fresh_per_run_compiler_cache_namespace_after_workspace_relocation",
            "cache_names": ["TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"],
            "cache_namespace": "project_root/cache/runtime-hnv/run_id",
            "old_cache_artifacts": "preserve_do_not_rewrite_or_delete",
            "model_prompt_data_decoding_changes": False, "output_repair": False, "training_ready": False}
        if cache_delta != expected:
            raise ValueError("V6 changes only compiler cache namespace, not the scientific smoke")
        config["run_id"] = cache_delta["run_id"]
        config["compiler_cache_delta"] = cache_delta
        hashes["compiler_cache_delta_sha256"] = file_sha256(cache_path)
    if version == 7:
        control_path = root / "configs/cmirank/real_memory_smoke_v7.json"
        control_delta = json.loads(control_path.read_text())
        expected = {"schema_version": 7,
            "approval_status": "researcher_approved_secondary_decoding_control_2026-10-07",
            "predecessor_contract_sha256": hashes["compiler_cache_delta_sha256"],
            "offline_v6_review_sha256": "b1a64149975c09514d439dafafa36702d89c96312eaacbdb77af5ff2ad052d6b",
            "run_id": "cmirank-real-memory-constrained-smoke-v7-20261007-hnv",
            "change_only": "label_blind_input_id_enums_in_structured_decoding",
            "comparison_role": "secondary_control_not_primary_replacement",
            "stage_r_source_ids": "packed_neighbor_headers_plus_visible_candidates_plus_current_user",
            "stage_r_edge_targets": "current_user_only", "rerank_item_ids": "visible_candidate_ids_only",
            "stage_w_neighbor_ids": "listed_propagation_neighbors_only_empty_domain_max_items_zero",
            "schema_numeric_text_and_fanout_changes": False, "model_prompt_data_temperature_changes": False,
            "decoding_changes": True, "unknown_id_schema_provenance_stage_w_gates": "unchanged_hard_fail",
            "candidate_context_citations": "retain_raw_output_and_role_warning",
            "output_repair": False, "training_ready": False}
        if control_delta != expected:
            raise ValueError("V7 changes only ID decoding domains as a separately approved control")
        config["run_id"] = control_delta["run_id"]
        config["constrained_decoding_delta"] = control_delta
        hashes["constrained_decoding_delta_sha256"] = file_sha256(control_path)
    return config, object_sha256(hashes)


def validate_memory_contract(config: dict) -> None:
    baseline_agent = {"k": 16, "tau": 1800, "n_facets": 7, "mix_min_users": 4,
        "mix_min_items": 6, "fanout_cap": 8, "reranker_mode": "llm", "pruner_mode": "llm_rules",
        "upstream_empty_facets_prompt": True}
    fixed = {"schema_version": 1, "run_id": "cmirank-real-memory-smoke-v1-20261005-hnv",
        "scope": "20_user_real_memory_infrastructure_not_full_cache_or_training",
        "memory_provider": "upstream_aligned", "users": 20, "warmup_scope": "same_20_smoke_users_only",
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "model_revision": "5a5a776300a41aaa681dd7ff0106608ef2bc90db", "backend": "vllm==0.10.2",
        "tensor_parallel_size": 1, "gpu_memory_utilization": 0.60, "max_model_len": 16384,
        "max_num_seqs": 1, "dtype": "auto", "seed": 42, "temperature": 0.0,
        "max_tokens": 4000, "agent": baseline_agent, "nominal_physical_requests": 100,
        "physical_request_cap": 110, "sdk_max_retries": 0, "request_timeout_seconds": 180,
        "server_startup_timeout_seconds": 600, "timeout_minutes": 90,
        "output_repair": False, "response_cache_read": False, "training_ready": False}
    bindings = {"candidate_run_id", "audit_run_id", "candidate_source_commit", "full_report_sha256",
                "full_candidates_sha256", "audit_report_sha256", "audit_features_sha256"}
    if set(config) != set(fixed) | bindings or any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("Memory smoke must keep the baseline contract and bounded diagnostic scope")
    for name in bindings - {"candidate_run_id", "audit_run_id"}:
        length = 40 if name == "candidate_source_commit" else 64
        if not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", config[name]):
            raise ValueError("Missing locked predecessor artifact hash")


def object_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def memory_sha256(storage) -> str:
    return object_sha256({"users": storage.user_profiles, "items": storage.item_descriptions,
                          "n_updates": storage.n_updates})


def validate_scores(scores: list, candidates: list[int]) -> None:
    if (len(candidates) != 10 or len(set(candidates)) != 10 or len(scores) != 10
            or any(type(row.get("item_id")) is not int for row in scores)
            or {row["item_id"] for row in scores} != set(candidates)):
        raise ValueError("Ranker must return all ten original candidate IDs exactly once")
    for row in scores:
        score = row.get("score")
        if (isinstance(score, bool) or not isinstance(score, (float, int))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Ranker score must be finite and between zero and one")


def inspect_retrieval_grounding(details: dict, uid: int) -> dict:
    """Classify citations by INPUT role, without rewriting outputs or using gold.

    Candidate context is visible in the upstream Stage-R prompt but is not
    collaborative evidence. This distinction is a diagnostic, not a claim that
    citing candidates demonstrates user preference or a clean memory graph.
    """
    packed = details["packed_context"]["neighbors_text"]
    neighbors = set(re.findall(r"\[(User-\d+|Item-\d+)\]", packed))
    candidate_ids = [int(value) for value in re.findall(
        r"^\d+\. \[(\d+)\]", details["packed_context"].get("candidates_text", ""), re.MULTILINE)]
    if "rank_request" in details:
        request = RankRequest.from_dict(details["rank_request"])
        if candidate_ids != [row["id"] for row in request.candidates]:
            raise ValueError("Visible Stage-R candidate identities differ from captured RankRequest")
    candidates = {f"Item-{item}" for item in candidate_ids}
    bundle = details["retrieval_bundle"]
    citations = []
    def classify(identifier, origin, index):
        role = ("collaborative_neighbor" if identifier in neighbors else
                "personal_user" if identifier == f"User-{uid}" else
                "candidate_context" if identifier in candidates else "unknown_context")
        citations.append({"identifier": identifier, "origin": origin, "index": index, "input_role": role})
    for index, facet in enumerate(bundle["facets"]):
        for identifier in facet["supporting_neighbors"]:
            classify(identifier, "facet_support", index)
    for index, edge in enumerate(bundle["support_edges"]):
        classify(edge["from"], "support_edge_from", index)
    candidate_citations = [row for row in citations if row["input_role"] == "candidate_context"]
    unknown_citations = [row for row in citations if row["input_role"] == "unknown_context"]
    return {"facet_count": len(bundle["facets"]), "packed_neighbors": details["packed_context"]["n_neighbors"],
            "selected_neighbors": len(details["pruned_subgraph"]["neighbors"]),
            "visible_neighbor_ids": sorted(neighbors), "visible_candidate_ids": candidate_ids,
            "citations": citations, "candidate_context_citations": candidate_citations,
            "unknown_context_citations": unknown_citations,
            "collaborative_only_citations": not candidate_citations and not unknown_citations,
            "semantic_grounding_proven": False}


def validate_retrieval(details: dict, uid: int, *, candidate_context_warning: bool = False) -> dict:
    """V1–3 retain collab-only hard fail; approved v4 diagnoses visible candidates.

    Neither mode permits invented IDs, invalid values, malformed schema or wrong
    target-user edges. Neither mode certifies semantic entailment.
    """
    audit = inspect_retrieval_grounding(details, uid)
    bundle = details["retrieval_bundle"]
    if not bundle["facets"]:
        raise ValueError("Empty Stage-R facets cannot validate a real-memory smoke")
    for facet in bundle["facets"]:
        if (not facet["facet"].strip() or not 0 <= facet["confidence"] <= 1
                or audit["unknown_context_citations"]
                or (not candidate_context_warning and audit["candidate_context_citations"])):
            raise ValueError("Stage-R facet has invalid confidence or out-of-context support IDs")
    for edge in bundle["support_edges"]:
        if edge["to"] != f"User-{uid}" or not 0 <= edge["w"] <= 1:
            raise ValueError("Stage-R support edge is not grounded in its visible context")
    return audit


def validate_write(result: dict, mutation: dict, details: dict, uid: int,
                   clicked: int, fanout_cap: int) -> None:
    allowed = {f"{nb['type'].capitalize()}-{nb['id']}" for nb in details["pruned_subgraph"]["neighbors"]}
    raw = result["raw_response"]
    neighbor_ids = [row["neighbor_id"] for row in raw["neighbor_updates"]]
    # Upstream silently ignores an invented neighbor. Diagnose it, do not repair
    # the response or treat an ignored mutation as a valid output.
    if (set(neighbor_ids) - allowed or len(set(neighbor_ids)) != len(neighbor_ids)
            or len(neighbor_ids) > fanout_cap):
        raise ValueError("Stage-W neighbor IDs/fanout violate its existing prompt")
    users = {uid} | {int(n.split("-")[1]) for n in neighbor_ids if n.startswith("User-")}
    items = {clicked} | {int(n.split("-")[1]) for n in neighbor_ids if n.startswith("Item-")}
    if set(mutation["users"]) - users or set(mutation["items"]) - items:
        raise ValueError("Mutation journal contains an unrequested entity")
    if result["stats"]["user_applied"] != 1 or result["stats"]["item_applied"] != 1:
        raise ValueError("Stage-W did not create both personal and clicked-item memory")


def run_memory_smoke(agent, *, inputs: dict, rows: list[dict], user_ids: list[int], emit,
                     candidate_context_warning: bool = False) -> dict:
    """Serial, all-smoke-warmups first, then read-only pseudo ranking.

    `emit(name, row)` must durably append JSON before the next event. Labels stay
    in a separate reward-audit file, never in RankRequest or a policy prompt.
    """
    if len(user_ids) != 20 or len(set(user_ids)) != 20 or user_ids != sorted(user_ids):
        raise ValueError("Exactly twenty predeclared ordered smoke users required")
    if agent.storage.n_updates or agent.storage.user_profiles:
        raise ValueError("Must start from fresh metadata-only storage")
    by_event = {(row["provenance"]["user_id"], row["provenance"]["kind"]): row for row in rows}
    if len(by_event) != len(rows):
        raise ValueError("Duplicate episode")
    initial_hash = memory_sha256(agent.storage)
    requests = []
    warning_users = {"warmup": [], "pseudo": []}
    candidate_citations = 0
    counters = {"warmup": 0, "pseudo": 0}
    for kind in ("warmup", "pseudo"):
        for uid in user_ids:
            row = by_event[uid, kind]
            candidates = deepcopy(row["policy_input"]["candidate_ids"])
            target = inputs["targets"][uid]
            history = inputs["snapshot"].train_data[uid]
            positive = inputs["warmups" if kind == "warmup" else "targets"][uid]
            if (target in history or positive != row["reward_audit"]["positive_item_id"]
                    or candidates.count(positive) != 1
                    or (set(candidates) - {positive}) & set(history)
                    or kind == "warmup" and target in candidates):
                raise ValueError("Episode temporal/candidate roles differ")
            before = memory_sha256(agent.storage)
            if hasattr(agent.llm_client, "set_episode"):
                agent.llm_client.set_episode(row["policy_input"]["episode_id"])
            ranked, details = agent.rerank(
                uid, candidates, NEUTRAL_PSEUDO_INSTRUCTION, return_details=True,
                capture_rank_request=True, rank_snapshot_id=inputs["snapshot_sha256"])
            emit("retrieval-trace", {"episode_id": row["policy_input"]["episode_id"],
                                    "details": details, "ranked_ids": ranked})
            validate_scores(details["rerank_scores"], candidates)
            # Persist the role diagnostic even when a later hard gate fails.
            diagnostic = inspect_retrieval_grounding(details, uid)
            emit("grounding-audit", {"episode_id": row["policy_input"]["episode_id"], **diagnostic})
            grounding = validate_retrieval(details, uid, candidate_context_warning=candidate_context_warning)
            if grounding["candidate_context_citations"]:
                warning_users[kind].append(uid)
                candidate_citations += len(grounding["candidate_context_citations"])
            if memory_sha256(agent.storage) != before:
                raise ValueError("Stage-R/ReRank mutated frozen memory")
            request = RankRequest.from_dict(details["rank_request"])
            if ([c["id"] for c in request.candidates] != candidates
                    or request.instruction != NEUTRAL_PSEUDO_INSTRUCTION
                    or request.snapshot_id != inputs["snapshot_sha256"]
                    or request.vanilla_mode or not request.upstream_empty_facets_prompt):
                raise ValueError("RankRequest differs from approved target-blind input")
            # Compare to the actual physical API messages, not just two mutually
            # agreeing serializers. Fake clients may omit this infrastructure hook.
            if hasattr(agent.llm_client, "last_request"):
                expected = agent.reranker.build_rerank_prompt(**request.baseline_prompt_kwargs())
                if agent.llm_client.last_request["messages"] != expected:
                    raise ValueError("Captured RankRequest cannot replay exact API prompt")
            if kind == "warmup":
                agent.storage.begin_mutation_record()
                feedback = {"action": "CLICK", "item_id": inputs["warmups"][uid], "position": 0}
                result = agent.write(user_id=uid, feedback=feedback,
                                     recent_facets=details["facets"], pruned_subgraph=details["pruned_subgraph"])
                mutation = agent.storage.finish_mutation_record()
                emit("stage-w-journal", {"episode_id": row["policy_input"]["episode_id"],
                    "user_id": uid, "feedback": feedback, "memory_before_sha256": before,
                    "memory_after_sha256": memory_sha256(agent.storage),
                    "mutation": mutation, "result": result, "grounding": grounding})
                validate_write(result, mutation, details, uid, positive, agent.fanout_cap)
            else:
                emit("policy-inputs", {"episode_id": row["policy_input"]["episode_id"],
                    "rank_request": request.to_dict(), "memory_state_sha256": before,
                    "warmup_scope_user_ids": user_ids})
                emit("reward-audit", {"episode_id": row["policy_input"]["episode_id"],
                                     **row["reward_audit"]})
                requests.append(request.sha256())
            counters[kind] += 1
            emit("progress", {"phase": kind, "completed_users": counters[kind],
                              "logical_requests": 3 * counters["warmup"] + 2 * counters["pseudo"]})
    if (agent.n_stage_w_calls, agent.n_stage_r_calls, agent.n_stage_rr_calls) != (20, 40, 40):
        raise ValueError("Unexpected stage count or pseudo-target write")
    return {"users": 20, "warmup_stage_w_calls": 20, "pseudo_stage_w_calls": 0,
            "stage_r_calls": 40, "stage_rerank_calls": 40,
            "target_blind_rank_requests": len(requests), "rank_request_hashes": requests,
            "initial_memory_sha256": initial_hash, "final_memory_sha256": memory_sha256(agent.storage),
            "candidate_context_warning_policy": candidate_context_warning,
            "candidate_context_warning_user_ids": warning_users,
            "candidate_context_citation_occurrences": candidate_citations,
            "warmup_scope_user_ids": user_ids, "full_1797_memory_cache": False,
            "semantic_memory_grounding_proven": False, "training_ready": False}
