"""Read-only review of partial/complete real-memory smoke; no API/GPU/metrics.

Replay the unchanged Stage-W implementation against its recorded responses.
Unknown citations remain failures; this reviewer never repairs or promotes them.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

from .memory_smoke import (
    inspect_retrieval_grounding, memory_sha256, object_sha256, validate_retrieval,
    validate_scores, validate_write,
)
from .policy_data import NEUTRAL_PSEUDO_INSTRUCTION
from .provenance import file_sha256
from .request import RankRequest
from src.memory.manager import MemRecManager
from src.models.llm_client import validate_json_shape
from src.models.memrec_agent import MemRecAgent
from src.models.reranker_llm import LLMReranker


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def response_schema(properties: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "response", "strict": True,
        "schema": {"type": "object", "properties": properties,
                   "required": list(properties), "additionalProperties": False}}}


class RecordedWriteClient:
    """One recorded response, no network client or fallback generation."""

    fail_fast = True

    def __init__(self, physical: dict, raw: dict, *, id_control: bool = False):
        self.physical, self.raw, self.calls = physical, raw, 0
        self.id_control = id_control

    def generate_json(self, *, messages, properties, temperature, max_tokens, **kwargs):
        if self.id_control:
            from .constrained_decoding import constrain_id_properties
            properties, _ = constrain_id_properties(messages, properties)
        expected = {**self.physical, "messages": messages,
                    "response_format": response_schema(properties),
                    "temperature": temperature, "max_tokens": max_tokens}
        if self.calls or expected != self.physical:
            raise ValueError("Stage-W replay differs from the actual API prompt/schema")
        self.calls += 1
        return deepcopy(self.raw)


def review_partial_records(run: Path, *, candidates: dict, config: dict,
                           manifest: dict, storage, metadata: dict,
                           completed_control: bool = False) -> dict:
    """Review a failed prefix, or the explicit complete secondary v7 control.

    This deliberately rejects interrupted/truncated or other failure layouts;
    such runs need a separately reviewed diagnostic, not guessed accounting.
    """
    physical = read_rows(run / "physical-requests.jsonl")
    traces = read_rows(run / "retrieval-trace.jsonl")
    audits = read_rows(run / "grounding-audit.jsonl")
    writes = read_rows(run / "stage-w-journal.jsonl")
    progress = read_rows(run / "progress.jsonl")
    users = manifest["query_user_ids"]
    policy, rewards, domains = [], [], []
    if completed_control:
        if "constrained_decoding_delta" not in config or len(users) != 20 or len(set(users)) != 20:
            raise ValueError("Only the explicit twenty-user secondary control can use complete mode")
        policy, rewards = read_rows(run / "policy-inputs.jsonl"), read_rows(run / "reward-audit.jsonl")
        domains = read_rows(run / "decoding-contracts.jsonl")
        if ((len(traces), len(audits), len(writes), len(progress), len(physical), len(policy), len(rewards), len(domains))
                != (40, 40, 20, 40, 200, 20, 20, 100) or (run / "failure.json").exists()):
            raise ValueError("Not a complete twenty-user warm-up then read-only pseudo control")
    elif (not traces or len(audits) != len(traces) or len(writes) != len(traces) - 1
          or len(progress) != len(writes) or len(physical) != 2 * (3 * len(writes) + 2)
          or any((run / name).exists() for name in ("policy-inputs.jsonl", "reward-audit.jsonl", "report.json"))):
        raise ValueError("Not a serial warm-up prefix ending at the retrieval gate")
    manager, reranker = MemRecManager(None), LLMReranker(None)
    properties = {"stage_r": manager.get_stage_r_schema(),
                  "rerank": reranker.get_rerank_schema(), "stage_w": manager.get_stage_w_schema()}
    pairs, tokens = [], {"input_tokens": 0, "output_tokens": 0}
    for offset in range(0, len(physical), 2):
        request, response = physical[offset:offset + 2]
        attempt = offset // 2 + 1
        kwargs = request["kwargs"]
        stage = next((name for name, props in properties.items()
                      if set(kwargs["response_format"]["json_schema"]["schema"]["properties"]) == set(props)), None)
        expected_properties = properties.get(stage, {})
        if completed_control:
            from .constrained_decoding import constrain_id_properties, validate_constrained_output
            expected_properties, domain = constrain_id_properties(kwargs["messages"], expected_properties)
            expected_domain = {"episode_id": request["episode_id"],
                "messages_sha256": object_sha256(kwargs["messages"]),
                "original_properties_sha256": object_sha256(properties[stage]),
                "constrained_properties_sha256": object_sha256(expected_properties), **domain}
            if domains[offset // 2] != expected_domain:
                raise ValueError("Decoding journal differs from label-blind API input domains")
        if (request["event"] != "request" or response["event"] != "response" or stage is None
                or kwargs["response_format"] != response_schema(expected_properties)
                or set(kwargs) != {"model", "temperature", "max_tokens", "messages", "response_format"}
                or kwargs["model"] != config["model_id"] or kwargs["temperature"] != config["temperature"]
                or kwargs["max_tokens"] != config["max_tokens"]
                or request["request_sha256"] != object_sha256(kwargs)
                or any(request[key] != response[key] for key in
                       ("request_sha256", "episode_id", "physical_attempt"))
                or request["physical_attempt"] != attempt or response["finish_reason"] != "stop"):
            raise ValueError("Physical request/response order, schema or contract differs")
        raw = json.loads(response["content"])
        validate_json_shape(raw, kwargs["response_format"]["json_schema"]["schema"])
        if completed_control:
            validate_constrained_output(raw, kwargs["response_format"]["json_schema"]["schema"])
        usage = response["usage"]
        if (any(type(usage[key]) is not int or usage[key] < 0 for key in
                ("prompt_tokens", "completion_tokens", "total_tokens"))
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError("Token accounting differs")
        tokens["input_tokens"] += usage["prompt_tokens"]
        tokens["output_tokens"] += usage["completion_tokens"]
        pairs.append((stage, request, raw))
    with sqlite3.connect(f"file:{run / 'request-budget.sqlite'}?mode=ro", uri=True) as connection:
        budgets = connection.execute("SELECT lim, used FROM budget").fetchall()
    if budgets != [(config["physical_request_cap"], len(pairs))]:
        raise ValueError("Durable physical budget differs from the journal")
    initial_hash = memory_sha256(storage)
    warnings = unknowns = citations = 0
    warning_episodes, failed_grounding, cursor = [], None, 0
    for index, trace in enumerate(traces):
        kind = "pseudo" if completed_control and index >= len(users) else "warmup"
        phase_index = index - len(users) if kind == "pseudo" else index
        details, episode = trace["details"], trace["episode_id"]
        source = candidates[episode]
        request = RankRequest.from_dict(details["rank_request"])
        uid = request.user_id
        ids = source["policy_input"]["candidate_ids"]
        if (uid != users[phase_index] or source["provenance"]["kind"] != kind
                or source["provenance"]["user_id"] != uid
                or source["provenance"]["snapshot_sha256"] != manifest["graph_snapshot_sha256"]
                or request.snapshot_id != manifest["graph_snapshot_sha256"]
                or request.instruction != NEUTRAL_PSEUDO_INSTRUCTION or request.vanilla_mode
                or not request.upstream_empty_facets_prompt
                or [row["id"] for row in request.candidates] != ids
                or request.sha256() != details["rank_request_sha256"]):
            raise ValueError("Episode trace differs from the locked target-blind input")
        if completed_control and any(memory != storage.get_item_memory(item)
                                     for item, memory in request.item_memories.items()):
            raise ValueError("Captured item memories differ from replayed Stage-W state")
        r_stage, r_request, r_raw = pairs[cursor]
        rr_stage, rr_request, rr_raw = pairs[cursor + 1]
        packed = details["packed_context"]
        expected_r = manager.build_stage_r_prompt(uid, packed["memory_text"],
            packed["neighbors_text"], packed["candidates_text"], config["agent"]["n_facets"])
        if (r_stage != "stage_r" or rr_stage != "rerank"
                or r_request["episode_id"] != episode or rr_request["episode_id"] != episode
                or r_request["kwargs"]["messages"] != expected_r
                or rr_request["kwargs"]["messages"] != reranker.build_rerank_prompt(**request.baseline_prompt_kwargs())
                or r_raw != details["retrieval_bundle"] or rr_raw["scores"] != details["rerank_scores"]
                or details["facets"] != r_raw["facets"] or details["support_edges"] != r_raw["support_edges"]):
            raise ValueError("Captured Stage-R/ReRank output or exact API prompt differs")
        validate_scores(details["rerank_scores"], ids)
        if trace["ranked_ids"] != [row["item_id"] for row in sorted(rr_raw["scores"], key=lambda row: row["score"], reverse=True)]:
            raise ValueError("Recorded order differs from the unmodified ranker scores")
        grounding = inspect_retrieval_grounding(details, uid)
        if audits[index] != {"episode_id": episode, **grounding}:
            raise ValueError("Grounding journal differs from the raw input roles")
        warnings += len(grounding["candidate_context_citations"])
        unknowns += len(grounding["unknown_context_citations"])
        citations += len(grounding["citations"])
        if grounding["candidate_context_citations"]:
            warning_episodes.append(episode)
        if (not r_raw["facets"]
                or any(not facet["facet"].strip() or not 0 <= facet["confidence"] <= 1
                       for facet in r_raw["facets"])
                or any(edge["to"] != f"User-{uid}" or not 0 <= edge["w"] <= 1
                       for edge in r_raw["support_edges"])):
            raise ValueError("Unexpected non-ID retrieval error")
        try:
            validate_retrieval(details, uid, candidate_context_warning=True)
        except ValueError as error:
            if completed_control or index != len(traces) - 1 or not grounding["unknown_context_citations"]:
                raise ValueError("Unexpected earlier/non-ID retrieval failure") from error
            original_prompt = json.dumps(r_request["kwargs"]["messages"], ensure_ascii=False)
            failed_grounding = {"episode_id": episode, "error": str(error), "grounding": grounding,
                "unknown_id_input_presence": [{"identifier": row["identifier"],
                    "literal_in_original_stage_r_input": row["identifier"] in original_prompt,
                    "numeric_token_in_original_stage_r_input": bool(re.search(
                        r"(?<!\d)" + re.escape(row["identifier"].split("-")[-1]) + r"(?!\d)", original_prompt))}
                    for row in grounding["unknown_context_citations"]]}
            cursor += 2
            continue
        if not completed_control and index == len(traces) - 1:
            raise ValueError("Failed run unexpectedly passes its terminal hard gate")
        if kind == "pseudo":
            expected_policy = {"episode_id": episode, "rank_request": request.to_dict(),
                               "memory_state_sha256": memory_sha256(storage), "warmup_scope_user_ids": users}
            if (policy[phase_index] != expected_policy
                    or rewards[phase_index] != {"episode_id": episode, **source["reward_audit"]}):
                raise ValueError("Read-only pseudo input/state or separate reward audit differs")
            p = progress[index]
            if (p["phase"] != "pseudo" or p["completed_users"] != phase_index + 1
                    or p["logical_requests"] != 60 + 2 * (phase_index + 1)):
                raise ValueError("Pseudo progress differs from the read-only episodes")
            cursor += 2
            continue
        journal = writes[index]
        w_stage, w_request, w_raw = pairs[cursor + 2]
        expected_feedback = {"action": "CLICK", "item_id": source["reward_audit"]["positive_item_id"], "position": 0}
        if (w_stage != "stage_w" or w_request["episode_id"] != episode or journal["episode_id"] != episode
                or journal["user_id"] != uid or journal["feedback"] != expected_feedback
                or journal["result"]["raw_response"] != w_raw or journal["grounding"] != grounding
                or memory_sha256(storage) != journal["memory_before_sha256"]):
            raise ValueError("Stage-W feedback, raw output or memory chain differs")
        typed_mutation = {**journal["mutation"],
                          "users": {int(key): value for key, value in journal["mutation"]["users"].items()},
                          "items": {int(key): value for key, value in journal["mutation"]["items"].items()}}
        validate_write(journal["result"], typed_mutation, details, uid,
                       expected_feedback["item_id"], config["agent"]["fanout_cap"])
        replay_client = RecordedWriteClient(w_request["kwargs"], w_raw, id_control=completed_control)
        actor = SimpleNamespace(storage=storage, dataset=SimpleNamespace(item_metadata=metadata),
            manager=MemRecManager(replay_client), fanout_cap=config["agent"]["fanout_cap"],
            temperature=config["temperature"], max_tokens=config["max_tokens"], n_stage_w_calls=0)
        storage.begin_mutation_record()
        result = MemRecAgent.write(actor, uid, journal["feedback"], details["facets"], details["pruned_subgraph"])
        mutation = storage.finish_mutation_record()
        mutation = json.loads(json.dumps(mutation))  # JSON journals stringify integer entity keys.
        if (result != journal["result"] or mutation != journal["mutation"] or replay_client.calls != 1
                or memory_sha256(storage) != journal["memory_after_sha256"]):
            raise ValueError("Raw Stage-W replay/mutation/state hash differs")
        p = progress[index]
        if (p["phase"] != "warmup" or p["completed_users"] != index + 1
                or p["logical_requests"] != 3 * (index + 1)):
            raise ValueError("Progress journal differs from the completed warm-ups")
        cursor += 3
    if (not completed_control and failed_grounding is None) or cursor != len(pairs):
        raise ValueError("No terminal grounding failure or unaccounted calls")
    if completed_control:
        report = json.loads((run / "report.json").read_text())
        expected = {"initial_memory_sha256": initial_hash, "final_memory_sha256": memory_sha256(storage),
            "users": 20, "warmup_stage_w_calls": 20, "pseudo_stage_w_calls": 0, "stage_r_calls": 40,
            "stage_rerank_calls": 40, "target_blind_rank_requests": 20, "controlled_decoding_logical_requests": 100,
            "candidate_context_citation_occurrences": warnings, "primary_provider_replaced": False,
            "comparison_role": "secondary_control_not_primary_replacement", "training_ready": False,
            "full_1797_memory_cache": False, "semantic_memory_grounding_proven": False,
            "rank_request_hashes": [RankRequest.from_dict(row["rank_request"]).sha256() for row in policy]}
        stats = report["token_stats"]
        if (any(report.get(key) != value for key, value in expected.items())
                or report["artifact_sha256"] != {p.name: file_sha256(p)
                                                for p in run.glob("*.jsonl")}
                or stats["total_requests"] != 100 or stats["total_physical_requests"] != 100
                or stats["total_cache_hits"] != 0 or stats["total_input_tokens"] != tokens["input_tokens"]
                or stats["total_output_tokens"] != tokens["output_tokens"]
                or stats["total_tokens"] != sum(tokens.values())):
            raise ValueError("Completed report/hash/token claims differ from the independent replay")
    return {"status": ("SECONDARY_CONTROL_REVIEW_PASS_LITERAL_VALIDITY_NOT_SEMANTIC_GROUNDING"
                       if completed_control else "FAILED_MEMORY_SMOKE_OFFLINE_REVIEW_COMPLETE_NOT_PROMOTED"),
        "physical_requests": len(pairs), "physical_responses": len(pairs), "physical_retries": 0,
        **tokens, "total_tokens": sum(tokens.values()), "stage_r_calls": len(traces),
        "stage_rerank_calls": len(traces), "stage_w_calls": len(writes),
        "completed_warmup_users": len(writes), "completed_pseudo_users": len(policy),
        "citation_occurrences": citations, "candidate_context_warning_occurrences": warnings,
        "candidate_context_warning_episodes": warning_episodes, "unknown_context_occurrences": unknowns,
        "failure": failed_grounding, "initial_memory_sha256": initial_hash,
        "replayed_prefix_memory_sha256": memory_sha256(storage),
        "exact_stage_r_rerank_write_api_prompt_parity": True, "raw_write_mutation_replay_valid": True,
        "literal_grounding_gate_passed": completed_control, "semantic_memory_grounding_proven": False,
        "primary_provider_replaced": False, "pseudo_memory_read_only_verified": completed_control,
        "full_memory_cache_promoted": False, "training_ready": False, "ranking_metrics_computed": False}
