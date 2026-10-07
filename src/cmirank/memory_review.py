"""Read-only review of partial real-memory smoke, without API, GPU or metrics.

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

    def __init__(self, physical: dict, raw: dict):
        self.physical, self.raw, self.calls = physical, raw, 0

    def generate_json(self, *, messages, properties, temperature, max_tokens, **kwargs):
        expected = {**self.physical, "messages": messages,
                    "response_format": response_schema(properties),
                    "temperature": temperature, "max_tokens": max_tokens}
        if self.calls or expected != self.physical:
            raise ValueError("Stage-W replay differs from the actual API prompt/schema")
        self.calls += 1
        return deepcopy(self.raw)


def review_partial_records(run: Path, *, candidates: dict, config: dict,
                           manifest: dict, storage, metadata: dict) -> dict:
    """Check serial complete warm-ups followed by a failed Stage-R/ReRank pair.

    This deliberately rejects interrupted/truncated or other failure layouts;
    such runs need a separately reviewed diagnostic, not guessed accounting.
    """
    physical = read_rows(run / "physical-requests.jsonl")
    traces = read_rows(run / "retrieval-trace.jsonl")
    audits = read_rows(run / "grounding-audit.jsonl")
    writes = read_rows(run / "stage-w-journal.jsonl")
    progress = read_rows(run / "progress.jsonl")
    if (not traces or len(audits) != len(traces) or len(writes) != len(traces) - 1
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
                      if kwargs["response_format"] == response_schema(props)), None)
        if (request["event"] != "request" or response["event"] != "response" or stage is None
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
        details, episode = trace["details"], trace["episode_id"]
        source = candidates[episode]
        request = RankRequest.from_dict(details["rank_request"])
        uid = request.user_id
        ids = source["policy_input"]["candidate_ids"]
        if (uid != manifest["query_user_ids"][index] or source["provenance"]["kind"] != "warmup"
                or source["provenance"]["user_id"] != uid
                or source["provenance"]["snapshot_sha256"] != manifest["graph_snapshot_sha256"]
                or request.snapshot_id != manifest["graph_snapshot_sha256"]
                or request.instruction != NEUTRAL_PSEUDO_INSTRUCTION or request.vanilla_mode
                or not request.upstream_empty_facets_prompt
                or [row["id"] for row in request.candidates] != ids
                or request.sha256() != details["rank_request_sha256"]):
            raise ValueError("Warm-up trace differs from the locked target-blind input")
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
            if index != len(traces) - 1 or not grounding["unknown_context_citations"]:
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
        if index == len(traces) - 1:
            raise ValueError("Failed run unexpectedly passes its terminal hard gate")
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
        replay_client = RecordedWriteClient(w_request["kwargs"], w_raw)
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
    if failed_grounding is None or cursor != len(pairs):
        raise ValueError("No terminal grounding failure or unaccounted calls")
    return {"status": "FAILED_MEMORY_SMOKE_OFFLINE_REVIEW_COMPLETE_NOT_PROMOTED",
        "physical_requests": len(pairs), "physical_responses": len(pairs), "physical_retries": 0,
        **tokens, "total_tokens": sum(tokens.values()), "stage_r_calls": len(traces),
        "stage_rerank_calls": len(traces), "stage_w_calls": len(writes),
        "completed_warmup_users": len(writes), "completed_pseudo_users": 0,
        "citation_occurrences": citations, "candidate_context_warning_occurrences": warnings,
        "candidate_context_warning_episodes": warning_episodes, "unknown_context_occurrences": unknowns,
        "failure": failed_grounding, "initial_memory_sha256": initial_hash,
        "replayed_prefix_memory_sha256": memory_sha256(storage),
        "exact_stage_r_rerank_write_api_prompt_parity": True, "raw_write_mutation_replay_valid": True,
        "literal_grounding_gate_passed": False, "semantic_memory_grounding_proven": False,
        "full_memory_cache_promoted": False, "training_ready": False, "ranking_metrics_computed": False}
