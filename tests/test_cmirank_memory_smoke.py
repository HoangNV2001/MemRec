"""Target-blind real-memory pipeline guards, exercised with NO API/GPU."""

from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.smoke_full_memrec_cpu import FakeJSONClient
from src.cmirank.memory_smoke import (
    inspect_retrieval_grounding, load_memory_contract, memory_sha256, run_memory_smoke,
    validate_memory_contract, validate_retrieval, validate_scores, validate_write,
)
from src.cmirank.snapshot import make_prefix_snapshot, snapshot_sha256
from src.memory.storage import MemoryStorage
from src.models.memrec_agent import MemRecAgent

ROOT = Path(__file__).resolve().parents[1]


def make_fixture(client_class=FakeJSONClient):
    users = list(range(20))
    histories = {uid: (100, 101, 102, 103, 104, 200 + uid, 300 + uid) for uid in users}
    metadata = {i: {"title": f"Book {i}", "description": "Static metadata"} for i in range(100, 510)}
    snapshot, warmups, targets = make_prefix_snapshot(histories, users, min_prefix_length=5, item_metadata=metadata)
    rows = []
    for uid in users:
        for kind, labels in (("warmup", warmups), ("pseudo", targets)):
            rows.append({"provenance": {"user_id": uid, "kind": kind},
                         "policy_input": {"episode_id": f"user-{uid}-{kind}",
                                          "candidate_ids": [labels[uid], *range(500, 509)]},
                         "reward_audit": {"positive_item_id": labels[uid]}})
    client = client_class()
    client.fail_fast = True
    agent = MemRecAgent(snapshot, client, k=16, tau=1800, n_facets=7, temperature=0.0,
                        max_tokens=4000, fanout_cap=8, reranker_mode="llm", pruner_mode="llm_rules",
                        upstream_empty_facets_prompt=True)
    inputs = {"snapshot": snapshot, "warmups": warmups, "targets": targets,
              "snapshot_sha256": snapshot_sha256(snapshot)}
    return users, rows, client, agent, inputs


def test_twenty_user_serial_smoke_has_exact_counts_target_blind_inputs_and_replay():
    users, rows, client, agent, inputs = make_fixture()
    emitted = defaultdict(list)
    summary = run_memory_smoke(agent, inputs=inputs, rows=rows, user_ids=users,
                               emit=lambda name, row: emitted[name].append(deepcopy(row)))
    assert client.total_physical_requests == 100
    assert (summary["stage_r_calls"], summary["stage_rerank_calls"], summary["warmup_stage_w_calls"]) == (40, 40, 20)
    assert summary["pseudo_stage_w_calls"] == 0
    assert summary["full_1797_memory_cache"] is summary["training_ready"] is False
    assert summary["semantic_memory_grounding_proven"] is False
    assert [x["phase"] for x in emitted["progress"]] == ["warmup"] * 20 + ["pseudo"] * 20
    assert len(emitted["policy-inputs"]) == len(emitted["reward-audit"]) == 20
    for row in emitted["policy-inputs"]:
        assert not {"positive_item_id", "positive_position", "negative_components"} & set(row["rank_request"])
    storage = MemoryStorage()
    storage.initialize_item_descriptions(inputs["snapshot"].item_metadata)
    for journal in emitted["stage-w-journal"]:
        assert memory_sha256(storage) == journal["memory_before_sha256"]
        storage.apply_mutation_record(journal["mutation"])
        assert memory_sha256(storage) == journal["memory_after_sha256"]
    assert memory_sha256(storage) == summary["final_memory_sha256"]


@pytest.mark.parametrize("failure", ["dirty_memory", "warmup_target", "wrong_positive", "duplicate_episode"])
def test_episode_guard_fails_closed(failure):
    users, rows, _, agent, inputs = make_fixture()
    if failure == "dirty_memory":
        agent.storage.update_user_memory(0, "Imported memory")
    elif failure == "warmup_target":
        rows[0]["policy_input"]["candidate_ids"][1] = inputs["targets"][0]
    elif failure == "wrong_positive":
        rows[0]["reward_audit"]["positive_item_id"] = 999
    else:
        rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        run_memory_smoke(agent, inputs=inputs, rows=rows, user_ids=users, emit=lambda *_: None)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_scores_not_repaired(bad):
    scores = [{"item_id": i, "score": 0.5} for i in range(10)]
    scores[0]["score"] = bad
    with pytest.raises(ValueError):
        validate_scores(scores, list(range(10)))


def test_duplicate_rank_ids_and_unsupported_memory_entities_are_rejected():
    with pytest.raises(ValueError):
        validate_scores([{"item_id": 0, "score": 0.5}] * 10, list(range(10)))
    details = {"packed_context": {"neighbors_text": "[Item-100] book", "n_neighbors": 1},
               "pruned_subgraph": {"neighbors": [{"type": "item", "id": 100}]},
               "retrieval_bundle": {"facets": [{"facet": "interest", "confidence": .8,
                                                "supporting_neighbors": ["Item-999"]}], "support_edges": []}}
    with pytest.raises(ValueError, match="out-of-context"):
        validate_retrieval(details, 0)
    result = {"raw_response": {"neighbor_updates": [{"neighbor_id": "User-999"}]}}
    with pytest.raises(ValueError, match="neighbor IDs"):
        validate_write(result, {"users": {}, "items": {}}, details, 0, 200, 8)


def test_baseline_model_and_agent_contract_cannot_be_relaxed():
    config = json.loads((ROOT / "configs/cmirank/real_memory_smoke_v1.json").read_text())
    validate_memory_contract(config)
    for key, bad in (("users", 200), ("training_ready", True), ("output_repair", True),
                     ("gpu_memory_utilization", .9), ("model_id", "Qwen/Qwen3.5-4B")):
        changed = deepcopy(config)
        changed[key] = bad
        with pytest.raises(ValueError):
            validate_memory_contract(changed)


def load_runner():
    spec = importlib.util.spec_from_file_location("memory_runner", ROOT / "scripts/cmirank/10_smoke_real_memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_request_journal_keeps_exact_messages_and_rejects_truncation(tmp_path, monkeypatch):
    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion
    runner = load_runner()
    config = json.loads((ROOT / "configs/cmirank/real_memory_smoke_v1.json").read_text())
    seen = []
    def fake_create(self, **kwargs):
        seen.append(deepcopy(kwargs))
        return ChatCompletion(id="fake", object="chat.completion", created=0, model=config["model_id"],
            choices=[{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": '{}'}}],
            usage={"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13})
    monkeypatch.setattr(Completions, "create", fake_create)
    monkeypatch.delenv("MEMREC_LLM_CACHE_DB", raising=False)
    client = runner.ObservedClient(tmp_path, config, "locked-test-contract", 18105)
    client.set_episode("smoke-event")
    messages = [{"role": "user", "content": "Baseline prompt byte-for-byte"}]
    with pytest.raises(ValueError, match="Truncated"):
        client.generate_json(messages, {"scores": {"type": "array"}}, temperature=0., max_tokens=4000)
    journal = [json.loads(x) for x in (tmp_path / "physical-requests.jsonl").read_text().splitlines()]
    assert client.request_budget.used == client.total_physical_requests == 1
    assert seen[0]["messages"] == messages == journal[0]["kwargs"]["messages"]
    assert journal[1]["finish_reason"] == "length"
    assert journal[0]["request_sha256"] == journal[1]["request_sha256"]
    client.client.close()
    client.request_budget.connection.close()


def test_audit_hash_mismatch_blocks_before_model_loading(tmp_path):
    runner = load_runner()
    config = json.loads((ROOT / "configs/cmirank/real_memory_smoke_v1.json").read_text())
    with pytest.raises(ValueError, match="artifacts differ"):
        runner.verify_audit(tmp_path, config, {}, {})


def test_v2_binding_delta_preserves_entire_memory_experiment():
    base, base_sha = load_memory_contract(ROOT, 1)
    effective, sha = load_memory_contract(ROOT, 2)
    delta = effective.pop("device_binding_delta")
    assert delta["predecessor_contract_sha256"] == base_sha
    effective["run_id"] = base["run_id"]
    assert effective == base
    assert sha != base_sha
    with pytest.raises(ValueError):
        load_memory_contract(ROOT, 7)


def test_numeric_binding_requires_inventory_uuid_and_pci_order_agreement():
    from src.cmirank.gpu_resources import GPUCard, verified_numeric_cuda_binding
    cards = [GPUCard(i, f"GPU-{i}", "H100", 0, 1, 81559) for i in range(4)]
    pci = "\n".join(f"{i}, GPU-{i}, 00000000:{40+i:02X}:00.0" for i in range(4))
    assert verified_numeric_cuda_binding(cards, cards[-1], pci) == "3"
    for bad in (pci.replace("GPU-3", "GPU-9"), pci.replace("2B:00", "01:00"),
                pci.splitlines()[0], pci + "\n" + pci.splitlines()[0]):
        with pytest.raises(ValueError):
            verified_numeric_cuda_binding(cards, cards[-1], bad)


def test_v3_preserves_every_v2_field_except_identity_and_run_id():
    v2, sha2 = load_memory_contract(ROOT, 2)
    v3, sha3 = load_memory_contract(ROOT, 3)
    delta = v3.pop("device_identity_delta")
    assert delta["change_only"] == "canonical_uuid_representation_for_device_identity_assertion"
    v3["run_id"] = v2["run_id"]
    assert v3 == v2 and sha3 != sha2


def test_cuda_and_nvml_uuid_representations_match_full_identity_only():
    from src.cmirank.gpu_resources import canonical_gpu_uuid
    value = "116eac5f-3ac9-a010-88b2-33f589cbd713"
    assert canonical_gpu_uuid(value) == canonical_gpu_uuid("GPU-" + value.upper())
    assert canonical_gpu_uuid(value) != canonical_gpu_uuid("216eac5f-3ac9-a010-88b2-33f589cbd713")
    for malformed in (value[:-1], " " + value, value + " extra", "MIG-" + value, "GPU-GPU-" + value, None):
        with pytest.raises(ValueError):
            canonical_gpu_uuid(malformed)


def grounding_fixture(identifier="Item-200"):
    return {"packed_context": {"neighbors_text": "[Item-100] book", "n_neighbors": 1,
                               "candidates_text": "**Candidates to Rank:**\n1. [200] book\n2. [201] other book"},
            "pruned_subgraph": {"neighbors": [{"type": "item", "id": 100}]},
            "retrieval_bundle": {"facets": [{"facet": "interest", "confidence": .8,
                                             "supporting_neighbors": ["Item-100", identifier]}],
                                 "support_edges": [{"from": identifier, "to": "User-0", "w": .8}]}}


def test_visible_candidate_role_warning_does_not_repair_or_relax_historical_gate():
    details = grounding_fixture()
    original = deepcopy(details)
    audit = inspect_retrieval_grounding(details, 0)
    assert len(audit["candidate_context_citations"]) == 2 and not audit["unknown_context_citations"]
    assert audit["collaborative_only_citations"] is audit["semantic_grounding_proven"] is False
    with pytest.raises(ValueError):
        validate_retrieval(details, 0)  # v1–3 must remain rejected.
    assert validate_retrieval(details, 0, candidate_context_warning=True) == audit
    assert details == original


@pytest.mark.parametrize("failure", ["invented_id", "unpacked_neighbor", "wrong_user", "confidence", "edge_weight"])
def test_role_warning_never_permits_unknown_ids_or_invalid_values(failure):
    details = grounding_fixture()
    if failure == "invented_id":
        details["retrieval_bundle"]["facets"][0]["supporting_neighbors"].append("Item-999")
    elif failure == "unpacked_neighbor":
        details["pruned_subgraph"]["neighbors"].append({"type": "item", "id": 999})
        details["retrieval_bundle"]["support_edges"][0]["from"] = "Item-999"
    elif failure == "wrong_user":
        details["retrieval_bundle"]["support_edges"][0]["to"] = "User-1"
    elif failure == "confidence":
        details["retrieval_bundle"]["facets"][0]["confidence"] = 1.1
    else:
        details["retrieval_bundle"]["support_edges"][0]["w"] = -0.1
    with pytest.raises(ValueError):
        validate_retrieval(details, 0, candidate_context_warning=True)


def test_v4_keeps_all_model_data_and_device_fields_and_no_promotion():
    v3, hash3 = load_memory_contract(ROOT, 3)
    v4, hash4 = load_memory_contract(ROOT, 4)
    delta = v4.pop("evidence_role_delta")
    assert delta["output_repair"] is delta["training_ready"] is False
    assert delta["unknown_context_ids"] == "hard_fail"
    v4["run_id"] = v3["run_id"]
    assert v4 == v3 and hash4 != hash3


def test_v5_resource_retry_preserves_every_v4_scientific_and_warning_field():
    v4, hash4 = load_memory_contract(ROOT, 4)
    v5, hash5 = load_memory_contract(ROOT, 5)
    delta = v5.pop("generator_mask_delta")
    assert delta["wrapper_initial_cuda_masks"] == ["1", "0,1"] and delta["python_cuda_mask"] == "1"
    assert not delta["model_prompt_data_decoding_changes"] and not delta["training_ready"]
    v5["run_id"] = v4["run_id"]
    assert v5 == v4 and hash5 != hash4


def test_v6_preserves_science_and_uses_only_fresh_compiler_namespace():
    v5, hash5 = load_memory_contract(ROOT, 5)
    v6, hash6 = load_memory_contract(ROOT, 6)
    delta = v6.pop("compiler_cache_delta")
    assert delta["old_cache_artifacts"] == "preserve_do_not_rewrite_or_delete"
    assert delta["cache_names"] == ["TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"]
    v6["run_id"] = v5["run_id"]
    assert v6 == v5 and hash6 != hash5


def test_candidate_role_classification_is_independent_of_positive_labels():
    details = grounding_fixture()
    before = inspect_retrieval_grounding(details, 0)
    details["reward_audit"] = {"positive_item_id": 200, "positive_position": 0}
    assert inspect_retrieval_grounding(details, 0) == before
    details["reward_audit"] = {"positive_item_id": 201, "positive_position": 1}
    assert inspect_retrieval_grounding(details, 0) == before


def test_visible_candidates_must_agree_with_actual_rankrequest_order():
    from src.cmirank.request import RankRequest
    details = grounding_fixture()
    details["rank_request"] = RankRequest.from_stage_rr_inputs(user_id=0,
        retrieval_bundle=details["retrieval_bundle"], candidates=[{"id": 201}, {"id": 200}]).to_dict()
    with pytest.raises(ValueError, match="identities differ"):
        validate_retrieval(details, 0, candidate_context_warning=True)


def test_v4_full_fake_pipeline_records_every_warning_with_zero_output_repair():
    import re
    class CandidateCitingFake(FakeJSONClient):
        def generate_json(self, messages, properties, **kwargs):
            result = super().generate_json(messages, properties, **kwargs)
            if "facets" in properties:
                first = re.findall(r"^1\. \[(\d+)\]", messages[0]["content"], re.MULTILINE)[0]
                result["facets"][0]["supporting_neighbors"] = [f"Item-{first}"]
            return result
    users, rows, client, agent, inputs = make_fixture(CandidateCitingFake)
    recorded = defaultdict(list)
    result = run_memory_smoke(agent, inputs=inputs, rows=rows, user_ids=users,
        candidate_context_warning=True, emit=lambda name, row: recorded[name].append(deepcopy(row)))
    assert client.total_physical_requests == 100
    assert result["candidate_context_citation_occurrences"] == 40
    assert result["candidate_context_warning_user_ids"] == {"warmup": users, "pseudo": users}
    assert len(recorded["grounding-audit"]) == 40
    assert result["semantic_memory_grounding_proven"] is result["training_ready"] is False
    for trace in recorded["retrieval-trace"]:
        first = trace["details"]["rank_request"]["candidates"][0]["id"]
        assert trace["details"]["facets"][0]["supporting_neighbors"] == [f"Item-{first}"]
