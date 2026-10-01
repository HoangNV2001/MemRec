"""CPU-only G0 checks for CM-IRank; no model or GPU required."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cmirank.iterative_env import rank_iteratively
from src.cmirank.labels import label_to_item, make_labels
from src.cmirank.parser import parse_action
from src.cmirank.policy_data import draft_policy_split, pseudo_target_from_train
from src.cmirank.prompts import render_step_prompt
from src.cmirank.request import RankRequest
from src.cmirank.snapshot import make_prefix_snapshot, snapshot_sha256
from src.cmirank.rewards import (
    mpss_rewards, ndcg_at_k, r1_binary_rewards, terminal_ndcg_rewards,
)
from src.memory.graph import UserItemGraph
from src.models.memrec_agent import MemRecAgent
from src.models.reranker_llm import LLMReranker


def test_approved_model_contract_is_pinned_but_not_train_ready():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/cmirank/policy_model_v1.json").read_text())
    assert config["model_id"] == "Qwen/Qwen3.5-4B"
    assert len(config["model_revision"]) == 40
    assert config["primary_algorithm"] == "ppo"
    assert config["training_ready"] is False


def sample_request() -> RankRequest:
    return RankRequest.from_stage_rr_inputs(
        user_id=7,
        instruction="Find a suitable book",
        retrieval_bundle={"facets": [
            {"facet": "likes science", "confidence": 0.8,
             "supporting_neighbors": ["U-2"]},
        ]},
        candidates=[{"id": i + 100, "title": f"Book {i}", "tags": []}
                    for i in range(10)],
        item_mems={100: "Useful science book"},
        upstream_empty_facets_prompt=True,
        snapshot_id="fake-clean-snapshot",
    )


def test_request_roundtrip_and_exact_baseline_prompt():
    request = sample_request()
    replayed = RankRequest.from_dict(request.to_dict())
    assert replayed.sha256() == request.sha256()
    assert replayed.baseline_prompt_kwargs() == request.baseline_prompt_kwargs()
    reranker = LLMReranker(None)
    original = reranker.build_rerank_prompt(
        user_id=request.user_id, facets=list(request.collaborative_facets),
        candidates=list(request.candidates), item_mems=request.item_memories,
        instruction=request.instruction,
        upstream_empty_facets_prompt=True,
    )
    assert original == reranker.build_rerank_prompt(**replayed.baseline_prompt_kwargs())
    changed_snapshot = RankRequest.from_stage_rr_inputs(
        user_id=request.user_id, retrieval_bundle={"facets": request.collaborative_facets},
        candidates=request.candidates, item_mems=request.item_memories,
        instruction=request.instruction, upstream_empty_facets_prompt=True,
        snapshot_id="replay-runtime-id",
    )
    assert changed_snapshot.sha256() == request.sha256()


def test_request_rejects_reward_fields_and_tampering():
    request = sample_request()
    row = request.to_dict()
    row["target_item_id"] = 100
    with pytest.raises(ValueError, match="Unexpected"):
        RankRequest.from_dict(row)
    row = request.to_dict()
    row["collaborative_facets"][0]["facet"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        RankRequest.from_dict(row)
    prompt = render_step_prompt(request, make_labels(range(10))).lower()
    assert "target_item_id" not in prompt and "ground_truth" not in prompt


def test_labels_and_strict_parser():
    labels = make_labels(list(range(10)))
    assert label_to_item(list(range(10)))["C09"] == 9
    assert parse_action("<reasoning>C02 maybe</reasoning><answer> c07 </answer>", labels).label == "C07"
    for raw in ("C07", "<answer>C07 C08</answer>",
                "<answer>C07</answer><answer>C08</answer>",
                "<answer>C99</answer>", "<answer>C07</answer>"):
        active = labels[:-3] if raw == "<answer>C07</answer>" else labels
        assert not parse_action(raw, active).valid


@pytest.mark.parametrize("rank", range(1, 11))
def test_mpss_matches_exact_ndcg_for_every_target_rank(rank):
    labels = make_labels(list(range(10)))
    target = labels[0]
    ranking = [label for label in labels if label != target]
    ranking.insert(rank - 1, target)
    exclusions = list(reversed(ranking[1:]))
    survivor = ranking[0]
    shaped, observed_rank = mpss_rewards(labels, exclusions, survivor, target)
    terminal = terminal_ndcg_rewards(labels, exclusions, survivor, target)
    assert observed_rank == rank
    assert len(shaped) == 9
    assert abs(sum(shaped) - ndcg_at_k(rank)) < 1e-12
    assert abs(sum(terminal) - sum(shaped)) < 1e-12
    assert r1_binary_rewards(labels, exclusions, survivor, target).count(0.0) == (rank != 1)


def test_mpss_rejects_invalid_trajectory():
    labels = make_labels(list(range(4)))
    with pytest.raises(ValueError):
        mpss_rewards(labels, ["C01", "C01", "C02"], "C00", "C00")


def test_iterative_fake_oracle_anti_oracle_and_invalid_step():
    request = sample_request()
    target_id = 100
    mapping = label_to_item([row["id"] for row in request.candidates])

    def oracle(prompt: str) -> str:
        active = [label for label in mapping if f"[{label}]" in prompt]
        choice = next(label for label in reversed(active) if mapping[label] != target_id)
        return f"<answer>{choice}</answer>"

    best = rank_iteratively(request, oracle)
    assert best.valid and best.ranked_candidate_ids[0] == target_id
    assert len(best.trace) == 9

    bad = rank_iteratively(request, lambda prompt: "<answer>C00</answer>"
                           if "[C00]" in prompt else oracle(prompt))
    assert bad.valid and bad.ranked_candidate_ids[-1] == target_id

    invalid = rank_iteratively(request, lambda prompt: "Book 7")
    assert not invalid.valid and invalid.failure_reason == "answer_span_count"
    assert len(invalid.trace) == 1
    assert set(invalid.ranked_candidate_ids) == set(best.ranked_candidate_ids)


def test_policy_split_and_pseudo_target_never_need_test_label():
    split = draft_policy_split(range(20), [0, 1, 2, 3], seed="g0-test", train_size=12)
    assert len(split["policy_train"]) == 12
    assert len(split["policy_val"]) == 4
    assert set(split["policy_train"]).isdisjoint(split["policy_val"])
    assert set(split["integration_exposed"]).isdisjoint(split["policy_train"])
    assert split == draft_policy_split(range(20), [0, 1, 2, 3],
                                       seed="g0-test", train_size=12)
    assert pseudo_target_from_train([10, 11, 12, 13], min_prefix_length=3) == (
        (10, 11, 12), 13,
    )
    assert pseudo_target_from_train([10, 11, 10], min_prefix_length=2) is None
    assert pseudo_target_from_train([10, 11], min_prefix_length=2) is None


def test_prefix_snapshot_separates_graph_warmup_and_target():
    source = {1: [10, 11, 12, 13], 2: [20, 13, 21, 22], 3: [13, 30]}
    snapshot, warmup, targets = make_prefix_snapshot(
        source, [1, 2], min_prefix_length=3,
    )
    # Query users lose the warm-up and target events by position; another
    # user's occurrence remains legitimate collaborative evidence.
    assert warmup == {1: 12, 2: 21}
    assert targets == {1: 13, 2: 22}
    assert snapshot.train_data[1] == (10, 11)
    assert snapshot.train_data[2] == (20, 13)
    assert snapshot.train_data[3] == (13, 30)
    assert source[1] == [10, 11, 12, 13]
    graph = UserItemGraph(snapshot)
    assert 13 not in graph.get_user_items(1)
    assert graph.get_item_users(13) == [2, 3]
    assert not hasattr(snapshot, "test_data")
    assert not hasattr(snapshot, "instructions")
    assert snapshot_sha256(snapshot) == snapshot_sha256(snapshot)
    with pytest.raises(TypeError):
        snapshot.train_data[1] = (13,)


def test_prefix_snapshot_removes_warmup_position_even_if_item_repeats():
    snapshot, warmup, targets = make_prefix_snapshot(
        {1: [10, 11, 10, 13]}, [1], min_prefix_length=3,
    )
    assert snapshot.train_data[1] == (10, 11)
    assert warmup == {1: 10} and targets == {1: 13}


def test_prefix_snapshot_rejects_ineligible_queries():
    with pytest.raises(ValueError, match="eligibility"):
        make_prefix_snapshot({1: [10, 11, 10]}, [1], min_prefix_length=2)


def test_agent_opt_in_capture_does_not_change_ranking():
    class FakeClient:
        def __init__(self):
            self.messages = None

        def generate_json(self, messages, properties, **kwargs):
            self.messages = messages
            return {"scores": [
                {"item_id": i, "score": 1 - i / 20, "rationale": "fake"}
                for i in range(10)
            ]}

    client = FakeClient()
    agent = MemRecAgent.__new__(MemRecAgent)
    agent.dataset = SimpleNamespace(item_metadata={i: {"title": f"Book {i}", "tags": []}
                                                   for i in range(10)})
    agent.graph = object()
    agent.pruner = SimpleNamespace(prune=lambda *args: {"n_items": 0, "n_users": 0})
    agent.packer = SimpleNamespace(pack=lambda **kwargs: {"n_neighbors": 0,
                                                           "estimated_tokens": 0})
    agent.storage = SimpleNamespace(render_user_summary=lambda user: "personal memory",
                                    get_item_memory=lambda item: "memory" if item == 0 else None)
    agent.manager = SimpleNamespace(run_stage_r=lambda **kwargs: {
        "facets": [{"facet": "science", "confidence": 0.7}], "support_edges": [],
    })
    agent.reranker = LLMReranker(client)
    agent.reranker_mode = "llm"
    agent.enable_stage_r = True
    agent.n_facets = 7
    agent.temperature = 0.0
    agent.max_tokens = 4000
    agent.vanilla_mode = False
    agent.upstream_empty_facets_prompt = True
    agent.debug = False
    agent.n_stage_r_calls = 0
    agent.n_stage_rr_calls = 0
    baseline, _ = agent.rerank(7, list(range(10)), "Find science", return_details=True)
    captured, details = agent.rerank(7, list(range(10)), "Find science",
                                     return_details=True, capture_rank_request=True,
                                     rank_snapshot_id="synthetic-clean-snapshot")
    assert captured == baseline
    request = RankRequest.from_dict(details["rank_request"])
    assert details["rank_request_sha256"] == request.sha256()
    assert request.snapshot_id == "synthetic-clean-snapshot"
    assert request.user_id == 7 and request.item_memories == {0: "memory"}
    assert LLMReranker(client).build_rerank_prompt(**request.baseline_prompt_kwargs()) == client.messages
    original_messages = client.messages
    assert len(LLMReranker(client).rerank_captured(request)) == 10
    assert client.messages == original_messages
