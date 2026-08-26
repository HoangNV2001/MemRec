from copy import deepcopy

import pytest

from src.models.reranker_llm import LLMReranker
from src.multihop.ce0 import evidence_row
from src.multihop import ce1


def _control():
    candidates = list(range(1, 11))
    snippets = {
        f"Item-{index}": f"Science fiction mystery evidence for candidate topic {index} and reader preferences"
        for index in range(1, 11)
    }
    return {
        "user_id": 7,
        "stage_r_context": {
            "K_u": 10,
            "T_u": 300,
            "selected_node_ids": list(snippets),
            "neighbor_snippets": snippets,
        },
        "ranking_context": {
            "candidates": candidates,
            "candidate_order_sha256": "fixed",
            "gold_item_id": 1,
            "instruction": "Looking for a science fiction mystery with strong characters",
            "candidate_titles": {str(index): f"Science Fiction Mystery {index}" for index in candidates},
            "candidate_memories": {str(index): f"A mystery novel set in science fiction world {index}" for index in candidates},
        },
    }


def test_candidate_evidence_is_budgeted_and_independent_of_gold_label():
    control = _control()
    first = evidence_row(control, token_budget=160, snippet_words=8, evidence_per_candidate=1)
    changed_gold = deepcopy(control)
    changed_gold["ranking_context"]["gold_item_id"] = 10
    second = evidence_row(changed_gold, token_budget=160, snippet_words=8, evidence_per_candidate=1)

    assert first == second
    assert first["evaluation_contract"]["gold_item_id_used_by_selector"] is False
    assert first["request_evidence"]["slot_count"] == 10
    assert first["request_evidence"]["estimated_tokens"] <= 160
    candidate = first["candidate_evidence"]
    assert candidate["total_slot_count"] == 10
    assert candidate["estimated_tokens"] <= 160
    assert set(candidate["by_candidate"]) == {str(index) for index in range(1, 11)}
    assert all(len(rows) == 1 for rows in candidate["by_candidate"].values())


def test_reranker_serializes_optional_evidence_without_changing_default_path():
    reranker = LLMReranker(llm_client=None)
    candidates = [{"id": 1, "title": "One"}, {"id": 2, "title": "Two"}]
    kwargs = {
        "user_id": 7,
        "facets": [{"facet": "science fiction", "confidence": 0.8}],
        "candidates": candidates,
        "item_mems": {1: "memory one", 2: "memory two"},
        "instruction": "find science fiction",
    }
    default = reranker.build_rerank_prompt(**kwargs)[0]["content"]
    evidence = reranker.build_rerank_prompt(
        **kwargs,
        shared_evidence=[{"node_id": "Item-9", "text": "shared text"}],
        candidate_evidence={1: [{"node_id": "Item-10", "text": "one text"}]},
    )[0]["content"]

    assert "Shared Collaborative Evidence" not in default
    assert "Candidate-Specific Collaborative Evidence" not in default
    assert "[Item-9] shared text" in evidence
    assert "Evidence for Item 1" in evidence
    assert "[Item-10] one text" in evidence
    assert "Evidence for Item 2" in evidence


def test_ce1_reports_only_independent_repeats_and_paired_delta():
    users = [{"user_id": 7}]
    cache = {}
    values = {"baseline_one_hop": 0.2, "request_one_hop": 0.4, "candidate_one_hop": 0.6}
    for arm, value in values.items():
        for repeat in (1, 2):
            cache[ce1.report_key(7, arm, repeat)] = {
                "metrics": {
                    "hit_at_1": value,
                    "hit_at_3": value,
                    "hit_at_5": value,
                    "ndcg_at_3": value,
                    "ndcg_at_5": value,
                }
            }

    report = ce1.report_metrics(users, cache, repeats=2, bootstrap_samples=20, seed=42)

    assert report["analysed_users"] == 1
    assert report["arms"]["candidate_one_hop"]["ndcg_at_5"] == 0.6
    assert report["comparisons"]["request_one_hop"]["delta_ndcg_at_5"] == pytest.approx(0.2)
    assert report["comparisons"]["candidate_one_hop"]["delta_ndcg_at_5"] == pytest.approx(0.4)
    assert report["comparisons"]["candidate_vs_request_one_hop"]["delta_ndcg_at_5"] == pytest.approx(0.2)


def test_ce1_error_retry_archives_failure_without_duplicate_cache_key(tmp_path):
    raw = tmp_path / "raw.jsonl"
    archive = tmp_path / "retry_errors.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"key":"success","value":1}',
                '{"key":"failed","error":"transient"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = ce1.archive_error_records(raw, archive)

    assert result == {"records_before": 2, "successful_kept": 1, "errors_archived": 1}
    assert [line for line in raw.read_text(encoding="utf-8").splitlines() if line] == ['{"key":"success","value":1}']
    archived = [line for line in archive.read_text(encoding="utf-8").splitlines() if line]
    assert len(archived) == 1 and '"key":"failed"' in archived[0]
