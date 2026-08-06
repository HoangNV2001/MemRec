"""
Parser robustness for src/rl/policy.py.

RL_PLAN.md M1 DoD: "Parser xử lý được 20 ca output méo mó viết tay."
Every case below is a shape a 3-4B model actually emits. The contract under test
is: never raise, and salvage facets whenever any are recoverable.
"""
import pytest

from src.rl.policy import (
    MAX_FACET_CHARS,
    build_prompt,
    normalize_node_id,
    parse_facets,
)

GOOD = '{"facets": [{"facet": "epic fantasy", "confidence": 0.9, "supporting_neighbors": ["Item-1"]}]}'


# --- the 20 malformed cases -------------------------------------------------
# (name, raw completion, expect_any_facets)
MALFORMED_CASES = [
    ("markdown_fence", '```json\n' + GOOD + '\n```', True),
    ("fence_no_lang", '```\n' + GOOD + '\n```', True),
    ("leading_prose", 'Sure! Here is the JSON:\n' + GOOD, True),
    ("trailing_prose", GOOD + '\n\nLet me know if you need more facets.', True),
    (
        "trailing_comma",
        '{"facets": [{"facet": "cozy mystery", "confidence": 0.8, "supporting_neighbors": ["Item-2"],},]}',
        True,
    ),
    (
        "single_quotes",
        "{'facets': [{'facet': 'hard sci-fi', 'confidence': 0.7, 'supporting_neighbors': ['Item-3']}]}",
        True,
    ),
    ("smart_quotes", '{“facets”: [{“facet”: “romance”, “confidence”: 0.6, “supporting_neighbors”: []}]}', True),
    ("missing_confidence", '{"facets": [{"facet": "war history", "supporting_neighbors": ["User-9"]}]}', True),
    ("confidence_as_string", '{"facets": [{"facet": "poetry", "confidence": "0.85", "supporting_neighbors": []}]}', True),
    ("confidence_as_percent", '{"facets": [{"facet": "poetry", "confidence": "85%", "supporting_neighbors": []}]}', True),
    ("confidence_out_of_range", '{"facets": [{"facet": "poetry", "confidence": 7.5, "supporting_neighbors": []}]}', True),
    ("sources_as_string", '{"facets": [{"facet": "noir", "confidence": 0.5, "supporting_neighbors": "Item-4, User-5"}]}', True),
    ("sources_underscore_ids", '{"facets": [{"facet": "noir", "confidence": 0.5, "source_ids": ["u_4412", "i_88301"]}]}', True),
    ("top_level_list", '[{"facet": "thriller", "confidence": 0.9, "supporting_neighbors": ["Item-6"]}]', True),
    ("facets_as_dict", '{"facets": {"1": {"facet": "memoir", "confidence": 0.4, "supporting_neighbors": []}}}', True),
    ("alt_key_text", '{"preference_facets": [{"text": "cookbooks", "confidence": 0.3}]}', True),
    ("bare_strings", '{"facets": ["young adult dystopia", "graphic novels"]}', True),
    (
        "truncated_midway",
        '{"facets": [{"facet": "space opera", "confidence": 0.9, "supporting_neighbors": ["Item-7"]}, '
        '{"facet": "milita',
        True,
    ),
    ("empty_string", "", False),
    ("pure_prose", "I could not find enough evidence to build preference facets.", False),
]


@pytest.mark.parametrize("name,raw,expect_facets", MALFORMED_CASES, ids=[c[0] for c in MALFORMED_CASES])
def test_parser_never_raises_and_salvages(name, raw, expect_facets):
    result = parse_facets(raw)
    assert isinstance(result.facets, list)
    if expect_facets:
        assert result.n_facets >= 1, f"{name}: expected to salvage at least one facet"
        for f in result.facets:
            assert isinstance(f["facet"], str) and f["facet"]
            assert 0.0 <= f["confidence"] <= 1.0
            assert isinstance(f["supporting_neighbors"], list)
    else:
        assert result.n_facets == 0
        assert result.error


def test_twenty_malformed_cases_covered():
    """The DoD asks for 20 hand-written cases; keep that count honest."""
    assert len(MALFORMED_CASES) == 20


def test_clean_output_is_valid():
    result = parse_facets(GOOD)
    assert result.is_valid
    assert result.error is None
    assert result.facets[0]["supporting_neighbors"] == ["Item-1"]


def test_truncated_output_is_marked_invalid_but_usable():
    raw = next(c[1] for c in MALFORMED_CASES if c[0] == "truncated_midway")
    result = parse_facets(raw)
    assert result.truncated
    assert not result.is_valid, "truncated output must still cost the format penalty"
    assert result.n_facets == 1


def test_hallucinated_ids_are_dropped_when_whitelist_given():
    raw = '{"facets": [{"facet": "x", "confidence": 0.5, "supporting_neighbors": ["Item-1", "Item-99999"]}]}'
    result = parse_facets(raw, valid_node_ids=["Item-1", "User-2"])
    assert result.facets[0]["supporting_neighbors"] == ["Item-1"]


def test_max_facets_truncates():
    raw = '{"facets": [' + ",".join(
        f'{{"facet": "f{i}", "confidence": 0.5, "supporting_neighbors": []}}' for i in range(12)
    ) + "]}"
    assert parse_facets(raw, max_facets=7).n_facets == 7


def test_overlong_facet_is_clipped():
    raw = '{"facets": [{"facet": "%s", "confidence": 0.5, "supporting_neighbors": []}]}' % ("z" * 5000)
    assert len(parse_facets(raw).facets[0]["facet"]) == MAX_FACET_CHARS


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Item-123", "Item-123"),
        ("item-123", "Item-123"),
        ("USER-7", "User-7"),
        ("u_4412", "User-4412"),
        ("i_88301", "Item-88301"),
        ("[Item-42]", "Item-42"),
        ("Item 42", "Item-42"),
        ("Item-007", "Item-7"),
        ("123", None),          # bare int: ambiguous
        (123, None),
        (None, None),
        ("", None),
        ("Book-3", None),
    ],
)
def test_normalize_node_id(raw, expected):
    assert normalize_node_id(raw) == expected


def test_retrieval_bundle_matches_frozen_reranker_shape():
    """A parsed result must drop straight into LLMReranker without an adapter."""
    from src.models.reranker_llm import LLMReranker

    bundle = parse_facets(GOOD).to_retrieval_bundle()
    messages = LLMReranker.build_rerank_prompt(
        self=None.__class__ and LLMReranker.__new__(LLMReranker),
        user_id=1,
        facets=bundle["facets"],
        candidates=[{"id": 5, "title": "Some Book", "tags": []}],
        item_mems={},
        instruction="I want a fantasy novel.",
    )
    assert "epic fantasy" in messages[0]["content"]


def test_prompt_is_candidate_blind():
    prompt = build_prompt(
        user_id=17,
        user_memory="Likes fantasy.",
        neighbors_text="1. [Item-1] Dune",
        n_facets=7,
    )
    assert "User 17" in prompt
    assert "Likes fantasy." in prompt
    assert "Candidate" not in prompt and "candidate" not in prompt


def test_prompt_handles_empty_memory():
    prompt = build_prompt(user_id=1, user_memory="", neighbors_text="1. [Item-1] Dune")
    assert "No personal memory recorded yet" in prompt
