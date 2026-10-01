"""Target-blind output-schema clarification, not ranking-prompt optimization."""

import re

import pytest

from src.cmirank.parser import parse_action
from src.cmirank.prompts import render_step_prompt
from src.cmirank.smoke_fixtures import synthetic_rank_requests


def test_schema_clarification_has_no_example_candidate_or_request_mutation():
    request = synthetic_rank_requests()[0]
    original_digest = request.sha256()
    active = ["C03", "C06"]
    prompt = render_step_prompt(request, active)
    assert re.findall(r"\[(C[0-9]{2})\] Title:", prompt) == active
    footer = prompt.split("Choose the single LEAST suitable remaining candidate.")[1]
    assert "bare label (C followed by two digits)" in footer
    assert "no square brackets" in footer
    assert "display delimiters, not part of the label" in footer
    assert re.search(r"C[0-9]{2}", footer) is None  # No positional demonstration.
    assert request.sha256() == original_digest


def test_schema_clarification_is_independent_of_context_and_active_positions():
    requests = synthetic_rank_requests()
    footers = {
        render_step_prompt(request, active).split(
            "Choose the single LEAST suitable remaining candidate.")[1]
        for request in requests
        for active in (["C00", "C09"], ["C02", "C05", "C08"])
    }
    assert len(footers) == 1


@pytest.mark.parametrize("raw,valid", [
    ("<answer>C03</answer>\n", True),
    ("<answer>[C03]</answer>\n", False),
])
def test_parser_still_enforces_original_bare_label_contract(raw, valid):
    action = parse_action(raw, ["C03", "C06"])
    assert action.valid is valid
    if not valid:
        assert action.failure_reason == "invalid_answer_label"
