"""Strict elimination-action parser; never guesses an item from free text."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_LABEL = re.compile(r"C[0-9]{2}", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAction:
    label: str | None
    valid: bool
    failure_reason: str | None = None


def parse_action(raw: str, active_labels: Sequence[str]) -> ParsedAction:
    if not isinstance(raw, str):
        return ParsedAction(None, False, "non_text_output")
    answers = _ANSWER.findall(raw)
    if len(answers) != 1:
        return ParsedAction(None, False, "answer_span_count")
    answer = answers[0].strip().upper()
    if not _LABEL.fullmatch(answer):
        return ParsedAction(None, False, "invalid_answer_label")
    if answer not in active_labels:
        return ParsedAction(None, False, "label_not_active")
    return ParsedAction(answer, True)
