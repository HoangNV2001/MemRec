"""Deterministic CPU harness for N-1-step candidate elimination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .labels import label_to_item
from .parser import parse_action
from .prompts import render_step_prompt
from .request import RankRequest


@dataclass(frozen=True)
class StepTrace:
    step: int
    active_labels_before: tuple[str, ...]
    action_label: str | None
    raw_output: str
    valid: bool
    failure_reason: str | None


@dataclass(frozen=True)
class IterativeResult:
    ranked_candidate_ids: tuple[int, ...]
    valid: bool
    failure_reason: str | None
    trace: tuple[StepTrace, ...]


def rank_iteratively(
    request: RankRequest,
    generate: Callable[[str], str],
) -> IterativeResult:
    """Run a policy; invalid steps remain misses in any scientific metric.

    A deterministic debug ranking is returned for storage on invalid output,
    but ``valid=False`` must override it in the evaluator.  No repair call is
    made and no target/reward field is accepted by this interface.
    """
    mapping = label_to_item([int(row["id"]) for row in request.candidates])
    active = list(mapping)
    exclusions: list[str] = []
    traces: list[StepTrace] = []
    while len(active) > 1:
        raw = generate(render_step_prompt(request, active))
        parsed = parse_action(raw, active)
        traces.append(StepTrace(
            step=len(traces), active_labels_before=tuple(active),
            action_label=parsed.label, raw_output=raw, valid=parsed.valid,
            failure_reason=parsed.failure_reason,
        ))
        if not parsed.valid:
            debug_labels = active + list(reversed(exclusions))
            return IterativeResult(
                tuple(mapping[label] for label in debug_labels), False,
                parsed.failure_reason, tuple(traces),
            )
        exclusions.append(parsed.label)
        active.remove(parsed.label)
    ranking_labels = active + list(reversed(exclusions))
    ranked_ids = tuple(mapping[label] for label in ranking_labels)
    if len(ranked_ids) != len(mapping) or set(ranked_ids) != set(mapping.values()):
        raise AssertionError("Iterative reconstruction is not a permutation")
    return IterativeResult(ranked_ids, True, None, tuple(traces))
