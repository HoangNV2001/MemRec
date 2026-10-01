"""Project-authored, target-blind iterative ranking prompt serializer."""

from __future__ import annotations

from typing import Sequence

from .labels import label_to_item
from .request import RankRequest


def render_static_context(request: RankRequest) -> str:
    if request.vanilla_mode:
        raise ValueError("CM-IRank v1 only supports full-MemRec memory context")
    parts = [
        "You are a ranking policy. At each turn, remove exactly one remaining "
        "book that is LEAST likely to satisfy the user's request. Use only "
        "the evidence provided. Do not invent candidates.\n",
        f"Target user: {request.user_id}\n",
        "User request:\n",
        request.instruction if request.instruction else "(No request provided)",
        "\nCollaborative preference evidence:\n",
    ]
    if request.collaborative_facets:
        for facet in request.collaborative_facets[:10]:
            text = facet.get("facet", facet.get("text", "N/A"))
            confidence = facet.get("confidence", 0)
            parts.append(f"- {text} (confidence: {confidence:.2f})\n")
    else:
        parts.append("(No facets extracted)\n")
    return "".join(parts)


def render_step_prompt(request: RankRequest, active_labels: Sequence[str]) -> str:
    mapping = label_to_item([int(row["id"]) for row in request.candidates])
    if not active_labels or len(active_labels) != len(set(active_labels)):
        raise ValueError("Active labels must be unique and nonempty")
    if set(active_labels) - set(mapping):
        raise ValueError("Unknown active label")
    by_id = {int(row["id"]): row for row in request.candidates}
    parts = [render_static_context(request),
             f"\nREMAINING CANDIDATES ({len(active_labels)}):\n"]
    for label in active_labels:
        item_id = mapping[label]
        row = by_id[item_id]
        title = row.get("title", f"Item {item_id}")
        memory = request.item_memories.get(item_id, "(No memory recorded)")
        # The baseline MemRec prompt exposes only title and first 150 memory
        # characters in full-memory mode; do not add tags or description here.
        if len(memory) > 150:
            memory = memory[:150] + "..."
        parts.append(f"[{label}] Title: {title}\nItem memory: {memory}\n")
    parts.append(
        "Choose the single LEAST suitable remaining candidate. "
        "Return exactly one active label inside <answer>...</answer>.\n"
    )
    return "".join(parts)
