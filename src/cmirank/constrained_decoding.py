"""Opt-in ID-only structured decoding control; never an upstream fidelity fix.

Derive legal identities from the exact unchanged API input, without labels,
target positions, metadata lookup, fuzzy matching or output repair. This module
is limited to the locked real-memory smoke prompt formats, not a general parser.
"""

from copy import deepcopy
import re

from src.models.llm_client import validate_json_shape


def section(text: str, start: str, end: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("Ambiguous/missing baseline prompt section; no decoder fallback")
    return text.split(start, 1)[1].split(end, 1)[0]


def constrain_id_properties(messages: list[dict], properties: dict) -> tuple[dict, dict]:
    """Copy only ID domains into an existing schema; preserve messages verbatim."""
    if len(messages) != 1 or messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
        raise ValueError("Control expects one unchanged baseline user message")
    text = messages[0]["content"]
    targets = re.findall(r"^\*\*Target User:\*\* User (\d+)$", text, re.MULTILINE)
    constrained = deepcopy(properties)
    if set(properties) == {"facets", "support_edges"}:
        if len(targets) != 1:
            raise ValueError("Missing/ambiguous Stage-R target user")
        uid = int(targets[0])
        neighbors = section(text, "**Collaborative Neighbor Memories:**", "**Context (Candidate Items):**")
        candidates = section(text, "**Context (Candidate Items):**", "(Note: These candidates are for context only, do not score them)")
        neighbor_ids = set(re.findall(r"\[(User-\d+|Item-\d+)\]", neighbors))
        candidate_ids = [int(value) for value in re.findall(r"^\d+\. \[(\d+)\]", candidates, re.MULTILINE)]
        if len(candidate_ids) != 10 or len(set(candidate_ids)) != 10:
            raise ValueError("Stage-R requires the ten visible candidate IDs")
        allowed = sorted(neighbor_ids | {f"User-{uid}"} | {f"Item-{item}" for item in candidate_ids})
        constrained["facets"]["items"]["properties"]["supporting_neighbors"]["items"]["enum"] = allowed
        edges = constrained["support_edges"]["items"]["properties"]
        edges["from"]["enum"], edges["to"]["enum"] = allowed, [f"User-{uid}"]
        audit = {"stage": "stage_r", "allowed_source_ids": allowed, "allowed_target_ids": [f"User-{uid}"],
                 "visible_neighbor_ids": sorted(neighbor_ids), "visible_candidate_ids": candidate_ids}
    elif set(properties) == {"scores"}:
        candidate_text = section(text, "**Candidate Item Memories:**", "**Your Task:**")
        # Baseline joins prompt_parts with "", so entries are not line-anchored.
        candidate_ids = [int(value) for value in re.findall(r"  • Item (\d+) \(", candidate_text)]
        if len(candidate_ids) != 10 or len(set(candidate_ids)) != 10:
            raise ValueError("ReRank requires the ten visible candidate IDs")
        constrained["scores"]["items"]["properties"]["item_id"]["enum"] = sorted(candidate_ids)
        audit = {"stage": "rerank", "allowed_candidate_ids": sorted(candidate_ids)}
    elif set(properties) == {"user_memory", "item_memory", "neighbor_updates"}:
        neighbors = section(text, "**Collaborative Neighbors Available for Memory Propagation:**", "**Your Task:**")
        allowed = sorted(set(re.findall(r"  \d+\. ((?:User|Item)-\d+)(?: |$)", neighbors)))
        if allowed:
            constrained["neighbor_updates"]["items"]["properties"]["neighbor_id"]["enum"] = allowed
        else:
            # An empty enum is invalid JSON Schema. An empty input domain permits
            # only the empty update array; it never invents a sentinel identity.
            constrained["neighbor_updates"]["maxItems"] = 0
        audit = {"stage": "stage_w", "allowed_neighbor_ids": allowed}
    else:
        raise ValueError("Unrecognized schema; no unconstrained control fallback")
    return constrained, {**audit, "label_accessed": False, "output_repair": False,
                         "prompt_modified": False, "primary_provider_replaced": False}


def validate_constrained_output(value, schema: dict) -> None:
    """Independently fail if the backend ignored an enum/empty-domain limit."""
    validate_json_shape(value, schema)

    def visit(node, spec):
        if "enum" in spec and node not in spec["enum"]:
            raise ValueError("Response violates the input-derived identity enum; no repair")
        if spec.get("type") == "object":
            for key, nested in spec.get("properties", {}).items():
                if key in node:
                    visit(node[key], nested)
        elif spec.get("type") == "array":
            if "maxItems" in spec and len(node) > spec["maxItems"]:
                raise ValueError("Response violates the empty propagation domain; no repair")
            for item in node:
                visit(item, spec.get("items", {}))

    visit(value, schema)
