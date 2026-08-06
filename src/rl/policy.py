"""
Stage-R policy: prompt construction and output parsing.

The policy (``LM_Mem``) is asked to emit exactly the JSON schema the *frozen*
``LLMReranker`` already consumes -- ``facet`` / ``confidence`` /
``supporting_neighbors`` -- so a trained policy can be dropped straight into the
original pipeline with no adapter. RL_PLAN.md §5.2 calls the evidence field
``source_ids``; that is this repo's ``supporting_neighbors``, same thing.

Two deliberate differences from ``MemRecManager.build_stage_r_prompt``:

* **candidate-blind** (RL_PLAN.md §5.3): the candidate list is never shown, so the
  policy cannot encode the answer into the memory it writes.
* **no InstructRec instruction**: the instruction paraphrases the target book, and
  in the original pipeline it goes to Stage-ReRank only. Feeding it to Stage-R
  would hand the policy the answer.

Parsing must never raise. A malformed generation is a *reward signal*
(``λ_fmt``, §5), not a crash -- one exception mid-rollout costs a rented GPU hour.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import json
import re

# Facet text longer than this is almost certainly the model rambling; we keep it
# but flag it so the length penalty can act on it.
MAX_FACET_CHARS = 400

_NODE_ID_RE = re.compile(r"^(user|item)-(\d+)$", re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

# Keys the model plausibly uses instead of the ones we asked for.
_FACETS_KEYS = ("facets", "preference_facets", "facet_list", "preferences", "output")
_TEXT_KEYS = ("facet", "text", "description", "preference", "facet_text")
_SOURCE_KEYS = ("supporting_neighbors", "source_ids", "sources", "evidence", "neighbors")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are an intelligent memory retrieval system for personalized recommendation. \
Your task is to analyze the user's personal memory and collaborative memories from their neighbors \
to extract preference facets.

**Target User:** User {user_id}

**User's Personal Memory:**
{user_memory}

**Collaborative Neighbor Memories:**
The following neighboring users and items provide collaborative signals for understanding this user's preferences:

{neighbors_text}

**Your Task:**
Analyze the user's personal memory and the collaborative memories from neighboring users and items to \
identify {n_facets} distinct preference facets that characterize this user's interests and tastes.

For each preference facet, provide:
1. "facet": a concise natural language description of the preference
2. "confidence": a number between 0 and 1 indicating how strongly the evidence supports it
3. "supporting_neighbors": a list of neighbor IDs from the list above that support it, \
written exactly as shown (e.g. "User-123", "Item-456")

Rules:
- Every ID in "supporting_neighbors" MUST appear in the neighbor list above. Never invent IDs.
- Keep each facet under 30 words. Be specific about genre, theme, and style; do not just list titles.
- Output ONLY a JSON object, no prose and no markdown fences.

**Expected Output Format:**
{{"facets": [{{"facet": "...", "confidence": 0.0, "supporting_neighbors": ["Item-1", "User-2"]}}]}}"""


def build_prompt(
    user_id: int,
    user_memory: str,
    neighbors_text: str,
    n_facets: int = 7,
) -> str:
    """Render the candidate-blind Stage-R prompt for one user."""
    memory = (user_memory or "").strip()
    if not memory:
        memory = "(No personal memory recorded yet for this user)"
    return PROMPT_TEMPLATE.format(
        user_id=user_id,
        user_memory=memory,
        neighbors_text=neighbors_text.strip(),
        n_facets=n_facets,
    )


def to_chat(prompt: str) -> List[Dict[str, str]]:
    """Single-user-message chat format, matching the original pipeline."""
    return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedFacets:
    """Result of parsing one policy generation."""

    facets: List[Dict[str, Any]] = field(default_factory=list)
    is_valid: bool = False
    error: Optional[str] = None
    n_dropped: int = 0          # facet-shaped entries discarded as unusable
    truncated: bool = False     # recovered from an incomplete JSON array

    @property
    def n_facets(self) -> int:
        return len(self.facets)

    def to_retrieval_bundle(self) -> Dict[str, Any]:
        """Shape expected by the frozen ``LLMReranker``."""
        return {"facets": self.facets, "vector_profile": None, "support_edges": []}


def normalize_node_id(raw: Any) -> Optional[str]:
    """
    Canonicalise a neighbour reference to ``User-<id>`` / ``Item-<id>``.

    Accepts the schema-native form plus the shapes models actually emit:
    ``u_4412``, ``item 88301``, ``[Item-123]``, bare integers are rejected
    (ambiguous between user and item).
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return None

    s = str(raw).strip().strip("[](){}<>\"' \t")
    if not s:
        return None

    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    # u-4412 / i-88301 shorthand from RL_PLAN §5.2
    s = re.sub(r"^u-", "User-", s, flags=re.IGNORECASE)
    s = re.sub(r"^i-", "Item-", s, flags=re.IGNORECASE)

    m = _NODE_ID_RE.match(s)
    if not m:
        return None
    return f"{m.group(1).capitalize()}-{int(m.group(2))}"


def _coerce_confidence(raw: Any) -> float:
    """Confidence clamped to [0, 1]; anything unreadable becomes 0.5."""
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not m:
            return 0.5
        value = float(m.group(0))
        if "%" in raw:
            value /= 100.0
    else:
        return 0.5
    if value != value:  # NaN
        return 0.5
    return max(0.0, min(1.0, value))


def _coerce_sources(raw: Any) -> List[str]:
    """Normalise the evidence field; a bare string is treated as one ID."""
    if raw is None:
        return []
    items: Sequence[Any]
    if isinstance(raw, str):
        # "Item-1, User-2" or "Item-1"
        items = [p for p in re.split(r"[,;]", raw) if p.strip()]
    elif isinstance(raw, dict):
        items = list(raw.values())
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [raw]

    out: List[str] = []
    for it in items:
        node = normalize_node_id(it)
        if node and node not in out:
            out.append(node)
    return out


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _extract_json_blob(text: str) -> Optional[str]:
    """Return the outermost balanced {...} or [...] region, ignoring braces in strings."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        # Unbalanced: hand back the tail so the truncation salvage can try.
        return text[start:]
    return None


def _loads_lenient(blob: str) -> Optional[Any]:
    for attempt in (
        blob,
        _TRAILING_COMMA_RE.sub(r"\1", blob),
        _TRAILING_COMMA_RE.sub(r"\1", blob.replace("'", '"')),
    ):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _salvage_objects(blob: str) -> List[Any]:
    """
    Pull every *complete* ``{...}`` out of a truncated generation.

    A run cut off at ``max_completion_length`` leaves the enclosing
    ``{"facets": [...`` unbalanced, so the facet objects only ever close at depth
    >= 1 -- we therefore record balanced spans at every depth and keep the ones
    that look like facets. Throwing these away would make the length penalty
    double-count as a format penalty.
    """
    spans: List[str] = []
    stack: List[int] = []
    in_str = False
    escaped = False
    for i, ch in enumerate(blob):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            spans.append(blob[stack.pop():i + 1])

    objects: List[Any] = []
    for span in spans:
        parsed = _loads_lenient(span)
        if isinstance(parsed, dict) and any(k in parsed for k in _TEXT_KEYS):
            objects.append(parsed)
    return objects


def _find_facet_list(obj: Any) -> Optional[List[Any]]:
    """Locate the facet array in whatever envelope the model produced."""
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return None
    for key in _FACETS_KEYS:
        if key in obj:
            value = obj[key]
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                # {"facets": {"1": {...}, "2": {...}}}
                return list(value.values())
            if isinstance(value, str):
                return [value]
    # A single bare facet object.
    if any(k in obj for k in _TEXT_KEYS):
        return [obj]
    return None


def _coerce_facet(raw: Any) -> Optional[Dict[str, Any]]:
    """One entry -> canonical facet dict, or None if there is no usable text."""
    if isinstance(raw, str):
        text = raw.strip()
        return (
            {"facet": text[:MAX_FACET_CHARS], "confidence": 0.5, "supporting_neighbors": []}
            if text
            else None
        )
    if not isinstance(raw, dict):
        return None

    text = None
    for key in _TEXT_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    if text is None:
        return None

    sources: List[str] = []
    for key in _SOURCE_KEYS:
        if key in raw:
            sources = _coerce_sources(raw[key])
            if sources:
                break

    return {
        "facet": text[:MAX_FACET_CHARS],
        "confidence": _coerce_confidence(raw.get("confidence", raw.get("score", 0.5))),
        "supporting_neighbors": sources,
    }


def parse_facets(
    completion: str,
    valid_node_ids: Optional[Sequence[str]] = None,
    max_facets: Optional[int] = None,
) -> ParsedFacets:
    """
    Parse a policy generation into canonical facets. Never raises.

    ``is_valid`` is the strict format signal for ``λ_fmt``: it is True only when a
    JSON object was recovered without salvage *and* at least one facet survived.
    Recovered-by-salvage output still yields usable facets but is marked invalid,
    so the format penalty keeps pushing toward clean JSON.

    If ``valid_node_ids`` is given, hallucinated neighbour IDs are dropped here so
    the grounding reward (§5.2) scores only real citations.
    """
    if not isinstance(completion, str) or not completion.strip():
        return ParsedFacets(error="empty completion")

    text = completion.translate(_SMART_QUOTES)
    text = _strip_fences(text).strip()

    blob = _extract_json_blob(text)
    if blob is None:
        return ParsedFacets(error="no JSON object found")

    truncated = False
    obj = _loads_lenient(blob)
    if obj is None:
        salvaged = _salvage_objects(blob)
        if not salvaged:
            return ParsedFacets(error="undecodable JSON")
        obj, truncated = salvaged, True

    raw_facets = _find_facet_list(obj)
    if raw_facets is None:
        return ParsedFacets(error="no facet list in JSON", truncated=truncated)

    allowed = set(valid_node_ids) if valid_node_ids is not None else None

    facets: List[Dict[str, Any]] = []
    n_dropped = 0
    for entry in raw_facets:
        facet = _coerce_facet(entry)
        if facet is None:
            n_dropped += 1
            continue
        if allowed is not None:
            facet["supporting_neighbors"] = [
                s for s in facet["supporting_neighbors"] if s in allowed
            ]
        facets.append(facet)

    if max_facets is not None:
        facets = facets[:max_facets]

    if not facets:
        return ParsedFacets(error="no usable facets", n_dropped=n_dropped, truncated=truncated)

    return ParsedFacets(
        facets=facets,
        is_valid=not truncated,
        error=None if not truncated else "recovered from truncated JSON",
        n_dropped=n_dropped,
        truncated=truncated,
    )
