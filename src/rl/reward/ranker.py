"""
Frozen listwise ranker used as the reward signal (RL_PLAN.md §5.1).

Scores all ten candidates in **one prefill-only forward pass**: the prompt ends
with a forced choice, and we read the next-token logits restricted to the
candidate letters A-J. Softmax over those ten logits gives a full ranking, hence
NDCG, with no sampling noise. GRPO has no critic, so reward noise lands straight
in the advantage variance -- determinism here is not a micro-optimisation.

Two modes:

* ``stub``  -- no model, deterministic scores. Lets the entire reward path be
  tested on CPU (M2 Part A). A parser crash or a shape bug found here costs
  nothing; found mid-GRPO it costs a rented session.
* ``hf``    -- the real model (default ``Qwen2.5-1.5B-Instruct``), M2 Part B.

The prompt deliberately mirrors ``LLMReranker.build_rerank_prompt`` in MemRec
mode (same section headers, same ``• Item {id} ({title}): {memory}`` lines) so
the proxy stays close to the real ``LLM_Rec``. Per RL_PLAN.md §4.2 it is copied,
not imported -- the original eval path must stay untouched.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import hashlib
import math

# Candidate labels. Ten candidates is the paper's protocol (§2).
LETTERS = "ABCDEFGHIJ"


@dataclass
class RankerOutput:
    """Result of scoring one user's candidate list."""

    ranking: List[int]                      # item ids, best first
    scores: Dict[int, float] = field(default_factory=dict)   # item_id -> probability
    logits: Dict[int, float] = field(default_factory=dict)   # item_id -> raw logit

    def top1(self) -> Optional[int]:
        return self.ranking[0] if self.ranking else None


def build_ranker_prompt(
    candidates: Sequence[int],
    candidate_titles: Dict[str, str],
    candidate_memories: Optional[Dict[str, str]] = None,
    m_collab: Optional[Sequence[Dict]] = None,
    instruction: Optional[str] = None,
    user_id: Optional[int] = None,
) -> str:
    """
    Render the forced-choice ranking prompt. Pure function -- testable on CPU.

    ``m_collab`` is the policy's facet list (the action being rewarded). Passing
    None or an empty list produces the "no collaborative memory" prompt, which is
    exactly the ``r_null`` arm of §5.4.
    """
    parts = [
        "You are an intelligent recommendation scoring system. Your task is to "
        "evaluate how well each candidate item matches the target user's "
        "preferences based on their personal memory and collaborative signals."
    ]
    if user_id is not None:
        parts.append(f"\n**Target User:** User {user_id}")

    if instruction:
        parts.append(f"\n**User's Current Request:**\n{instruction}")

    # Only claim to have found patterns when there are some -- asserting
    # "we identified preferences" and then listing none reads as contradictory
    # and measurably degrades scoring (docs/RESULTS.md, M0 bug #3).
    if m_collab:
        parts.append("\n**User Preferences (Extracted from Collaborative Memories):**")
        parts.append(
            "Based on collaborative signals from neighboring users and items, "
            "we have identified the following preference patterns:"
        )
        for i, facet in enumerate(m_collab[:10], 1):
            text = facet.get("facet", facet.get("text", "N/A"))
            conf = facet.get("confidence", 0.0)
            parts.append(f"  {i}. {text} (confidence: {conf:.2f})")

    parts.append("\n**Candidate Items:**")
    for letter, cid in zip(LETTERS, candidates):
        title = candidate_titles.get(str(cid), f"Item-{cid}")
        memory = (candidate_memories or {}).get(str(cid), "")
        if memory and len(memory) > 150:
            memory = memory[:150] + "..."
        line = f"  {letter}. Item {cid} ({title})"
        parts.append(f"{line}: {memory}" if memory else line)

    parts.append(
        "\n**Your Task:**\nChoose the single candidate the user is most likely to "
        "interact with next. Reply with one letter only.\n\nAnswer:"
    )
    return "".join(p if p.startswith("\n") else "\n" + p for p in parts).strip()


def _stub_logit(prompt: str, item_id: int) -> float:
    """
    Deterministic pseudo-logit from (prompt, item). Same input -> same score,
    forever, on any machine. Not meaningful; just stable and well-spread.
    """
    digest = hashlib.sha256(f"{prompt}|{item_id}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big") / 2 ** 64      # [0, 1)
    return (raw - 0.5) * 8.0                                # roughly [-4, 4]


class FrozenRanker:
    """Frozen listwise ranker. Never trained, never updated."""

    def __init__(
        self,
        mode: str = "stub",
        model_name: str = "Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        include_instruction: bool = True,
        stub_fn: Optional[Callable[[str, int], float]] = None,
        max_prompt_tokens: int = 3072,
    ):
        """
        Args:
            mode: ``stub`` (CPU, no model) or ``hf`` (real model).
            include_instruction: whether the InstructRec instruction goes into the
                ranker prompt. The real ``LLM_Rec`` does receive it, so True keeps
                the proxy faithful (§9.5). But the instruction is generated from
                the target item, so it can dominate and flatten the reward across
                different ``M_collab`` -- which is exactly the degenerate-group
                failure of §9.2. M2 Validation A/B decides this empirically; see
                ``docs/RESULTS.md``.
            stub_fn: override the stub scoring, for tests that need a known order.
        """
        if mode not in ("stub", "hf"):
            raise ValueError(f"unknown ranker mode: {mode!r}")
        self.mode = mode
        self.model_name = model_name
        self.device = device
        self.include_instruction = include_instruction
        self.stub_fn = stub_fn or _stub_logit
        self.max_prompt_tokens = max_prompt_tokens

        self._model = None
        self._tokenizer = None
        self._letter_token_ids: Optional[List[int]] = None

    # -- real model, loaded lazily so CPU tests never touch it ---------------

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()
        self._letter_token_ids = self._resolve_letter_tokens()

    def _resolve_letter_tokens(self) -> List[int]:
        """
        Map A-J to single token ids.

        Tokenizers differ on whether the leading space belongs to the letter, so
        try both and require the ten ids to be distinct -- a collision would make
        two candidates share a logit and silently corrupt every reward.
        """
        ids = []
        for letter in LETTERS:
            candidates = [letter, f" {letter}"]
            chosen = None
            for form in candidates:
                encoded = self._tokenizer.encode(form, add_special_tokens=False)
                if len(encoded) == 1:
                    chosen = encoded[0]
                    break
            if chosen is None:
                chosen = self._tokenizer.encode(letter, add_special_tokens=False)[0]
            ids.append(chosen)
        if len(set(ids)) != len(ids):
            raise RuntimeError(
                f"letter tokens collide for {self.model_name}: {ids}. "
                "Two candidates would share a logit; pick another ranker."
            )
        return ids

    # -- scoring ------------------------------------------------------------

    def score(
        self,
        candidates: Sequence[int],
        candidate_titles: Dict[str, str],
        candidate_memories: Optional[Dict[str, str]] = None,
        m_collab: Optional[Sequence[Dict]] = None,
        instruction: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> RankerOutput:
        """Rank one candidate list. See ``score_batch`` for the training path."""
        return self.score_batch(
            [dict(
                candidates=candidates,
                candidate_titles=candidate_titles,
                candidate_memories=candidate_memories,
                m_collab=m_collab,
                instruction=instruction,
                user_id=user_id,
            )]
        )[0]

    def score_batch(self, requests: Sequence[Dict]) -> List[RankerOutput]:
        """
        Score many candidate lists at once.

        Batched because §5.1 budgets 64 rollouts per forward pass and the M2 DoD
        requires >= 20 reward/s.
        """
        prompts = [
            build_ranker_prompt(
                candidates=r["candidates"],
                candidate_titles=r["candidate_titles"],
                candidate_memories=r.get("candidate_memories"),
                m_collab=r.get("m_collab"),
                instruction=r.get("instruction") if self.include_instruction else None,
                user_id=r.get("user_id"),
            )
            for r in requests
        ]
        if self.mode == "stub":
            return [
                self._rank_from_logits(
                    r["candidates"],
                    [self.stub_fn(p, cid) for cid in r["candidates"]],
                )
                for p, r in zip(prompts, requests)
            ]
        return self._score_hf(prompts, [r["candidates"] for r in requests])

    def _score_hf(self, prompts: Sequence[str], candidate_lists) -> List[RankerOutput]:
        import torch

        self._ensure_loaded()
        texts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        batch = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",   # last position must be the real end of every prompt
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).to(self._model.device)

        with torch.no_grad():
            logits = self._model(**batch).logits[:, -1, :]   # prefill only, no generation

        outputs = []
        for row, candidates in zip(logits, candidate_lists):
            letter_logits = [row[self._letter_token_ids[i]].item() for i in range(len(candidates))]
            outputs.append(self._rank_from_logits(candidates, letter_logits))
        return outputs

    @staticmethod
    def _rank_from_logits(candidates: Sequence[int], letter_logits: Sequence[float]) -> RankerOutput:
        """Softmax over the candidate letters, then sort. Ties break by input order."""
        top = max(letter_logits)
        exps = [math.exp(x - top) for x in letter_logits]
        total = sum(exps) or 1.0
        probs = [e / total for e in exps]

        order = sorted(range(len(candidates)), key=lambda i: (-probs[i], i))
        return RankerOutput(
            ranking=[int(candidates[i]) for i in order],
            scores={int(candidates[i]): probs[i] for i in range(len(candidates))},
            logits={int(candidates[i]): letter_logits[i] for i in range(len(candidates))},
        )
