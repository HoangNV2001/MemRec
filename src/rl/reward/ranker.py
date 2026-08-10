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


def build_pointwise_prompt(
    candidate_id: int,
    title: str,
    memory: str = "",
    m_collab: Optional[Sequence[Dict]] = None,
    instruction: Optional[str] = None,
    user_id: Optional[int] = None,
) -> str:
    """
    Render the yes/no prompt for **one** candidate. Pure function.

    Deliberately identical to ``build_ranker_prompt`` above the candidate block --
    same header, same section titles, same ``M_collab`` rendering -- so switching
    ``scoring`` mode changes the question asked and nothing else. Only the
    candidate section and the final instruction differ.

    Why this exists (M2 Part B, docs/RESULTS.md). The listwise prompt derives all
    ten scores from a single 10-way softmax, so the ranking below rank 1 comes from
    tiny logit gaps and NDCG@5 takes only six distinct values -- two different
    memories tie 71% of the time. Scoring each candidate independently gives ten
    continuous, unconstrained scores, so ties stop being structural rather than
    being papered over with an extra reward term.
    """
    parts = [
        "You are an intelligent recommendation scoring system. Your task is to "
        "evaluate how well a candidate item matches the target user's "
        "preferences based on their personal memory and collaborative signals."
    ]
    if user_id is not None:
        parts.append(f"\n**Target User:** User {user_id}")

    if instruction:
        parts.append(f"\n**User's Current Request:**\n{instruction}")

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

    mem = memory[:150] + "..." if memory and len(memory) > 150 else memory
    parts.append("\n**Candidate Item:**")
    line = f"  Item {candidate_id} ({title})"
    parts.append(f"{line}: {mem}" if mem else line)

    parts.append(
        "\n**Your Task:**\nWould this user be likely to interact with this item "
        "next? Answer Yes or No.\n\nAnswer:"
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
        # M2 (docs/RESULTS.md). The planned Qwen2.5-1.5B-Instruct failed outright
        # -- Spearman 0.307 against a 0.6 DoD, and `lorem` memory outscored real
        # memory. §M2's prescribed fallback, Qwen2.5-3B, only reached 0.5573 and
        # was at chance on the within-user comparisons a GRPO group actually sees.
        # Qwen3.5-4B is the first to pass: rho = 0.7726, Validation B with room,
        # and 62.9% within-user agreement. It also outranks the judge it
        # approximates (NDCG@5 0.75 vs gpt-4o-mini's 0.71).
        #
        # This default is load-bearing. `backfill_baselines.py` once carried its
        # own copy of the model name, silently wrote every split with an
        # unvalidated ranker, and the numbers looked perfectly normal. Everything
        # now inherits from here; `data/rl/baselines_provenance.json` records what
        # actually produced each file.
        model_name: str = "Qwen/Qwen3.5-4B",
        device: str = "cuda",
        include_instruction: bool = True,
        stub_fn: Optional[Callable[[str, int], float]] = None,
        max_prompt_tokens: int = 3072,
        # bfloat16, not float32. The fp32 default existed because Qwen2.5-3B was
        # not batch-invariant in bf16 (2/48 users moved between batch 1 and 24),
        # and §5.1 chose the one-forward-pass design *because* it is deterministic.
        # Qwen3.5-4B measures 0/48 under the torch attention path, so the fp32 tax
        # -- double the VRAM, half the speed -- buys nothing here.
        dtype: str = "bfloat16",
        scoring: str = "listwise",
    ):
        """
        Args:
            mode: ``stub`` (CPU, no model) or ``hf`` (real model).
            include_instruction: whether the InstructRec instruction goes into the
                ranker prompt. The real ``LLM_Rec`` does receive it, so True keeps
                the proxy faithful (§9.5). But the instruction is generated from
                the target item, so it can dominate and flatten the reward across
                different ``M_collab`` -- which is exactly the degenerate-group
                failure of §9.2. **Settled by M2 Part B: keep True.** Dropping the
                instruction more than halved the correlation with the real
                ``LLM_Rec`` (Spearman 0.141 vs 0.307 on the 1.5B ranker), so it
                does not flatten the signal -- it carries much of it. See
                ``docs/RESULTS.md``.
            stub_fn: override the stub scoring, for tests that need a known order.
            dtype: ``float32`` (default) or ``bfloat16`` on GPU.

                **float32 is not a precision luxury here, it is what makes the
                reward reproducible.** §5.1 chose the single-forward-pass design
                precisely because it is deterministic ("GRPO has no critic, so
                reward noise lands straight in the advantage variance"). In bf16
                that promise is false: padding changes the floating-point
                reduction order, so the *same* (prompt, completion) scores
                differently depending on which other rollouts happen to share its
                batch. Measured on Qwen2.5-3B-Instruct over 48 val users
                (docs/RESULTS.md, M2 Part B): batch 24 vs batch 1 flips NDCG@5 for
                2/48 users in bf16 and 0/48 in float32.

                The model is extremely peaked -- ~0.99 of the probability mass
                lands on one letter -- so the logit gaps *below* rank 1 are tiny
                and a bf16-sized perturbation is enough to reorder them. NDCG@5
                reads exactly that tail.
            scoring: ``listwise`` (§5.1 as written: one forward pass, softmax over
                the candidate letters A-J) or ``pointwise`` (one forward pass *per
                candidate*, score = P("Yes")).

                Pointwise is §M2's second prescribed fallback, and M2 Part B is
                what sent us to it: listwise derives ten scores from a single
                10-way softmax, so NDCG@5 takes only six distinct values and two
                different memories tie 71% of the time, which zeroes the GRPO
                advantage. Ten independent continuous scores remove that ceiling
                structurally. It costs ~10x the forward passes, which is why
                listwise remains the default until pointwise is shown to be worth
                it.
        """
        if mode not in ("stub", "hf"):
            raise ValueError(f"unknown ranker mode: {mode!r}")
        if scoring not in ("listwise", "pointwise"):
            raise ValueError(f"unknown scoring mode: {scoring!r}")
        self.mode = mode
        self.scoring = scoring
        self.model_name = model_name
        self.device = device
        self.include_instruction = include_instruction
        self.stub_fn = stub_fn or _stub_logit
        self.max_prompt_tokens = max_prompt_tokens
        if dtype not in ("float32", "bfloat16"):
            raise ValueError(f"unknown ranker dtype: {dtype!r}")
        self.dtype = dtype
        # Pointwise expands one request into len(candidates) prompts, so the caller's
        # batch_size no longer describes what reaches the GPU. This caps the real
        # forward-pass batch independently.
        self.pointwise_chunk = 64

        self._model = None
        self._tokenizer = None
        self._letter_token_ids: Optional[List[int]] = None
        self._yes_no_token_ids: Optional[List[int]] = None

    # -- real model, loaded lazily so CPU tests never touch it ---------------

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Left padding: we read the logits at the final position, so every prompt's
        # last real token must sit at index -1.
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Plain load + .to() rather than device_map=: the ranker is a single small
        # model on one device, so sharded loading buys nothing and device_map would
        # make `accelerate` a hard dependency of the reward path.
        # float32 unless bf16 was asked for explicitly, and never on CPU where bf16
        # matmuls are slow or unsupported. See the `dtype` note in __init__ for why
        # the default is not bf16.
        dtype = (torch.bfloat16
                 if self.dtype == "bfloat16" and str(self.device).startswith("cuda")
                 else torch.float32)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
        self._model.to(self.device)
        self._model.eval()
        self._letter_token_ids = self._resolve_letter_tokens()
        self._yes_no_token_ids = self._resolve_yes_no_tokens()

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

    def _resolve_yes_no_tokens(self) -> List[int]:
        """
        Map Yes/No to single token ids, for ``scoring="pointwise"``.

        Same leading-space ambiguity as the letters, and the same hard requirement
        that the two ids differ -- if they collided every candidate would score
        identically and the reward would be silently constant.
        """
        ids = []
        for word in ("Yes", "No"):
            chosen = None
            for form in (word, f" {word}"):
                encoded = self._tokenizer.encode(form, add_special_tokens=False)
                if len(encoded) == 1:
                    chosen = encoded[0]
                    break
            if chosen is None:
                chosen = self._tokenizer.encode(word, add_special_tokens=False)[0]
            ids.append(chosen)
        if ids[0] == ids[1]:
            raise RuntimeError(
                f"Yes/No map to the same token for {self.model_name}: {ids}. "
                "Every candidate would score identically; pick another ranker."
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
        if self.scoring == "pointwise":
            return self._score_hf_pointwise(requests)
        return self._score_hf(prompts, [r["candidates"] for r in requests])

    def _score_hf_pointwise(self, requests: Sequence[Dict]) -> List["RankerOutput"]:
        """
        One yes/no forward pass per candidate; score = P("Yes").

        Every candidate of every request is flattened into a single list before
        batching. Batching per request instead would cap the batch at one user's
        candidate list (10), wasting most of the GPU on prompts that are already
        ~half the length of the listwise one.
        """
        import torch

        self._ensure_loaded()
        flat, spans = [], []
        for r in requests:
            titles = r.get("candidate_titles") or {}
            mems = r.get("candidate_memories") or {}
            start = len(flat)
            for cid in r["candidates"]:
                flat.append(build_pointwise_prompt(
                    candidate_id=cid,
                    title=titles.get(str(cid), f"Item-{cid}"),
                    memory=(mems or {}).get(str(cid), ""),
                    m_collab=r.get("m_collab"),
                    instruction=r.get("instruction") if self.include_instruction else None,
                    user_id=r.get("user_id"),
                ))
            spans.append((start, len(flat)))

        yes_id, no_id = self._yes_no_token_ids
        probs: List[float] = []
        for start in range(0, len(flat), self.pointwise_chunk):
            chunk = flat[start:start + self.pointwise_chunk]
            texts = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True,
                )
                for p in chunk
            ]
            batch = self._tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_prompt_tokens,
            ).to(self._model.device)
            with torch.no_grad():
                logits = self._last_position_logits(batch)
            # Rank by the raw logit margin, NOT by softmax(P("Yes")).
            #
            # The model answers "No" to almost every candidate, so P("Yes") is
            # routinely ~1e-30 and underflows to exactly 0.0 in float32 -- which
            # silently recreates the tie problem pointwise exists to remove
            # (measured: 8 of 10 candidates collapsing to one value). The margin is
            # a strictly monotonic transform of P("Yes"), so the ranking is
            # identical wherever the probability is representable, and it stays
            # well-separated where the probability is not.
            margin = (logits[:, yes_id] - logits[:, no_id]).float()
            probs.extend(margin.tolist())

        return [self._rank_from_scores(r["candidates"], probs[lo:hi])
                for r, (lo, hi) in zip(requests, spans)]

    @staticmethod
    def _rank_from_scores(candidates: Sequence[int],
                          scores: Sequence[float]) -> "RankerOutput":
        """
        Rank by independent per-candidate probabilities.

        Unlike ``_rank_from_logits`` these are NOT renormalised across candidates:
        forcing them to sum to 1 would reintroduce exactly the coupling that makes
        the listwise scores tie.

        ``scores`` here is the yes/no logit margin, not a probability in [0, 1] --
        see ``_score_hf_pointwise`` for why the probability form is unusable. Any
        consumer that needs a probability should apply a sigmoid, and should expect
        it to underflow for most candidates.
        """
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return RankerOutput(
            ranking=[int(candidates[i]) for i in order],
            scores={int(candidates[i]): float(scores[i]) for i in range(len(candidates))},
            logits={int(candidates[i]): float(scores[i]) for i in range(len(candidates))},
        )

    def _templated(self, prompt: str) -> str:
        """
        Apply the chat template so that the NEXT token is the answer itself.

        The whole scorer reads the logits at one position and expects the model to
        be about to emit a candidate letter. Reasoning models break that: their
        template ends the generation prompt inside an open ``<think>`` block, so
        the next token is the first word of a chain of thought, and the letter
        logits being read are noise from the tail of the distribution. Measured on
        Qwen3.5-4B before this fix: mass on A-J = **0.00002** (Qwen2.5-3B: 0.9998),
        top token 'The' at p=0.82 -- and the resulting NDCG@5 was 0.34 on every
        arm, i.e. random (0.295), which reads as "the model is weak" rather than
        "the scorer is pointed at the wrong token".

        ``enable_thinking=False`` is the supported way to ask for a template with
        no thinking block; templates that do not accept the kwarg are unaffected.
        If one opens a thinking block anyway, close it explicitly.
        """
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:      # template does not take the kwarg -- the common case
            text = self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        stripped = text.rstrip()
        if stripped.endswith("<think>"):
            text = stripped + "\n</think>\n\n"
        return text

    def letter_mass(self, prompt: str) -> float:
        """
        Softmax mass the model puts on A-J at the position the scorer reads.

        A number near 1.0 means the one-forward-pass design of §5.1 holds for this
        model. A number near 0 means every score this ranker returns is noise, and
        it is invisible downstream -- the NDCG just looks like a weak model. Cheap
        enough to assert before trusting any ranker on a checkpoint not seen before.
        """
        import torch

        self._ensure_loaded()
        batch = self._tokenizer(
            [self._templated(prompt)], return_tensors="pt",
            truncation=True, max_length=self.max_prompt_tokens,
        ).to(self._model.device)
        with torch.no_grad():
            row = self._last_position_logits(batch)[0].float()
        probs = torch.softmax(row, dim=-1)
        return float(sum(probs[i].item() for i in self._letter_token_ids))

    def _score_hf(self, prompts: Sequence[str], candidate_lists) -> List[RankerOutput]:
        import torch

        self._ensure_loaded()
        texts = [
            self._templated(p)
            for p in prompts
        ]
        batch = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).to(self._model.device)

        with torch.no_grad():
            logits = self._last_position_logits(batch)   # prefill only, no generation

        outputs = []
        for row, candidates in zip(logits, candidate_lists):
            letter_logits = [row[self._letter_token_ids[i]].item() for i in range(len(candidates))]
            outputs.append(self._rank_from_logits(candidates, letter_logits))
        return outputs

    def _last_position_logits(self, batch):
        """
        Vocabulary logits at the final position only, shape ``(B, V)``.

        Calling ``model(**batch).logits`` would materialise logits for *every*
        position: at batch 64 x 3072 tokens x 151936 vocab that is **60 GB** in
        bf16, to use 19 MB of it. That OOMs before the first reward is ever
        computed. Two ways to avoid it, tried in order of officialness:

        1. ``logits_to_keep=1`` (transformers >= 4.49; ``num_logits_to_keep`` on
           4.45-4.48) -- the supported way to ask for a trailing slice.
        2. Run the base transformer and apply ``lm_head`` to the last hidden state
           ourselves. Works on any decoder exposing ``.model`` / ``.lm_head``.
        """
        for kwarg in ("logits_to_keep", "num_logits_to_keep"):
            try:
                return self._model(**batch, **{kwarg: 1}).logits[:, -1, :]
            except TypeError:
                continue        # this transformers version does not accept it

        base = getattr(self._model, "model", None)
        head = getattr(self._model, "lm_head", None)
        if base is not None and head is not None:
            hidden = base(**batch).last_hidden_state[:, -1, :]
            return head(hidden)

        raise RuntimeError(
            f"cannot take a last-position logit slice for {self.model_name}: it "
            f"accepts neither logits_to_keep nor exposes .model/.lm_head. Computing "
            f"full logits would need ~60 GB at batch 64. Use a different ranker."
        )

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
