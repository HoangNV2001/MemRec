# Books reduced-cost full-architecture MemRec — 700/200 dev subset

**Status (2026-09-25):** cohort and code frozen locally; CPU fake-LLM wiring
test passed. The new real-LLM 30-user smoke and 700/200 GPU run have not yet
started. This is an **exploratory, reduced-memory benchmark**, not the full
7,377-warm-up/2,000-dev result or a replacement for it.

## Why this protocol

The original full-dev run would require 7,377 warm-up users and 2,000 scored
dev users, about 26,131 logical LLM calls. The user requested approximately
one tenth of each phase and at most about ten H100-hours. This protocol uses
exactly **700 warm-up users** and **200 scored users**, giving
`700 × (Stage-R + ReRank + Stage-W) + 200 × (Stage-R + ReRank) = 2,500`
logical LLM calls before exact-input cache hits. The durable physical-request
cap is 2,750; the inference timeout is 570 minutes, leaving startup and
cleanup within roughly ten GPU-hours. An interrupted run is not a result.

## Frozen user and evidence contract

- Data, 10 original candidates per user, graph snapshot, Qwen3-30B-A3B FP8
  revision `5a5a776300a41aaa681dd7ff0106608ef2bc90db`, vLLM 0.10.2,
  prompts, decoding, one H100, memory cap 0.60, seed 42, Stage-R/RR/W, and
  absence of test-label Stage-W are unchanged from the promoted smoke.
- Evaluation users are the first **200 IDs of the locked 2,000-user dev cohort**,
  in original cohort order; SHA-256 of sorted IDs is
  `5395ee7775d9e5d0add11ed386715c34a002138c282f4f910f775ea90871a837`.
  No held-out targets are opened.
- Warm-up starts from the lowest 700 original user IDs and, where necessary,
  swaps out the highest **non-eval** IDs to include all 200 scored users.
  Exactly 700 pre-test histories are warmed in sorted ID order; SHA-256 is
  `e8b9e14032df6e2ea8d0389c62de13d0e8fb087edac127aacbfba8b4062050c9`.
  Selection uses only user IDs and the pre-existing dev split, never targets.
- The collaborative graph still comes from the frozen permitted training
  interactions. Only Stage-W memory construction is downsampled. Therefore
  scores cannot be presented as the full MemRec baseline or compared directly
  with the paper's full-data numbers. Any SASRec comparison must score this
  same 200-user/candidate subset and disclose the unequal warm-up/training
  budgets.

## Gates and planned artifacts

1. Local tests and CPU fake-LLM integration: 700/700 Stage-R/RR/W warm-up,
   200/200 Stage-R/RR evaluation, original candidates, 2,500 budget charges,
   zero failures. Replaying the journal must leave predictions unchanged.
2. On the authorized Slurm allocation, recheck all four GPUs, select one
   sufficiently idle H100 and run `smoke700` on 30 dev users with the new
   subset config. Its exact-input responses must match the earlier promoted
   30-user real-LLM smoke byte-for-byte; inspect format, stage counts and
   GPU cleanup before promotion. Cache hits are allowed and separately
   counted; no manual prompt/model tuning.
3. Only after promotion, run `dev700` with its own run ID, journal and durable
   budget. Gate requires 700 committed warm-up rows, 200 valid dev predictions,
   exact original candidate permutations, no held-out artifact, and GPU VRAM
   returned after vLLM exits. Pull artifacts locally and verify hashes before
   interpreting results. No automatic held-out run or full-dev resume.

Run IDs: `books-memrec-llm-dev700-smoke-v1-hnv` and
`books-memrec-llm-dev700-v1-hnv`. Both use the existing checkpoint/cache
namespace; exact-input cache keys ensure only identical requests are reused.

## Progress and results

| Gate | State | Evidence |
|---|---|---|
| Original real-LLM smoke | Passed | 30/30 rankings, 150 physical calls, 0 failures; see [baseline contract](BOOKS_SELFHOST_LLM_BASELINE.md) |
| Local 700/200 CPU fake-LLM wiring | Passed | 700 warm-up, 200 eval, 2,500 fake requests, cap 2,750, 0 failures; journal replay gives identical prediction SHA-256 `d5b5a43ac89dc53a0b12f5dd98408d0a81d7df5be0da2173974af1eebe937ebe` |
| New real-LLM 30-user smoke | Pending | — |
| 700/200 GPU run | Blocked on new smoke and fresh GPU preflight | — |
| Held-out | Sealed | No reduced-cost held-out evaluation planned |

The fake-LLM NDCG is **not** a research result. Record NDCG@5, Hit@1,
failures, per-user predictions, physical/cache requests, timing and GPU
before/after only after the real run passes its gate.
