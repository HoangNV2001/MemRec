# Books reduced-cost full-architecture MemRec — 700/200 dev subset

**Status (2026-09-28):** CPU fake-LLM wiring and the new real-LLM 30-user
smoke passed. The 700/200 real-LLM run completed and passed its gate:
**NDCG@5 = 0.7479, Hit@1 = 0.5950** on the locked 200-user dev subset.
This is an **exploratory, reduced-memory benchmark**, not the full
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
2. On the authorized Slurm allocation, recheck all visible H100s, select one
   **empty** H100 (low utilization, <512 MiB baseline memory, no driver-listed
   compute process) with per-run priority **GPU 3 → 2 → 1 → 0** and run
   `smoke700` on 30 dev users with the new
   subset config. Its exact-input responses must match the earlier promoted
   30-user real-LLM smoke byte-for-byte; inspect format, stage counts and
   GPU cleanup before promotion. Cache hits are allowed and separately
   counted; no manual prompt/model tuning.
3. Only after promotion, run `dev700` with its own run ID, journal and durable
   budget. Gate requires 700 committed warm-up rows, 200 dev prediction rows,
   exact original candidate sets and valid permutations for successful rows,
   explicit failure markers for malformed outputs, no held-out artifact, and
   GPU VRAM returned after vLLM exits. Pull artifacts locally and verify hashes
   before interpreting results. No automatic held-out run or full-dev resume.

Run IDs: `books-memrec-llm-dev700-smoke-v1-hnv` and
`books-memrec-llm-dev700-v1-hnv`. Both use the existing checkpoint/cache
namespace; exact-input cache keys ensure only identical requests are reused.

## Progress and results

| Gate | State | Evidence |
|---|---|---|
| Original real-LLM smoke | Passed | 30/30 rankings, 150 physical calls, 0 failures; see [baseline contract](BOOKS_SELFHOST_LLM_BASELINE.md) |
| Local 700/200 CPU fake-LLM wiring | Passed | 700 warm-up, 200 eval, 2,500 fake requests, cap 2,750, 0 failures; journal replay gives identical prediction SHA-256 `d5b5a43ac89dc53a0b12f5dd98408d0a81d7df5be0da2173974af1eebe937ebe` |
| New real-LLM 30-user smoke | **Passed** | 2026-09-28, worker-5 GPU 1, commit `bda9ed2`; 30/30 ranking, 0 failures, 0 new physical requests (150 exact-input cache hits), prediction SHA identical to original promoted smoke; GPU before/after 1 MiB; 7 promoted artifact hashes verified locally. |
| 700/200 GPU run | **Passed** | `books-memrec-llm-dev700-v1-hnv`, commit `bda9ed2`, Slurm job `17729`, worker-5 GPU 1; completed 2026-09-28 12:20 +07, about 4 h 05 min from GPU-before snapshot. 700 warm-up and 200 dev journal entries; 2 malformed rankings counted as misses; 2,052 physical attempts reserved against cap 2,750. Exit 0; GPU 1 at 1 MiB after cleanup. |
| Held-out | Sealed | No reduced-cost held-out evaluation planned |

### Real-LLM dev-subset score

| Metric | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| Hit | 0.5950 | 0.7700 | 0.8900 | 0.9900 |
| NDCG | 0.5950 | 0.6989 | 0.7479 | 0.7803 |

The denominator is **all 200 locked dev users**. Two LLM outputs were marked
`malformed_ranking` and assigned position 10 (miss at every reported K); they
were **not discarded**. There were 700 Stage-R, 700 ReRank and 700 Stage-W
warm-up calls, then 200 Stage-R and 200 ReRank dev calls; no dev Stage-W and
no held-out evaluation. The model was Qwen3-30B-A3B-Instruct-2507-FP8 at
revision `5a5a776300a41aaa681dd7ff0106608ef2bc90db` under vLLM 0.10.2,
one H100, GPU-memory fraction 0.60. The 2,052 budget reservations include
both LLM clients; `llm_token_stats` in the result JSON describes only the
Stage-R/Stage-W client, so its 1,311 physical requests must not be mistaken
for the combined count.

Artifacts were copied to
`results/full_memrec_books_baselines/books-memrec-llm-dev700-v1-hnv/`.
All eight SHA-256 entries in `completion.json` match the local copies; the
independent local `check_books_memrec_full_dev.py --protocol dev700` gate also
passed. The run's `gpu-after-attempt-1.csv` records GPU 1 back at 1 MiB after
vLLM exited. Later GPU occupancy is a separate cluster state and does not
imply this run still holds VRAM.

The fake-LLM score is **not** a research result. This real-LLM score is a
reduced-memory, 200-user exploratory result. It cannot establish superiority
over original full MemRec, SASRec or a proposed graph method without a paired
evaluation on the **same users and candidate lists** under a declared training
budget. Do not tune methods on these dev labels manually or present this as a
held-out or full-dev score.
