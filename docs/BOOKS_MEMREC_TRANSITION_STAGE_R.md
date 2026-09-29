# Directed transition evidence inside full MemRec Stage-R — Books 700/200

**Preregistered 2026-09-29; real-LLM 30-user smoke passed, 200-user result pending.** This is a bounded integration
experiment, not a retune of the negative [post-ranking residual transfer](BOOKS_MEMREC_TRANSITION_TRANSFER.md).
The primary arm is full MemRec with one-step directed transition evidence in
Stage-R. No PPR, graph-walk, hand-selected case, or search over weights/prompts
is planned.

## Hypothesis and fixed method

The baseline pruner reads a user's train-history items and co-interacting users.
A directed transition graph supplies a distinct source: items other users
encountered *after* one of this user's recent train items. These items could
inform facet synthesis even when a final candidate is not directly reachable.
The prior residual reached only 8/200 candidate lists and gave an inconclusive
`+0.001001` NDCG@5; it does not predict this method will win.

Build directed edges from consecutive `train_data` pairs only, using canonical
`build_transition_graph_from_users` and `one_step_scores`. There are 185,628
directed train transitions across 7,377 users. Each target user has up to six
unique most-recent **train** source items, the historical one-step seed count.
Validation/test items are neither edges nor seeds; candidate IDs, labels and
rankings are not used in evidence selection. Eligible successors must be new
to the user and have a title. Sort by empirical one-step probability, then ID.

Run the baseline `llm_rules` selector unchanged. Retain its first four user
and first six item neighbors (existing mix minima). Fill only remaining slots
with directed successors; retain remaining baseline neighbors if slots remain.
No more than original `k=16` neighbors. Without eligible successors the output
is exactly the baseline. The snippet states directed source/probability and is
packed under original `tau=1800`; its priority affects context packing only,
not the final ranking score. All other model, checkpoint, prompt/schema,
Stage-W fanout, candidate, LLM decoding and evaluation settings equal the
promoted full-MemRec baseline. Stage-R, LLM ReRank and Stage-W warm-up all run
again, so this is full-agent integration rather than a post-ranking residual.

InstructRec `.inter` timestamps are within-user positions, not global time.
This is a sequential train-graph experiment, not strict online replay. The
train-only graph may contain another user's later train pair, just as the
baseline user–item graph contains all train histories. No test outcome is
propagated.

## Cohort, comparison and stop rule

Use the user-designated complete 200 Books dev users, fixed 700 warm-up users,
ten original candidates/user, and self-hosted
`Qwen3-30B-A3B-Instruct-2507-FP8` revision
`5a5a776300a41aaa681dd7ff0106608ef2bc90db`, vLLM 0.10.2, TP=1,
memory fraction 0.60. Full-MemRec baseline NDCG@5 is `0.747918`, Hit@1
`0.595`; SASRec on the same 200 is `0.343320`. Primary comparison is paired
ΔNDCG@5 versus full MemRec. Report Hit@1/5, paired bootstrap 95% interval,
improved/worsened/unchanged and malformed counts; failures stay in the
denominator. No post-result adjustment to seeds, quota, scores, token budget
or model. If Δ is nonpositive or CI crosses zero, do not claim improvement.
These 200 users' labels were already inspected for the residual experiment,
so even a positive outcome is **exploratory**, not fresh confirmation. No
held-out run is authorized here.

## Gates and resources

1. Unit tests and fake-LLM 30-user CPU smoke: only `memrec.pruner.mode` may
   differ from baseline config. Check train-only edges, label independence,
   unique neighbors, `k<=16`, `tau<=1800` and actual packed transition text.
2. Commit/push, pull exact SHA. Resolve one RUNNING `train_TTS` allocation,
   snapshot all cards and use one truly idle H100 in priority 3→2→1→0. Real
   LLM smoke: 30 users, 100% valid permutations, all warm-up stages, no test
   Stage-W, hard cap 165 physical reservations. Fail closed, unload model and
   confirm VRAM baseline.
3. Only after smoke promotion: run 700 warm-up/200 evaluation with same
   code/config/checkpoint/cache contract, max 2,750 physical reservations,
   570-minute timeout. Check final gate, release VRAM, retrieve/check hashes,
   then do offline paired analysis. No second GPU or automatic held-out run.

Run IDs are `books-memrec-transition-stage-r-smoke-v1-hnv` and
`books-memrec-transition-stage-r-dev700-v1-hnv`. The runner modes are
`transitionsmoke700` and `transitiondev700`. Resource preflight/cleanup obey
local-only `internal_docs/H100_RESOURCE_RULES.md`.

## Progress and results

| Gate | Status | Evidence/result |
|---|---|---|
| Frozen residual transfer | Complete, inconclusive | One-step `+0.001001` NDCG@5; PPR `−0.014129`; [details](BOOKS_MEMREC_TRANSITION_TRANSFER.md) |
| Label-blind coverage | Complete | 176/200 have unseen train-only successors; only 3/200 candidate lists overlap them; no target labels used |
| Unit tests + full suite | Passed | 102 tests; train-only edges, label-independent selection, exact fallback, config parity |
| Fake-LLM full-agent smoke | Passed | 30 warm-up and 30 eval; Stage-R/ReRank/W active, 0 failures, 150 fake requests; not a ranking result |
| Fake-LLM 700/200 journal dry-run | Passed | 700 warm-up/200 eval, 0 failures, 2,500 fake requests, hard cap 2,750; not a ranking result |
| Packed evidence check | Passed | 22/30 have transition evidence selected and packed; `k<=16`, estimated context `<=1800` |
| GPU preflight | Passed | Slurm job `17729`, worker-5; GPU 0 idle at 1 MiB, GPU 1 occupied and untouched; exact source commit `89e0d0d` |
| Real-LLM 30-user smoke | Passed | 30/30 valid rankings, 0 failures, 88 physical requests (exact-input cache allowed), cap 165; promotion/source gate passed, 7/7 artifact hashes verified after local pull; prediction SHA-256 `f13c1e0eb61199cd46e45b2d1feeac295998863f0fcee1ddeb4c4c22e1ed9bee` |
| Post-smoke GPU release | Passed | GPU 0 returned from 1 MiB baseline to 1 MiB after runner exit; session ended; GPU 1 process was not touched |
| 700/200 full run | Pending | — |
| Paired analysis | Pending | — |

Smoke artifacts are retained locally under ignored
`results/full_memrec_books_baselines/books-memrec-transition-stage-r-smoke-v1-hnv/`.
The real-LLM smoke score is **not** used for method selection or tuning;
promotion checks schema, cohort, stage counts, request budget, source hash and
GPU cleanup. The full 700/200 run still requires a fresh idle-card preflight.
