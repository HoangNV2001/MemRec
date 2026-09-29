# Directed transition evidence inside full MemRec Stage-R — Books 700/200

**Completed 2026-09-29; primary improvement gate failed.** This is a bounded integration
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
| 700/200 full run | Passed output gate | Slurm job `17729`, worker-5, one H100 GPU 1, source commit `d0026f5`; 700 warm-up/200 eval, all Stage-R/ReRank/W warm-up calls present, 1 malformed output retained as a miss, 1,504 physical reservations of 2,750 cap. Runtime from GPU snapshots: 3h 19m 09s. |
| Post-full GPU release | Passed | GPU 1 returned to 1 MiB; process/session ended. GPU 0 occupancy after the run belongs to other activity and was not touched. |
| Artifact provenance | Passed | 8/8 completion hashes match local copies; independent local `check_books_memrec_full_dev.py --protocol dev700 --expected-pruner-mode transition_one_step` passed. Prediction SHA-256 `ec61dd656303624cfa34fa94875f08c44b34169cd2786f2f43c6606cdfc8dcd4`. |
| Paired analysis | Complete; improvement gate failed | Method NDCG@5 `0.742968`, full MemRec `0.747918`; delta `−0.004950`, CI95% `[−0.032974,+0.023356]`. |

Smoke artifacts are retained locally under ignored
`results/full_memrec_books_baselines/books-memrec-transition-stage-r-smoke-v1-hnv/`.
The real-LLM smoke score is **not** used for method selection or tuning;
promotion checks schema, cohort, stage counts, request budget, source hash and
GPU cleanup. The full run artifacts are under ignored
`results/full_memrec_books_baselines/books-memrec-transition-stage-r-dev700-v1-hnv/`.
The copied metrics JSON and run log additionally match remote SHA-256 values
`2e4c3e71123023f9d857ffde5b3501700e7475cf24cdefb4c38c0bf2c3acf25f`
and `756038ece2f0b48600ce5ae9ceff743193720cedbcf3ab1c4862f895ab9b371c`.

### Paired 200-user result

| Arm | NDCG@5 | Hit@1 | Hit@5 | Malformed rankings |
|---|---:|---:|---:|---:|
| Full MemRec baseline | 0.747918 | 0.595 | 0.890 | 2 |
| **Transition evidence in full MemRec Stage-R** | **0.742968** | **0.575** | **0.900** | **1** |

The primary paired ΔNDCG@5 (method − baseline) is `−0.004950` with 10,000
paired-bootstrap resamples and seed `20260928`, CI95%
`[−0.032974,+0.023356]`. Hit@1 delta is `−0.020` (CI95%
`[−0.060,+0.020]`); Hit@5 delta is `+0.010`. Target NDCG@5 improved for
19 users, worsened for 23 and was unchanged for 158; 163 full permutations
changed. Both arms use exactly the same 200 user IDs, 10 candidate IDs per
user, target IDs, model revision and 700-user warm-up scope. The baseline's
two malformed outputs and method's one were retained as misses. The method
successfully ranked one baseline-failed user, but still had lower overall
NDCG@5. Per-user ranking validation and metric recomputation agree with the
saved result JSON.

Thus this specific integration **did not improve full MemRec** on the locked
Books task. The CI includes zero: the observed negative delta is not evidence
of a reliably harmful effect either. The baseline is already strong, and
transition successors may displace useful baseline context or add weakly
related facets; these are hypotheses, not established causes. The method
remains numerically above SASRec `0.343320` on this cohort, but the thesis
gate requires beating full MemRec too and is not met. Do not retune neighbor
quota, seed count, priority or prompt against these now-exposed 200 labels,
and do not promote this variant to held-out. A new conceptual mechanism needs
a separately frozen development/evaluation protocol.
