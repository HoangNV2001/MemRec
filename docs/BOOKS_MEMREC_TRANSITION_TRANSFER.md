# Temporal Transition augmentation of full MemRec — Books 200-user transfer

**Status (2026-09-28):** v2 completed on the locked 200-user Books set after
its 30-user smoke. The primary one-step arm **did not establish improvement**:
ΔNDCG@5 `+0.001001`, paired CI95% `[−0.005536,+0.008540]`. Secondary PPR
was worse by `−0.014129`. No new LLM request or GPU was used.
[GraphWalk3](BOOKS_GRAPH_WALK3_EXPERIMENT.md) is retired and is **not** the
method. Do not retune alpha/restart/depth on these 200 outcomes.

## Method identity and exact scope

The method is **directed temporal item-transition graph augmentation**, not
multi-hop expansion on an undirected user–item graph. The previously validated
[Temporal Transition/PPR method](TRANSITION_PPR_METHOD.md) supplies the
unchanged scoring primitives. One-step transition is the **primary arm**;
terminal-state Transition-PPR is a **secondary propagation variant**. Both are
fused with the **completed full-MemRec ranking**, so Stage-R, LLM ReRank and
Stage-W warm-up are present but not rerun. This first transfer is precisely a
*post-ranking augmentation of full MemRec*, not a claim that transition
evidence has been integrated into memory synthesis or propagation.

The user-designated complete evaluation set is the already frozen **200 Books
dev users** from `books-memrec-llm-dev700-v1-hnv`, with the same 700-user
pre-test warm-up and original 10 candidates. Baseline NDCG@5 is `0.747918`,
Hit@1 is `0.595`; its two malformed rankings remain misses for every method
arm. This is not the paper-style 7,377-user result or an external held-out
replication.

## Books temporal and leakage contract

The InstructRec `.inter` file stores `timestamp=position+1` **within each
user**, not real or globally comparable times. This transfer therefore treats
the ordered `train_data[u]` sequence as the only source of directed edges:
each adjacent pair `item_t → item_(t+1)` is an observed transition. Validation
and test target rows supply **no graph edges**. For each evaluated user, up
to six unique most-recent seed items come from train history plus the
validation item, both known before the final test target. Original candidate
IDs enter scoring symmetrically; the positive label is never passed to graph
scoring or fusion. The graph is shared over all users' train sequences, as in
the baseline's frozen train graph. Because global timestamps do not exist,
this is a **within-user sequential transition transfer**, not the strict
global-time replay of the Kaggle Amazon/MovieLens experiments.

One-step score is the empirical probability of reaching each candidate from
the six seeds in one directed edge. PPR uses the original 50,000 deterministic
walks per user, geometric stop probability `0.15`, and maximum 64 steps;
candidate score is terminal frequency. For both arms, convert the frozen
MemRec permutation to reciprocal-log rank scores, min–max normalize graph
scores within the 10 candidates and add `alpha=0.80`, **frozen from the prior
transition study**. All-zero/equal graph scores preserve the MemRec ranking.
There is no Books-specific tuning, rule selection or outcome-based parameter
change. The old alpha may transfer poorly; that is an empirical result, not
a reason to retune on these 200 labels.

Implementation reuses `src/temporal_common/{graph,metrics}.py` through
`scripts/evaluate_books_memrec_transition.py`. No LLM request or GPU is
needed because the full-MemRec 200-user predictions are already sealed.

## Smoke-first and result gates

1. Verify the existing MemRec completion hashes, locked 700/200 output gate,
   candidate order and all 200 prediction rows. Build directed graph from
   train sequences only; assert its edge count equals consecutive train
   positions, without reading targets.
2. Run **30-user smoke** using the first 30 of the same frozen cohort and the
   full 50,000-walk setting. Require ten candidate scores/permutations per
   user, deterministic replay, zero malformed baseline rankings in this
   smoke, no candidate/history overlap and no new LLM/GPU use. Hash the smoke
   artifact, graph primitives, baseline predictions and data.
3. Only if smoke passes, reuse its 30 scores and score the remaining 170
   users. Preserve all 200 in the denominator; a malformed baseline row is
   a miss in both augmented arms. Produce per-user predictions, NDCG@5,
   Hit@1/5, paired bootstrap 95% intervals and improved/worsened/unchanged
   counts against full MemRec. Do not select the better arm post hoc:
   one-step is primary, PPR is secondary irrespective of outcomes.
4. A positive result is evidence only for the **post-ranking augmentation**
   on this frozen 200-user task. A thesis claim about improving collaborative
   memory itself requires a separately pre-registered integration into
   Stage-R/Stage-W and evaluation, not relabeling this residual arm.

Run IDs: `books-memrec-transition-transfer-v1-hnv` (smoke only; invalid full
attempt) and `books-memrec-transition-transfer-v2-hnv` (corrected failure
contract). No manual tuning after smoke/full outcomes. No H100 is requested
for this offline evaluation.

## Progress and results

| Gate | Status | Evidence |
|---|---|---|
| Baseline 700/200 | Passed | NDCG@5 `0.747918`, Hit@1 `0.595`, 2 failures counted as misses |
| Transfer unit tests | Passed | Train-only directed edges; label-independent scoring; failure retention |
| v1 30-user transition smoke | Passed | 30/30, deterministic replay, 0 LLM/GPU |
| v1 200-user transfer | Aborted, no score | Failure-row validation bug at user 364; no full artifact |
| v2 30-user transition smoke | Passed | 30/30, identical smoke-score SHA-256 to v1, deterministic replay, 0 LLM/GPU |
| v2 200-user one-step/PPR transfer | Complete; primary gate failed | 200/200 rows, baseline's 2 malformed rankings retained as misses, 0 LLM/GPU |

### Frozen transfer result (v2)

| Arm | NDCG@5 | Hit@1 | ΔNDCG@5 vs full MemRec | Paired bootstrap CI95% |
|---|---:|---:|---:|---:|
| Full MemRec baseline | 0.747918 | 0.595 | — | — |
| **One-step transition residual (primary)** | **0.748919** | **0.595** | **+0.001001** | **[−0.005536,+0.008540]** |
| Transition-PPR residual (secondary) | 0.733789 | 0.550 | −0.014129 | [−0.036241,+0.009521] |

Primary improved/worsened/unchanged users: **1/1/198**; PPR: **6/20/174**.
One-step had nonzero score for at least one candidate in only **8/200 users**
(8/2,000 candidate slots; gold reachable in 7 users), changing just two
rankings. PPR reached at least one candidate in **53/200 users** (68 slots;
gold in 25 users) and changed 32 rankings, but worsened more cases than it
improved. This sparse/mixed candidate support is a plausible explanation for
the failed transfer, not proof of a single causal mechanism. The original
Kaggle/MovieLens results remain valid in their own timestamped protocols;
they do **not** imply transfer to these InstructRec candidates.

Result artifacts are in ignored local directory
`results/full_memrec_books_baselines/books-memrec-transition-transfer-v2-hnv/`.
`full_predictions.jsonl` SHA-256 is
`7e92ea4896da40023c4e84526e7cf555b7fbdc0a6c199cd4d4f8c8481cc524bd`;
`full_result.json` SHA-256 is
`4c57c5a9218395c72ac2532b152ac26343050f3841d65481ed61428e42d6f2d3`.
The manifest pins the baseline prediction/data/scoring-source hashes and
frozen parameters; the saved smoke hash was verified before scoring the
remaining 170 users. NDCG@5/Hit@1 were recomputed independently from all
200 per-user target positions and matched the report. This result **does not
meet** the paired-improvement gate and is not a thesis claim of improved full
MemRec. Stop this residual transfer on this cohort; any new integration
hypothesis needs a fresh preregistered evaluation rather than adjustments to
these 200 labels.
