# Temporal Transition augmentation of full MemRec — Books 200-user transfer

**Status (2026-09-28):** v1 30-user CPU smoke passed. Its 200-user execution
stopped before writing results at user 364 because the adapter incorrectly
required even a baseline `malformed_ranking` row to be a permutation. No
method metric was produced. v2 changes **only failure-row handling**: preserve
the original malformed ranking and count it as a miss in every arm. Graph,
seeds, walk count, restart, fusion and alpha are unchanged; v2 must repeat
the 30-user smoke before full evaluation. [GraphWalk3](BOOKS_GRAPH_WALK3_EXPERIMENT.md)
is retired and is **not** the method.

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
| v2 30-user transition smoke | Not run | Must pass before v2 full |
| v2 200-user one-step/PPR transfer | Not run | No method score yet |
