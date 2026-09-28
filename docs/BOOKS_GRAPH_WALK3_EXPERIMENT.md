# GraphWalk3 evidence for full MemRec — locked Books 700/200 experiment

**Status (2026-09-28):** implementation and CPU wiring smoke passed, source
commit `e3064e2` is on the server, but real-LLM 30-user smoke has **not** run.
GPU preflight found both H100s busy (78,658 MiB each, compute contexts on both),
so no MemRec GPU process was started. No method score exists yet.

## Evaluation contract

Per user decision, the already frozen **200 Books dev users** are the complete
evaluation cohort for **this experiment**. Their original 10 candidates,
target IDs and order are unchanged. The warm-up remains the same 700 pre-test
users as baseline run `books-memrec-llm-dev700-v1-hnv`; changing that would
destroy the paired comparison. Both arms use the same train graph, seed 42,
Qwen3-30B-A3B-Instruct-2507-FP8 revision
`5a5a776300a41aaa681dd7ff0106608ef2bc90db`, vLLM 0.10.2, one H100,
GPU-memory fraction 0.60, Stage-R, LLM ReRank, Stage-W warm-up, fixed `k=16`,
`tau=1800`, type minima 4 users/6 items, and no dev-target Stage-W.

Baseline: NDCG@5 **0.7479179811**, Hit@1 **0.595** on all 200 users; two
malformed rankings count as misses. See [baseline evidence](BOOKS_DEV700_PROTOCOL.md).
This 200-user protocol is called *full* only within the present experiment;
it is not the paper's 7,377-user full-data benchmark or a held-out result.

## One intervention, fixed before model evaluation

Only the graph evidence pruner changes. For target user `u`, choose an item
uniformly from `u`'s train history, then a different user uniformly from that
item's train users. Normalize this two-step probability distribution over
other users. One more step chooses uniformly among each reached user's train
items. Select Stage-R user evidence by two-step mass and item evidence by
three-step mass, preserving the existing top-`k` and user/item quota. If no
other user can be reached, fall back to direct history items. Scores and ties
are deterministic; no domain rule, fitted weight, path-strength threshold,
candidate label or manually tuned depth is involved. Candidate IDs are *not*
used by the pruner; the unchanged Stage-R and ReRank still see all candidates.

This is a replacement for evidence selection **inside full MemRec**. It is
not a graph-score residual or a new local ranker. Stage-W propagates through
the selected pruned subgraph as in baseline. Implementation:
`src/memory/graph_walk_pruner.py`, wired through `src/memory/pruner.py`;
config: `configs/memrec_instructrec-books_dev700_graph_walk3.yaml`.

Label-free exposure check on the 200 locked users: 3,193 selected evidence
nodes in total (at most 16/user); 638 are previously unseen item nodes
reachable in three steps; 179/200 users receive at least one such item.
This checks that the intervention actually changes graph reach, **not** that
it improves ranking. No target labels were inspected for method selection.

## Execution gates

1. Local test suite and 30-user CPU fake-LLM wiring smoke. The fake score is
   never reported as model performance.
2. Resolve exactly one authorized `train_TTS` allocation; run the strict
   empty-H100 preflight from `internal_docs/H100_RESOURCE_RULES.md`. If no
   card is empty, stop without starting inference.
3. Real-LLM smoke: `graphsmoke700`, exactly the first 30 frozen dev users,
   warm-up on those 30, 30 complete candidate permutations, zero malformed
   rankings, Stage-R/ReRank/Stage-W counts, pinned model/config/source hashes,
   bounded physical requests, and GPU released afterward. Cache may reuse
   only exact identical inputs; this method smoke is **not** required to match
   baseline predictions.
4. Only after the method smoke is promoted, run `graphdev700`: exactly the
   same 700 warm-up and 200 scored users as baseline, budget cap 2,750
   physical requests, 570-minute inference timeout. Preserve every user in
   the denominator, including malformed outputs. Pull and hash-check journal,
   manifest, predictions, metrics and GPU snapshot before comparing.
5. Report paired per-user ΔNDCG@5, bootstrap 95% CI, Hit@1, failures and cost
   against baseline. Success requires a positive paired ΔNDCG@5 with CI lower
   bound above zero; otherwise report a negative/inconclusive result. No
   manual rule/weight tuning or second method variant after seeing outcomes.

Run IDs: `books-memrec-graph-walk3-dev700-smoke-v1-hnv` and
`books-memrec-graph-walk3-dev700-v1-hnv`. The script refuses overwrite and
requires smoke promotion before the 200-user run.

## Progress and results

| Gate | Status | Evidence |
|---|---|---|
| Baseline 700/200 | Pass | NDCG@5 0.747918; Hit@1 0.595; 2/200 malformed, counted as misses |
| GraphWalk3 unit/regression tests | Pass | 97/97 local tests; config differs only at `memrec.pruner.mode` |
| GraphWalk3 CPU wiring smoke | Pass | 30/30 rankings, 30 Stage-W warm-up calls, 0 failures; fake LLM only |
| Real-LLM graph smoke | Blocked by GPU occupancy | 2026-09-28 preflight: both visible H100s busy with compute contexts; no MemRec GPU process started |
| GraphWalk3 700/200 | Blocked on real-LLM smoke | No method metric yet |
| Paired result | Waiting | No conclusion yet |
