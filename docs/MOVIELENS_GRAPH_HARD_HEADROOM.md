# MovieLens graph-hard multi-hop headroom protocol

> Status: **complete, sealed, stop before LLM**. Both preregistered multi-hop
> headroom gates failed on 200 fresh development users. This study did not
> retune or reopen the prior 500-event primary cohort and used zero LLM/GPU
> requests.

## 0. Execution status

| Phase | Status | Result |
|---|---|---|
| Unit/data smoke | Pass | 42/42 tests; 50-user adapter smoke |
| Graph smoke | Pass | 20 events; 180/180 negatives reachable in both views |
| Full prepare | Pass | 200 users; zero overlap with sealed 520 users |
| Label-blind score lock | Pass | 200 rows; 50.000 walks/event/view |
| Development evaluation | **Fail / sealed** | Exact +0,0194 with CI crossing 0; session -0,0353 |
| LLM/GPU promotion | **Stopped** | Preregistered stop rule applied |

## 1. Motivation and question

The frozen MovieLens transfer passed by a large margin, but one-step graph-only
was stronger than PPR graph-only and all graph arms benefited from a very easy
one-positive/nine-uniform-negative candidate set. Gold one-step reachability
was 90–95%, while random-negative reachability was only 30–55%.

The next question is therefore narrower:

> When all negatives already carry one-step transition evidence, does
> multi-hop PPR improve ranking beyond one-step transition scores?

This phase is a CPU-only development/headroom gate. It cannot become a final
test result. No LLM/GPU request is allowed until the gate is sealed.

## 2. Fresh development cohort

- Use singleton-five-minute-session positive targets from the existing
  MovieLens development interval.
- Exclude every user in the sealed M1 development and 500-event primary
  cohorts.
- Deterministically select one target per remaining user, rank users by a new
  hash salt and inspect exactly the first 500 users.
- Keep the first 200 users within that fixed scan set for whom the candidate
  contract below is feasible. Do not expand beyond 500 if fewer than 200 pass.
- Keep the original rating threshold, minimum positive history, six recent
  seeds and ten-candidate listwise interface.

For development events, both graph views use only ratings strictly before the
global **train cutoff**. Thus the entire graph snapshot predates every target.

## 3. Graph-hard candidate contract

For each event, compute the set of items with positive one-step transition
score from the six recent seeds in:

1. the exact-timestamp transition graph; and
2. the five-minute-session transition graph.

Eligible negatives are the intersection of those two reachable sets with the
positive pre-train candidate pool, after removing the gold and every item in
the user's strict-past history. Select nine items uniformly by deterministic
hash order, then deterministically permute gold plus negatives.

Important controls:

- selection does not use one-step score magnitude, PPR score or gold score;
- gold reachability is not an eligibility condition;
- all nine negatives must have positive one-step evidence in both views;
- no popularity matching, semantic filter or post-outcome fallback is used.

This removes the previous zero-versus-nonzero shortcut for negatives while
keeping candidate count and LLM compatibility unchanged.

## 4. Frozen graph/scoring contract

Both views retain the Amazon/M1 parameters:

| Parameter | Value |
|---|---:|
| Recent unique seeds | 6 |
| Restart/stop probability | 0,15 |
| Monte Carlo walks/event | 50.000 |
| Maximum steps | 64 |
| Candidate tie-break | frozen candidate order |

For each view, score two graph-only arms on the identical candidates:

- one-step transition ranking;
- multi-hop terminal-state PPR ranking.

No alpha or local LLM score is involved in this headroom phase.

## 5. Gate and interpretation

Primary exact-view gate:

- mean `PPR NDCG@5 - one-step NDCG@5 >= +0,02`; and
- paired-bootstrap 95% CI lower bound `> 0`.

Apply the same fixed gate independently to the session view. Promotion logic:

| Exact | Session | Decision |
|---|---|---|
| Pass | Pass | Promote both views to a fresh primary protocol |
| Pass | Fail | Promote exact only |
| Fail | Pass | Promote session adaptation only; frozen exact multi-hop fails |
| Fail | Fail | Stop before any new LLM/GPU run |

Also report Hit@5, improved/worsened/unchanged events, gold reachability and a
non-deployable per-event best-of-one-step/PPR oracle. The oracle cannot trigger
promotion.

A pass demonstrates incremental multi-hop headroom under graph-hard negatives;
it does not yet establish a test-set gain. A failure means the sealed MovieLens
gain is principally transition-graph/coverage evidence rather than evidence
that deeper propagation is needed.

## 6. Smoke-first execution plan

1. Unit-test deterministic candidate construction on synthetic graphs.
2. Run a 50-user data adapter smoke without writing artifacts.
3. Run the exact full graph contract on 20 development events with 5.000
   walks; require deterministic replay and all negative reachability checks.
4. Materialize and hash the 200-event development cohort.
5. Score and hash both views without reading labels in ranking code.
6. Open development outcomes once, compute the frozen gate, document the
   result and either stop or draft a separate fresh-primary protocol.

Configuration: `configs/temporal_movielens32m/m5_graph_hard_headroom.yaml`.

## 7. Sealed development result

The fixed 500-user scan produced 494 feasible graph-hard events; six lacked
nine negatives in the exact/session reachable intersection. The first 200
feasible events were used. They are 200 distinct users with zero overlap with
the sealed M1 cohort. Reachable-intersection size was at least 60 items and had
a median of 7.518 items.

| View / graph-only arm | NDCG@5 | Hit@5 |
|---|---:|---:|
| Exact one-step | 0,593659 | 0,725 |
| Exact PPR | 0,613055 | 0,765 |
| Session one-step | **0,643045** | 0,755 |
| Session PPR | 0,607746 | 0,760 |

Exact PPR minus one-step:

- delta NDCG@5: **+0,019396**;
- paired-bootstrap 95% CI: **[-0,015394; +0,053643]**;
- improved / worsened / unchanged: 35 / 34 / 131;
- gate requires delta `>= +0,02` and CI lower `> 0`: **fail**.

Session PPR minus one-step:

- delta NDCG@5: **-0,035299**;
- paired-bootstrap 95% CI: **[-0,067050; -0,004538]**;
- improved / worsened / unchanged: 25 / 44 / 131;
- same gate: **fail**.

The preregistered decision is therefore `stop_before_llm`. No fresh primary
cohort was sampled and no self-hosted model/GPU run was started.

### Coverage audit

| View | Gold one-step | Gold PPR | Negative one-step | Negative PPR |
|---|---:|---:|---:|---:|
| Exact | 166/200 (83,0%) | 164/200 (82,0%) | 1.800/1.800 (100%) | 1.156/1.800 (64,22%) |
| Session 300s | 177/200 (88,5%) | 165/200 (82,5%) | 1.800/1.800 (100%) | 1.247/1.800 (69,28%) |

The candidate contract worked as intended: one-step could not win by merely
assigning zero to most negatives. Under this harder condition, a fixed PPR
policy did not provide reliable incremental depth gain.

### Oracle and interpretation

The non-deployable per-event best-of-one-step/PPR oracle reaches:

- exact: 0,668351 NDCG@5, `+0,074692` over exact one-step;
- session: 0,687003 NDCG@5, `+0,043957` over session one-step.

This oracle cannot change the failed gate, but it shows heterogeneous depth
utility: PPR helps some events and harms nearly as many. The sealed M1 result is
therefore best interpreted as strong transition-graph/coverage evidence, not
evidence that globally applying multi-hop PPR is necessary on MovieLens.

The only methodologically justified continuation inside the graph direction is
a separately preregistered **depth-routing** study: learn from development
labels whether an event should use one-step or PPR, using only gold-agnostic
graph structural features at inference. Hand-tuned thresholds or an oracle
selector are not acceptable substitutes.

That follow-up was executed with a frozen 12-feature, zero-threshold,
hyperparameter-free OLS contract. Five-fold out-of-fold routing failed in both
views: -0,000376 exact and -0,008650 session versus one-step. No final router
model was written. See
[MOVIELENS_DEPTH_ROUTER_PROTOCOL.md](MOVIELENS_DEPTH_ROUTER_PROTOCOL.md).

## 8. Compute audit

- Graph smoke: 8m52s, peak RSS 1.656.408 KiB, zero swap.
- Full prepare: 9m04s, peak RSS 1.656.904 KiB, zero swap.
- Full label-blind score: 8m01s, peak RSS 1.279.276 KiB, zero swap.
- Evaluation: 1,65s, peak RSS 24.596 KiB.
- Entire study: zero LLM requests and zero GPU use.

## 9. Sealed hashes

| Artifact | SHA256 |
|---|---|
| Config | `4bd912552d815b47ee733a075761685432e4024b6faec4f9784d9c5771fcd96b` |
| Graph smoke | `5352f9662ade628fb3a6211b8672cdd54fe5a371beb13b1c51989103dd963e97` |
| Prepared cohort | `76c26ab9f221c76b0238c86be7d01b6c1a63f204cb0035fd9d31c75974dd7bbc` |
| Prepare manifest | `bbd72b531dd6c758fc8397dee1e19e4e562980074855bf995763357be11d1be6` |
| Graph scores | `0cd09c083f19bbc79c64916dd5957493002431e8663f4e43d6696f5f9ca5c8d6` |
| Score manifest | `2d46e3bea149d7cfa9656422d7734c240aceaaa0f56feb3f86bc1a0f9f8eb689` |
| Metrics | `4186759288360bc4f71c7baaa366ef2faea0889e6e084c7895529687aeb1f9dc` |
| Evaluation manifest | `e3e7152186549fa458c6fb01d0ad2b9bfdcbde1bedcd3699a451bdaf42288c06` |

Smoke, cohort preparation and label-blind scoring ran from commit `46c72f3`;
the sealed evaluator including the preregistered oracle report ran from commit
`b037e93`. The final suite passed 42/42 tests.

Outcome artifacts are sealed. Do not tune candidate construction, walk depth,
restart probability, graph view or gate on these 200 development outcomes.
