# MovieLens graph-hard multi-hop headroom protocol

> Status: protocol frozen before graph-hard development outcomes. This is a
> new study motivated by the sealed MovieLens 32M result; it does not retune or
> reopen the prior 500-event primary cohort.

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
