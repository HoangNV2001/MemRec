# MovieLens 32M protocol — frozen cross-domain study

> Status: **complete and sealed**. The 500-event frozen transfer study passed
> both the primary exact-PPR and fixed secondary session-PPR gates. Outcomes
> were opened once after local/graph score hashes were locked; no test-time
> tuning was performed. M0 config SHA256 is
> `fb5e5356a2b693401f7fb5034238bbb3fb14c5888c7e64c3516de3fbe88b467f`;
> audit artifact SHA256 is
> `625c811ce2734fa9885742da84e74a319617f2c6bf55661f11375ca19154040e`.
> Frozen-transfer config SHA256 is
> `51d749dbb09ae866e2dd041f336daaa75eaa38b33078938c6d8f4f206431b79b`.

## 0. Execution status

| Phase | Status | Result |
|---|---|---|
| M0 50-row smoke | Pass | Four schemas parsed; no artifact written |
| M0 full audit | Pass | 32.000.204 ratings; integrity and feasibility gates pass |
| M1 shared core | Pass | 31 tests; Amazon golden outputs unchanged; 50-user adapter smoke pass |
| M2 cohort/graph smoke | Pass | Cohorts locked; small and full dual-view graph smoke pass |
| M3 local ranker | Complete | 1.000/1.000 success; 0 retry/parser/permutation repair |
| M4 score lock | Complete | 500 graph rows and all local artifacts hashed before labels |
| M4 evaluation | **Pass, sealed** | Exact-PPR +0,3663; session-PPR +0,3774 NDCG@5 |

## 1. Research question

This is not an exact replication of Amazon Books because the native semantic
context changes from reviews to `title + genres`. The study is:

> Cross-domain frozen-parameter adaptation of Temporal Transition-PPR from
> Amazon Books to MovieLens 32M.

The primary question is whether the Amazon graph/fusion parameters transfer to
a fixed MovieLens local ranker. The claim is limited to predicting the next
**rating event**. MovieLens timestamps do not establish when a movie was
watched, so the result must not be described as next-watch prediction.

## 2. Verified raw data

Source directory: `data/ml-32m/`. All official MD5 checks pass.

| File | Rows/content | MD5 status |
|---|---:|---|
| `ratings.csv` | 32.000.204 ratings | Pass |
| `movies.csv` | 87.585 movies | Pass |
| `tags.csv` | 2.000.072 tag applications | Pass |
| `links.csv` | external IDs | Pass |

Observed ratings data:

- 200.948 users and 84.432 rated movies; duplicate user–movie rating = 0.
- Rating range 0,5–5,0; 15.938.231 ratings (49,81%) are `>= 4,0`.
- Timeline: 1995-01-09 to 2023-10-13 UTC.
- Every rated movie has a native metadata row; 7.080 metadata rows use
  `(no genres listed)`.
- Tags and external links are excluded from the primary study.

## 3. Construct-validity audit

Rating timestamps are highly bursty:

| Consecutive interval | Share |
|---|---:|
| Same second | 16,55% |
| <= 10 seconds | 50,76% |
| <= 60 seconds | 83,05% |
| <= 5 minutes | 91,66% |
| <= 1 hour | 93,56% |
| <= 1 day | 95,47% |

Additionally, 22,68% of events belong to a same-second batch of size greater
than one. Median per-user fraction of intervals `<= 60s` is 90,86%.

Therefore exact timestamp ordering mainly captures rating-entry behavior, not
movie-consumption order. A naive target from a multi-item timestamp batch can
also turn a simultaneously rated item into a false negative.

### Frozen pre-outcome decision

Use only targets that are a **singleton five-minute session**:

- the target timestamp contains exactly one rating;
- gap from the preceding rating timestamp is greater than 300 seconds;
- gap to the following rating timestamp is greater than 300 seconds.

This is an outcome-independent data-quality rule. M0 feasibility is large:
9.848 test users and 82.516 events satisfy it together with the rating, history
and candidate-pool requirements below. Development has 9.389 eligible users
and 70.317 eligible events.

## 4. Temporal split

Keep the Amazon event-quantile split philosophy, with all equal timestamps on
the same side of a boundary:

| Partition | Rule | Approximate period | Events |
|---|---|---|---:|
| Train | `t < 1538551305` | 1995 to 2018-10-03 | 25.600.163 |
| Development | `1538551305 <= t < 1604605535` | 2018-10-03 to 2020-11-05 | 3.200.020 |
| Primary test | `t >= 1604605535` | 2020-11-05 to 2023-10-13 | 3.200.021 |

An event at `t` can read only interactions with timestamp `< t`. Exact graph
batches use equal timestamps. Five-minute sessions are built per user from
consecutive gaps `<= 300s`; a graph session is admitted only if its final
timestamp is strictly before the graph cutoff.

## 5. Frozen cohort and candidates

Primary cohort:

- 500 events from 500 users, selected by a recorded deterministic hash salt;
- one singleton-session target per user;
- target rating `>= 4,0`;
- at least five positive strict-past ratings;
- gold movie has a positive observation before the train cutoff;
- no user overlap with development/prompt-validation events.

Candidate protocol remains comparable to Amazon:

- one gold plus nine deterministic uniform negatives;
- pool = movies positively rated before the train cutoff (preliminary size:
  34.640);
- every negative is absent from the target user's strict-past history;
- ten distinct movies and deterministic candidate order;
- target-session items cannot become negatives; singleton targets make this
  constraint unambiguous.

The primary cohort size must not be changed after local or graph outputs are
inspected. If fewer than 500 eligible users survive the implemented audit,
stop and report the exact count instead of relaxing eligibility.

## 6. Frozen local ranker

Reuse the Amazon model contract:

- `Qwen/Qwen3-4B-Instruct-2507`;
- revision `cdbee75f17c01a7cc42f958dc650907174af0554`;
- bf16, greedy, seed `20260921`, tensor parallel 1;
- Stage-R returns at most five facets;
- listwise reranker returns a permutation A–J;
- identical structured-decoding and permutation-completion contract.

Only the dataset rendering changes:

- history: up to six recent strict-past ratings with movie title, genres and
  rating value;
- candidate memory: title and genres only;
- `(no genres listed)` is represented explicitly;
- tags, IMDb/TMDb text and future aggregates are forbidden.

Prompt adaptation must be finalized on development events using only
parse/schema/truncation checks, never NDCG or gold rank. Any content-level
prompt change creates a new run ID and requires a new smoke.

Primary budget: 500 events × two phases = 1.000 generations, plus a maximum
retry reserve of 100. A 20-event primary smoke uses the exact full contract and
its 40 successful generations must be reused by the full run.

## 7. Two pre-registered graph views

Both graph views use all rating values, the same candidates, seeds, local
rankings and target events.

### A. Exact-timestamp graph — primary frozen-transfer arm

This is the direct Amazon mapping. Per user, equal timestamps form unordered
batches; edges connect consecutive batches. Interpretation: rating-event
transition.

### B. Five-minute-session graph — construct-validity arm

Consecutive ratings separated by at most 300 seconds form one unordered
session; edges connect consecutive sessions. The 300-second threshold is fixed
from the raw burst audit before ranking outcomes are available.

For both views, freeze the Amazon parameters:

| Parameter | Value |
|---|---:|
| Recent unique seeds | 6 |
| Restart/stop probability | 0,15 |
| Monte Carlo walks/event | 50.000 |
| Maximum steps | 64 |
| Fusion alpha | 0,80 |
| Candidate normalization | within-candidate min–max |

No rating filter, semantic edge, genre similarity, hub penalty, path selector
or target-aware routing is allowed.

## 8. Hypotheses and decision rules

Primary comparison:

```text
frozen local ranker
vs
local + exact-timestamp Transition-PPR, alpha = 0.80
```

Primary success requires both:

- mean delta NDCG@5 `>= +0,03`;
- paired-bootstrap 95% CI lower bound `> 0`.

The session graph is a fixed secondary comparison, not a fallback used to
replace a failed primary. Interpret the pair as follows:

| Exact graph | Session graph | Conclusion |
|---|---|---|
| Pass | Pass | Robust cross-domain transfer |
| Pass | Fail | Gain is likely tied to rating-entry sequences |
| Fail | Pass | Session adaptation helps; frozen transfer failed |
| Fail | Fail | No MovieLens transfer evidence |

Report NDCG@5, Hit@5, paired CI, improved/worsened/unchanged counts, one-step
and PPR reachability for gold/negative slots, and graph-only rankings. Add
pre-registered descriptive buckets by historical burstiness (`<0,80`,
`0,80–0,95`, `>0,95` fraction of intervals `<=60s`) and target preceding gap
(`5m–1h`, `1h–1d`, `>1d`). Buckets cannot alter the primary decision.

One-step residual and both PPR arms must be scored and hashed before labels are
opened. Post-hoc best-of arms is oracle-only and non-deployable.

## 9. Implementation plan

### M0 — reproducible audit, CPU only

1. Add a MovieLens audit module with a 50-row smoke mode.
2. Verify schemas, MD5, counts, timestamps, burst/session distributions,
   metadata coverage and eligibility.
3. Write an audit artifact and decision manifest; hash both.

### M1 — shared core without changing sealed Amazon behavior

1. Add golden tests for current Amazon graph/PPR/fusion behavior.
2. Extract dataset-independent graph, walk, fusion and metric functions into a
   shared temporal module.
3. Keep Amazon entry points backward-compatible and require all existing tests
   and artifact-level regression checks to pass.
4. Add a MovieLens adapter for rating iteration, sessionization, history and
   item-memory rendering. Do not fork the graph algorithm.

### M2 — cohort lock and offline smoke

1. Materialize development and 500-event primary cohorts with independent
   deterministic salts.
2. Assert strict-past access, user disjointness, singleton sessions, candidate
   uniqueness, no history/gold collision and fixed split boundaries.
3. Hash config, raw inputs, cohort, candidates, prompts and model contract.
4. Run deterministic graph smoke on 20–30 events with 5.000 walks; require
   bit-for-bit replay for both graph views.

### M3 — LLM smoke and full local ranking

1. Follow `internal_docs/H100_RESOURCE_RULES.md`.
2. Dynamically select exactly one idle H100 inside the shared Slurm allocation.
3. Smoke 20 primary events; require 40/40 valid generations, no parser repair,
   no OOM and VRAM within the locked envelope.
4. Promote only with a valid smoke manifest; reuse smoke cache in the full
   1.000-generation run.
5. Exit/unload immediately and confirm VRAM returns to baseline.

### M4 — score lock and one-time evaluation

1. Produce 50.000-walk exact/session graph scores without reading labels in
   ranking code.
2. Lock local journal, graph scores and hashes.
3. Open outcomes once, compute all pre-registered comparisons and bootstrap
   intervals, then seal the result manifest.

## 10. Stop rules

- Any raw checksum/schema mismatch: stop before cohort creation.
- Fewer than 500 eligible, user-disjoint primary targets: stop; do not relax
  session/history/rating rules.
- Leakage, nondeterminism or cross-view candidate mismatch: stop before GPU.
- LLM smoke schema/OOM/hash failure: seal the run ID; fix under a new ID and
  smoke again.
- Primary failure: do not tune alpha, restart, seeds, depth, session threshold,
  prompt or rating filter on test labels.

## 11. Deferred studies

- Strict-past tags are a separate semantic-context adaptation.
- Exact/sparse PPR versus Monte Carlo can be evaluated after the primary.
- A 100-candidate study is not a small ablation: the current A–J listwise
  contract supports exactly ten candidates. It needs a separately designed
  retrieval/ranking protocol and must not be mixed into this primary result.
- TMDb/IMDb descriptions, popularity-matched negatives and learned routing
  remain out of scope for this sealed frozen-transfer study.

## 12. Locked execution artifacts

| Artifact | SHA256 |
|---|---|
| Prepared cohort | `7794a19a354de0a4bf4859f1795216b748ac029e9622e4093654feff8d2c169c` |
| Method lock | `258901956b4c3a91401407c71fbfa671fee8ed91974aff585b66ad7e60c90083` |
| Prepare manifest | `f4478962d4018ef7c986d95c88d913925a2b9fa3dba57288b48737ca44d1a1d2` |
| Full graph-smoke manifest | `a7d692895a0fcc80c72f93b2f6205c1ac63c1a2d6993e1ab1cb7505bd0448dbe` |

Prepared audit: 20 development events and 500 primary events from distinct
users, no cross-split user overlap, ten unique candidates/event, zero
candidate/history collision and six strict-past history rows/event. Cohort
preparation used zero LLM requests and no GPU.

Full graph smoke used 20 primary events and 5.000 walks/event/view. Exact view
has 23.718.412 temporal pairs and 28.533.332 source-to-group links; the
five-minute-session view has 2.210.221 pairs and 22.019.905 links. Nonzero
one-step/PPR candidate slots were 72/20 for exact and 119/19 for session out of
200 slots. Replay was deterministic; labels were not used. Peak RSS was
1.446.944 KiB with zero swap and no GPU.

The frozen MovieLens prompt contract renders only prior `title + genres +
rating` and candidate `title + genres`. Its CPU dry-run SHA256 is
`6eacf64600391323e4bab1d4eba09be7d05966eae5bbb9d9382e6707fe0e781b`;
20 development events produced 40 valid jobs without writing a journal.

## 13. Final sealed result

All rows below use the same 500 events, ten-candidate order and frozen
parameters. `Graph-only` is reported to expose how much signal is already in
the transition graph; it does not replace the preregistered residual-fusion
gate.

| Graph view / ranking arm | NDCG@5 | Hit@5 |
|---|---:|---:|
| Frozen local ranker | 0,471944 | 0,702 |
| Exact one-step, graph-only | **0,861449** | 0,950 |
| Exact one-step, residual | 0,848785 | 0,948 |
| Exact PPR, graph-only | 0,844459 | 0,936 |
| **Exact PPR, residual — primary** | **0,838235** | **0,952** |
| Session one-step, graph-only | **0,869054** | 0,960 |
| Session one-step, residual | 0,841716 | 0,946 |
| Session PPR, graph-only | 0,854884 | 0,948 |
| **Session PPR, residual — fixed secondary** | **0,849310** | **0,960** |

Primary exact-PPR versus local:

- delta NDCG@5: **+0,366290**;
- paired-bootstrap 95% CI: **[+0,330511; +0,400667]**;
- improved / worsened / unchanged: 310 / 36 / 154 events;
- changed rankings: 443/500;
- gate `delta >= +0,03` and CI lower `> 0`: **pass**.

Fixed secondary session-PPR versus local:

- delta NDCG@5: **+0,377365**;
- paired-bootstrap 95% CI: **[+0,343037; +0,411524]**;
- improved / worsened / unchanged: 314 / 27 / 159 events;
- changed rankings: 459/500;
- same gate: **pass**.

The preregistered conclusion matrix therefore yields
`robust_cross_domain_transfer`: both exact-timestamp and five-minute-session
Transition-PPR residual arms beat the frozen local ranker by the locked margin
with positive lower confidence bounds.

### Reachability and descriptive buckets

| View | Gold one-step | Gold PPR | Negative one-step | Negative PPR |
|---|---:|---:|---:|---:|
| Exact timestamp | 452/500 (90,4%) | 450/500 (90,0%) | 1.351/4.500 (30,02%) | 847/4.500 (18,82%) |
| Session 300s | 477/500 (95,4%) | 460/500 (92,0%) | 2.453/4.500 (54,51%) | 987/4.500 (21,93%) |

Both PPR residual arms have positive descriptive deltas in every preregistered
historical-burstiness and preceding-gap bucket. These buckets are descriptive,
not additional hypothesis tests. The full per-bucket numbers remain in the
sealed metrics artifact.

### Interpretation boundary

This result is strong evidence that the **transition-graph signal** transfers
from Amazon Books to the frozen MovieLens rating-event protocol. It is not
evidence that deeper propagation always dominates one hop:

- exact one-step graph-only is +0,0170 NDCG@5 above exact PPR graph-only;
- session one-step graph-only is +0,0142 above session PPR graph-only;
- exact one-step residual is +0,0105 above exact PPR residual, while session
  PPR residual is only +0,0076 above session one-step residual;
- graph-only PPR is also slightly above residual PPR in both views, so the
  frozen local fusion is not the source of the large gain on MovieLens.

The task has one positive and nine uniformly sampled negatives. Gold graph
reachability is much higher than negative reachability, making this candidate
set substantially graph-separable. The numbers are valid for the frozen
protocol and useful as cross-domain evidence, but must not be presented as a
production-scale retrieval result or as proof that multi-hop itself causes the
entire improvement. A harder candidate protocol requires a new, separately
preregistered study.

## 14. Compute and request audit

- LLM smoke: 40 physical requests; 17.272 input + 1.767 output tokens.
- Full invocation: reused all smoke calls and issued 960 new requests; 415.165
  input + 41.733 output tokens.
- Total: **1.000 physical requests**, 432.437 input + 43.500 output = 475.937
  tokens; retry/parser/permutation repair = **0/0/0**.
- Model: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, bf16, greedy, TP=1.
- Peak VRAM: smoke 7,702 GiB; full 7,707 GiB on exactly one H100. The process
  exited and the selected GPU returned to 1 MiB baseline.
- Full dual-view graph score: 13m49s CPU, peak RSS 1.447.452 KiB, zero swap,
  zero GPU and zero LLM request.
- Evaluation: 15,7s CPU, zero GPU and zero LLM request.

## 15. Final artifact hashes

| Artifact | SHA256 |
|---|---|
| Frozen config | `51d749dbb09ae866e2dd041f336daaa75eaa38b33078938c6d8f4f206431b79b` |
| Prepared cohort | `7794a19a354de0a4bf4859f1795216b748ac029e9622e4093654feff8d2c169c` |
| Method lock | `258901956b4c3a91401407c71fbfa671fee8ed91974aff585b66ad7e60c90083` |
| Local calls, 1.000 rows | `a6cbd2afcd78e7d0923a121ec1300965d0300b851f25e92c31f486c53848131f` |
| Local manifest | `5f0a49110897f48ba6f92ad09aa3474f8e2496f6e93790898d3694caccf76e14` |
| Graph scores, 500 rows | `cf2ba269f6140080a5498920e140ad9c66e9486ef2e5a54a66c1babe2c8cffdb` |
| Pre-outcome score lock | `5a0938a99c236d4dcbb7576e0931ccea21450d80d2465a0bf13d04727fac7afc` |
| Metrics | `8241dc975da4a8f66954fac79a3b1613b9bc6fe2ab7073ffb5df09c2c3affade` |
| Evaluation manifest | `7f9fca95f533e54891b0727c25f365b6cf0170826e0fb5960f8e5d7813a11058` |

Code provenance: self-hosted inference ran from commit `76f8927`; full graph
scoring, score lock and sealed evaluation ran from commit `1122a73`. The final
local test suite passed 39/39 tests before documentation was sealed.

Outcome artifacts are sealed. Do not retune alpha, restart probability, walk
depth, graph view, session threshold, prompt or candidates on these 500 labels.
