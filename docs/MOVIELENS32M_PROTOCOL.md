# MovieLens 32M protocol — frozen cross-domain study

> Status: protocol accepted and frozen; **M0 complete / pass**. No cohort has
> been sampled, no recommendation outcome has been evaluated and no LLM/GPU
> request has run. M0 config SHA256 is
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
| M2 cohort/graph smoke | Not started | No cohort materialized |
| M3 local ranker | Not started | 0 LLM requests; 0 GPU use |
| M4 evaluation | Not started | Labels not evaluated |

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
- TMDb/IMDb descriptions, popularity-matched negatives and learned routing are
  out of scope until the frozen transfer is sealed.
