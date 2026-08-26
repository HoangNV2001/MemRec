# AMAZON_BOOKS_2014_TEMPORAL_PLAN.md — Causal Item-Memory Pilot

> **Dataset:** Kaggle Amazon Books Reviews (`Books_rating.csv` +
> `books_data.csv`), a repackage of the older Amazon reviews collection. This
> is a distinct temporal protocol, not a rerun of InstructRec MH2/CE1.
>
> **Status (2026-08-26):** P0/P1 passed. P2-v1 was technically invalid; the
> independently journalled P2-v2 smoke-first rerun completed and **failed its
> preregistered oracle headroom gate**.

## Question

> With a real global review-time field, can candidate-blind, item-only 2-hop
> propagation from past review packets make future-item ranking better than
> local/no-propagation and immediate-neighbor propagation?

The Kaggle review field is `review/time`; P0 must establish its actual range
and tie structure before any causal claim.  It is not assumed to offer a total
event order. The protocol uses **strict-past timestamp batches**: an event at
time `t` sees only state made from timestamps `< t`; all events with `t` are
simultaneous and cannot propagate into each other.

## P0 — data/temporal audit (0 LLM calls)

`python -m src.temporal_books.p0_audit --config configs/temporal_amazon_books_2014/pilot.yaml --force`

- Streams the 2.7 GB review CSV; validates columns, timestamp parse/range/ties,
  drops rows without `Id`/`User_id`/valid time (never imputes anonymous users),
  derives absolute 80/10/10 time cuts, and hashes inputs.
- Audits metadata separately. Its only common field is title, not `Id`; P0
  permits metadata only for an exact title match and reports its coverage. No
  fuzzy/normalized title join is permitted.
- Pass condition: valid timestamps permit strict-past *batched* replay. It does
  not claim an arbitrary within-day event ordering.

**Materialized result.** 2,438,194/3,000,000 rows (81.27%) have usable
`Id`/`User_id`/integer timestamp; 561,787 userless rows and 19 invalid-time rows
are excluded, never imputed. Time spans 1996-08-17 through 2013-03-04 with
5,736 distinct timestamps. Nearly all usable events tie (2,438,160); the
claim is therefore daily **strict-past batch** replay, not a strict event order.
The fixed cutoffs are train `< 2011-10-30`, validation `[2011-10-30,
2012-11-21)`, then test: 1,950,481 / 242,896 / 244,817 usable events.

Metadata has 212,403 nonempty, exact-unique titles. Exact title matching covers
2,437,999 usable review rows (99.992%); 195 usable reviews have no title, and
zero nonempty review titles fail the exact match. P2 may use metadata only under
this exact-match rule.

## P1 — source packet/route feasibility (0 LLM calls)

`python -m src.temporal_books.p1_preflight --config configs/temporal_amazon_books_2014/pilot.yaml --force`

1. User IDs are hash-sampled (`blake2b % 48 == 0`), then only train events are
   retained. This yields a deterministic pilot subgraph without selecting users
   by future activity.
2. Up to 560 source users with at least five train events are selected by a
   second stable hash. One potential packet is their last training event; every
   route from packet time `t` reads graph events strictly before `t`.
3. Route only `source -> up to 3 anchors -> up to 16 peers -> up to 16 remote
   items`. Candidate lists/gold labels are absent until the ledger closes.
4. Report source-only coverage for all novel validation targets and for 100
   deterministic target events. Evaluation targets must also exist in the
   global pre-cutoff item catalog; this standard evaluation filter is applied
   before route coverage and never uses whether a target is reachable. Admit P2
   only when all 560 packet sources exist, fixed-cohort support>=1 coverage is
   at least 20%, and support>=2 at least 10%.

P1 emits source packet inputs with six reviews max, summary <=240 characters and
review body <=480 characters. These are not sent anywhere in P1.

**Materialized result.** The final deterministic 1/48 hash sample contains
21,182 observed users (17,037 with train history; 1,282 with >=5 train events),
and materializes all 560 source packets plus 4,357 endpoint items. The source
ledger is candidate-blind and was complete before reading validation labels.

| Fixed 100-event validation cohort | Covered gold | Coverage |
|---|---:|---:|
| Independent source support >=1 | 24 / 100 | 24.0% |
| Independent source support >=2 | 17 / 100 | 17.0% |
| Independent source support >=3 | 12 / 100 | 12.0% |

The gate requires 560 packet sources, support>=1 >=20%, and support>=2 >=10%;
therefore P1 **passes**. The corresponding all-target (118 event) coverage is
22.88% / 15.25% / 10.17%. An earlier 1/128 hash sample was only 7,915 users,
476 source packets and 43 evaluation targets, outside the pre-stated 10K–30K
pilot-cohort intent. It was replaced by 1/48 before the final feasibility
decision; no LLM call or routing/gate relaxation occurred in either run.

## P2 — only if P1 passes: fixed 1,000-request pilot

| Use | Requests | Rule |
|---|---:|---|
| Semantic Stage-W packet generation | 560 | One reusable structured packet/source; use the P1 truncated history only. |
| Stage-R | 100 | One frozen local user-history preference profile/evaluation event, shared across arms. |
| LLM rerank | 300 | 100 fixed events × 3 arms: local/no-prop, 1-hop, 2-hop item overlay. |
| Retry reserve | 40 | Identical request key/prompt only; never a new sample or tuning attempt. |
| **Maximum** | **1,000** | Abort rather than exceed. |

Stage-R is intentionally fixed across the three P2 arms. It is a local
user-history profile, not a new dynamic retrieval graph. The experiment isolates
**candidate item-memory enrichment** at the reranker input. Every evaluation
event gets one candidate-blind set of 10 items, sampled only from items observed
before its timestamp. Temperature, model deployment and schema are fixed; no
independent repeat pass fits this budget.

**Prepared-artifact audit (0 API calls).** `p2_prepared.json` fixes 100 events,
560 packet sources, and ten unique candidates/event (gold exactly once). No
history review is at or after its target timestamp and no gold is in that
history. The actual direct-anchor 1-hop overlay is sparse under the fixed
candidate sets: it affects 13 candidate slots in 12 events (including 3 gold
items). The support>=2 two-hop pool reaches 17 gold endpoints. Thus the 2-hop
arm is explicitly a **target-aware, non-deployable upper bound**: after the
candidate set is fixed, it may attach up to two routed source packets only to
the labelled gold endpoint. It measures semantic headroom of the existing
routed pool; it is not evidence that a candidate-blind router has been built.

P2 gate: bounded oracle 2-hop must achieve at least `+0.05` NDCG@5 with a
positive lower bootstrap CI before any buffer/router implementation or locked
test. A failure closes this dataset/mechanism path without spending more API
budget.

### P2 execution incident — no result

The initial run journalled 768 request attempts: all 560 semantic packets and
all 100 shared Stage-R profiles succeeded; 10 rerank responses parsed, but 90
rerank responses were truncated while producing the required free-text
`rationale` under the configured 180-token output cap. They consequently raised
`JSONDecodeError` and contain no usable ranking. The process was stopped before
it queued the remaining reranks or spent the retry reserve. No metric, gate, or
headroom conclusion may be calculated from the 10 parsed ranks.

The runner now uses a ranking-only strict JSON schema and batch-level fail-fast
submission, so a future schema failure costs at most one worker batch plus its
identical retries rather than a large queued phase. That correction changes the
ranker output contract; therefore the old journal cannot be resumed or mixed
with it. A clean rerun requires explicit authorization for a **new** budget and
run identifier; the original 1,000-request P2 allocation must not be exceeded.

### P2-v2 smoke-first contract

P2-v2 has its own prepared artifact, journal, and run ID. Its full command is
programmatically blocked until a deterministic smoke run succeeds. The smoke
cohort has 20 future events; to make its rerank prompts identical to the later
full run, it also generates all 46 source packets reachable from those events
(rather than an arbitrary packet-only sample). It therefore consumes 126 calls:
46 packet + 20 Stage-R + 60 rerank. These outputs are cached under the same keys
and are reused in the 100-event run, so the complete v2 primary total remains
560 + 100 + 300 = 960, plus the existing 40-call identical-retry reserve.

**Smoke result.** The locked cohort completed 126/126 calls successfully: 46
packet, 20 Stage-R, and 60 rerank calls, with zero retries. Its manifest locks
the config/prepared hashes and every cache key. The remaining full run may send
only 514 packet + 80 Stage-R + 240 rerank = 834 primary requests.

### P2-v2 result — headroom gate fails

P2-v2 completed all 100 event × 3 arm rankings with 960 primary requests and
one identical retry (961 attempts, below the 1,000 cap). The retry followed one
invalid duplicate/missing ranking label and then parsed successfully; all 300
final rankings are present. The 10,000-resample paired bootstrap is deterministic
(seed `20260826`) and is recorded in `p2_v2_analysis.json`.

| Arm | NDCG@5 | Δ vs local | 95% paired bootstrap CI | H@5 | Interpretation |
|---|---:|---:|---:|---:|---|
| local | 0.6474 | reference | — | 0.84 | Base candidate memory. |
| direct 1-hop | 0.6353 | −0.0121 | [−0.0432, +0.0179] | 0.84 | No benefit; only 13 candidate slots (3 gold) actually receive a direct overlay. |
| oracle 2-hop | 0.6799 | **+0.0324** | **[+0.0073, +0.0592]** | 0.87 | Positive but target-aware, non-deployable upper bound; only 17 gold endpoints can receive it. |

The oracle lower CI is positive, but its point gain is **below the pre-stated
`+0.05` NDCG@5 admission threshold**. Therefore P2-v2 fails: no candidate-blind
router, buffer implementation, selector tuning, or locked test is admitted on
this dataset/mechanism. The positive but small oracle value is evidence that the
routed pool has limited semantic signal, not evidence for a deployable 2-hop
method.
