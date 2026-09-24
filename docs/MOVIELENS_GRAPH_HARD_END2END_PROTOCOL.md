# MovieLens graph-hard end-to-end MemRec protocol

**Protocol date:** 2026-09-23

**Status:** final result sealed; primary gate passed; no further test tuning

**Task boundary:** candidate-set ranking, matching the MemRec paper. “End-to-end”
here means local semantic ranking plus graph augmentation on the same supplied
candidate set. It does not mean full-catalog item retrieval.

## 1. Research question

The frozen MovieLens transfer used one positive plus nine uniform negatives and
showed a large transition-graph gain, but gold reachability was much higher than
negative reachability. M5 removed that shortcut and found that PPR did not
reliably outperform one-step graph-only ranking. It did not run the local LLM.

The missing question is therefore:

> Does direct temporal transition evidence improve the frozen local LLM when
> every negative already has valid one-step graph evidence?

One-step transition is the primary method. PPR is retained only as a frozen
secondary propagation variant.

## 2. Compatibility with the MemRec benchmark

The MemRec formulation receives a candidate set `C` and ranks within `C`.
Stage-R retrieves collaborative memories, not catalog items; Stage-ReRank scores
only supplied candidates. Its main benchmark uses `N=10`, with `H@1/3/5` and
`NDCG@3/5`; the paper also reports an `N=20` robustness study.

M7 keeps the same task family:

- exactly ten candidates;
- one held-out positive target;
- ranking metrics at K below candidate-set size;
- identical candidates for graph, LLM and classical baselines.

`Hit@10` is not reported because the target is included by construction.

## 3. No-tuning contract

- No manual hyperparameter, prompt, graph, feature, threshold or seed search.
- M5/M6 labels may not be reused for model selection.
- Exact-timestamp is the primary graph view; session-300s is fixed secondary.
- `alpha=0.80`, model revision, prompt, six seeds, PPR restart/depth/walks and
  candidate count transfer unchanged.
- BPR-MF and SASRec each use exactly one preregistered configuration and seed.
  Deterministic validation early stopping is allowed; no config is compared.
- All score files are hashed before M7 outcomes are evaluated once.
- A secondary arm cannot replace a failed primary arm.

## 4. Fresh user-disjoint cohort

### Exclusions

Exclude:

1. all 20 development and 500 primary users from sealed M1;
2. all 500 users in the fixed M5 development scan, including the 300 not
   selected into the 200-event M5 cohort;
3. M6 automatically, because it reused M5 events.

The M1 prepared artifact, M5 config and M5 prepared artifact are checksum
locked in the M7 config. The M5 fixed scan is deterministically replayed from
its sealed config solely to recover the full exclusion set.

### Targets

- Use positive singleton-five-minute-session events from the MovieLens test
  interval, after the global validation cutoff.
- Require rating `>=4`, at least five earlier positive events and six recent
  strict-past events for the prompt/graph seeds.
- Select one target per user by fixed hash.
- Sort remaining eligible users with a new salt and inspect exactly the first
  1.000 users.
- Keep the first 500 graph-hard-feasible events. Do not extend the scan if fewer
  than 500 pass.

### Candidates

For each event, eligible negatives are positive pre-train items that:

- are absent from the user's strict-past history;
- have positive one-step transition score from the six seeds in both exact and
  session-300s graph views.

Select nine negatives uniformly by deterministic hash, then deterministically
permute them with the gold. Selection cannot use transition magnitude, PPR,
genre, popularity, LLM score or gold reachability. Gold reachability is not an
eligibility condition.

Both graph snapshots contain only events before the global validation cutoff.

## 5. Ranking arms

Primary exact view:

1. frozen local LLM;
2. exact one-step graph-only / first-order Markov;
3. **local + exact one-step residual — primary treatment**;
4. exact PPR graph-only;
5. local + exact PPR residual.

Fixed secondary session-300s view repeats arms 2–5. Classical comparison arms
on the identical candidates are:

- global MostPopular;
- BPR-MF;
- SASRec.

The paper's LightGCN baseline remains optional and is not part of the M7 score
lock. It cannot be added after outcomes are opened.

## 6. Frozen scoring and models

### Local and graph

- Qwen3-4B-Instruct-2507 at revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`;
- bf16, greedy, temperature 0, one visible GPU;
- same MovieLens Stage-R/listwise prompt and permutation parser as M1;
- one-step/PPR score normalized within the ten candidates;
- reciprocal-log local score plus `0.80 * normalized_graph_score`;
- PPR: 50.000 walks, restart 0,15, maximum 64 steps.

### Classical baselines

All positives have rating `>=4` and timestamp before the validation cutoff.
MostPopular uses global positive counts. BPR-MF and SASRec use the single config
recorded in `m7_graph_hard_end2end.yaml`. Their only checkpoint selection is
deterministic early stopping on a separate validation protocol with 100 fixed
sampled candidates. Every epoch and selected checkpoint must be recorded; the
test candidates cannot drive training or stopping.

## 7. Primary hypothesis and gate

The sole primary comparison is:

```text
Local + exact one-step residual  vs  Local
```

Pass requires both:

- mean delta NDCG@5 `>= +0,03`;
- paired-bootstrap 95% CI lower bound `> 0`.

Report Hit@5, every preregistered arm, improved/worsened/unchanged event counts,
coverage and paired confidence intervals. Classical baselines answer whether
the local LLM is weak; they do not alter the primary gate.

## 8. Smoke-first and outcome seal

1. Unit-test exclusion, candidate construction, rankings and score locks.
2. Run a 50-user adapter smoke without artifacts.
3. Replay the fixed exclusions and run candidate/dual-graph smoke on 20 events.
4. Materialize/hash 500 events and the method lock.
5. Run graph score smoke on 20 events, then full blind graph scoring.
6. Run each baseline on a 20-event smoke before full training/scoring.
7. Run the local LLM on a 20-event smoke; full run must reuse all smoke calls.
8. Hash config, cohort, graph, baseline and LLM artifacts into one score lock.
9. Open outcomes once and write metrics/evaluation manifest.

Any failed smoke produces a new run ID/config. No partial outputs from changed
contracts may be merged.

## 9. Compute contract

- CPU-only phases use no LLM and no GPU.
- GPU phases use exactly one H100 at a time; BPR, SASRec and LLM serving are
  sequential, never concurrent.
- Each task unloads immediately and records the post-run VRAM baseline.
- The 500-event local run has 1.000 primary generations plus a fixed reserve of
  100 retries; smoke calls are reused.
- All cluster work follows `internal_docs/H100_RESOURCE_RULES.md`.

## 10. Decision boundary

- Pass: transition evidence is complementary even under structurally plausible
  negatives; stop method expansion and write the thesis.
- Positive delta but failed gate: report uncertainty; do not tune rescue rules.
- No gain: earlier effect is largely explained by candidate reachability;
  narrow the claim and stop method expansion.
- SASRec/BPR-MF stronger: do not claim SOTA; report graph/LLM mechanism evidence
  under the common MemRec candidate-ranking setup.

## 11. Execution record

### Offline validation

- Full local suite after the final validation-sampler fix: **61/61 tests pass**.
- Adapter smoke: 50 users, 7.927 ratings and 3.690 rated movies; pass.
- Local prompt/schema dry contract: 20 events, 40 jobs, zero request; pass.
- Prompt contract SHA256:
  `45baf5ded0a87c02fb5561a080122ca5238a007df2330f278161d385f539b2a8`.

### Candidate smoke and cohort lock

The 20-event candidate smoke passed with:

- 1.020 excluded users: 520 from M1 and all 500 in the fixed M5 scan;
- 9.143 eligible fresh primary users after exclusion;
- 992/1.000 fixed-scan users graph-hard feasible;
- 180/180 negative slots with positive one-step evidence in both views;
- zero GPU, LLM request or outcome evaluation.

The full prepare selected 500 events without extending the scan:

| Artifact | SHA256 |
|---|---|
| Frozen config | `5311a8008a5ad2bf775b0f411509e328e24fd479f0d77e50a6ad9c989a5f1e2e` |
| Candidate smoke | `2116b38b17294c7325090ec65972cdd4c4e03209aa8739b15038adac96c9aeb1` |
| Prepared cohort | `c19a64f18e7b340b5e278840d691b2dbb6d5f90947e10e5dbeb2c263d04c8a2e` |
| Method lock | `eddb551cd0c2d2ccb4c0ccd464413e87d61f537b5b14bc7ad08c526f786e9ca1` |

The prepare manifest records `manual_tuning_performed=false`,
`recommendation_outcomes_evaluated=false`, zero GPU and zero LLM request.

### Blind graph scoring

Both the 50-user scorer smoke and the full-graph 20-event smoke passed.
Deterministic replay matched. Full scoring then produced 500/500 rows:

| View | One-step nonzero slots | PPR nonzero slots |
|---|---:|---:|
| Exact | 4.961/5.000 | 2.979/5.000 |
| Session-300s | 4.987/5.000 | 3.177/5.000 |

Graph-score JSONL SHA256:
`d0ac0831fc975de02612a6616798195dcabb8ba973bd74bed1ebcc402344e72e`.

These are coverage counts, not recommendation metrics. No gold label was passed
to graph scoring and no NDCG/Hit outcome has been computed.

### Baseline and evaluation implementation

The locked baseline runner now includes:

- global MostPopular counts from positives strictly before the validation
  cutoff;
- 64-dimensional BPR-MF with pairwise negative sampling, the single frozen
  optimizer/seed/config and automatic validation early stopping;
- 64-dimensional, two-block SASRec with causal masking, maximum history 50 and
  the single frozen optimizer/seed/config;
- refit for the automatically selected epoch count on all positives strictly
  before the validation cutoff, then blind scoring of the 500 fixed candidate
  sets;
- a 20-event real-GPU smoke that trains one bounded epoch and verifies all 400
  BPR/SASRec candidate scores are finite before promotion.

The separate M7 evaluator reports H@1/3/5 and NDCG@3/5 for all preregistered
arms. It hard-codes `local + exact one-step residual` as the sole primary arm
and refuses to read outcomes until cohort, graph scores, baseline scores,
checkpoints, local calls and manifests have all passed blind validation and
been SHA256-locked.

No hyperparameter search or manual tuning was added. CPU smoke and the full
local suite pass; no recommendation label was accessed by these checks.

### GPU execution and baseline lock

The authorized replacement allocation was Slurm job `17272` on `worker-7`.
Every model step dynamically selected the lowest-utilization/lowest-memory GPU,
exported exactly one `CUDA_VISIBLE_DEVICES` entry and ran with TP=1. The final
baseline and LLM runs used source commit
`93e28b3adfb1ee0ef712d22c952af2f536d82683`.

The final 20-event baseline smoke produced 400/400 finite BPR/SASRec candidate
scores, used no target labels and peaked at 0,173 GiB VRAM. The blind full run
then scored 500/500 events and peaked at 0,451 GiB. Automatic validation chose:

| Baseline | Validation events | Selected epoch | Validation NDCG@10 |
|---|---:|---:|---:|
| BPR-MF | 9.389 | 20 | 0,338131 |
| SASRec | 9.389 | 17 | 0,673287 |

No hyperparameter/config/seed was compared manually. Both selected models were
refit for their locked epoch counts on positives strictly before the test
cutoff. The final score file contains exactly 500 blind rows.

### Local LLM execution

The final self-hosted smoke passed on 20 events:

- 40/40 physical requests and successful keys;
- zero retry, JSON/schema repair or permutation repair;
- 19.068 total tokens;
- 7,700 GiB peak VRAM.

The full run reused all 40 smoke calls and issued 960 new calls. Across smoke
and full, the sealed run used exactly 1.000 physical requests, 434.912 input
tokens and 43.463 output tokens, or 478.375 tokens total. It had zero retry and
zero JSON/schema repair. One ranking required the preregistered deterministic
permutation-completion repair, below the locked cap of five. Full peak VRAM was
7,706 GiB. The full invocation ran from 13:50 to 15:26 local time; immediately
after exit all four H100s reported 1 MiB used.

### Operational failure audit

Pre-outcome failures did not produce or merge scores:

- two baseline attempts stopped before model creation because PyTorch 2.10
  required an initialized CUDA context for peak-memory telemetry;
- one baseline full attempt stopped before training because the independent
  validation sampler incorrectly called the ten-item prompt sampler for its
  locked 100-item validation set; a general deterministic sampler and unit
  test were added, the old smoke was invalidated, and a fresh smoke passed;
- the first LLM attempt stopped before loading the model or writing a Journal
  because the expected Slurm job environment guard was absent.

The outer overlapping `srun` wrapper sometimes reported exit code 1 after a
successful child workload and post-run snapshot. Promotion used the separately
captured workload status, immutable manifest and independent artifact
validator; all successful child workloads returned status 0. No outcome was
opened during these fixes and the scientific config remained unchanged.

## 12. Final sealed result

The combined score lock validated 500 graph rows, 500 baseline rows and 1.000
successful LLM keys before accessing `gold_item_id`. Outcome evaluation then
ran once. All values below use the identical ten candidates for each event.

| Ranking arm | H@1 | H@3 | H@5 | NDCG@3 | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Local LLM | 0,204 | 0,480 | 0,666 | 0,361378 | 0,436926 |
| MostPopular | 0,498 | 0,772 | 0,862 | 0,654378 | 0,691561 |
| BPR-MF | 0,384 | 0,582 | 0,696 | 0,499497 | 0,547455 |
| **SASRec** | **0,660** | **0,894** | **0,960** | **0,800044** | **0,827329** |
| Exact one-step, graph-only | 0,546 | 0,780 | 0,866 | 0,682901 | 0,718712 |
| **Exact one-step residual — primary** | **0,528** | **0,754** | **0,854** | **0,658283** | **0,699773** |
| Exact PPR, graph-only | 0,528 | 0,778 | 0,848 | 0,675258 | 0,704266 |
| Exact PPR residual | 0,518 | 0,750 | 0,864 | 0,650235 | 0,697317 |
| Session-300s one-step, graph-only | 0,596 | 0,806 | 0,878 | 0,718806 | 0,748939 |
| Session-300s one-step residual | 0,554 | 0,798 | 0,892 | 0,696425 | 0,735243 |
| Session-300s PPR, graph-only | 0,534 | 0,776 | 0,862 | 0,674378 | 0,709663 |
| Session-300s PPR residual | 0,516 | 0,766 | 0,866 | 0,660116 | 0,701693 |

### Primary gate

`Local + exact one-step residual` versus `Local`:

- delta NDCG@5: **+0,262847**;
- paired-bootstrap 95% CI: **[+0,226284; +0,298401]**;
- improved/worsened/unchanged events: **256/83/161**;
- ranking changed on 477/500 events;
- preregistered gate (`delta >= 0,03` and CI lower bound `>0`): **PASS**.

This is strong evidence that direct transition magnitude improves the frozen
LLM even after eliminating the easy “gold reachable, negative unreachable”
shortcut. Every one of 4.500 negative slots has positive one-step evidence in
both views. Gold one-step coverage is actually lower: 461/500 (92,2%) for exact
and 487/500 (97,4%) for session-300s.

### Interpretation boundary

The result supports **transition-graph augmentation**, not a SOTA claim:

- exact one-step graph-only is 0,018940 NDCG@5 above the primary fusion;
- session-300s one-step graph-only is the best graph arm at 0,748939;
- direct one-step graph-only exceeds PPR graph-only by 0,014446 exact and
  0,039276 session, so deeper fixed propagation is not supported;
- MostPopular reaches 0,691561, only 0,008212 below the primary fusion;
- SASRec is strongest overall and exceeds the primary fusion by 0,127556.

Therefore the thesis may claim that a temporal transition graph supplies a
large, statistically reliable complementary signal to the frozen LLM under
structurally hard candidates. It must also state that the LLM baseline is weak,
that direct graph ranking slightly beats the frozen residual fusion, and that a
standard sequential recommender remains substantially stronger. No further
alpha, prompt, graph, candidate or model tuning is allowed on these labels.

## 13. Final artifact hashes

| Artifact | SHA256 |
|---|---|
| Frozen config | `5311a8008a5ad2bf775b0f411509e328e24fd479f0d77e50a6ad9c989a5f1e2e` |
| Prepared cohort | `c19a64f18e7b340b5e278840d691b2dbb6d5f90947e10e5dbeb2c263d04c8a2e` |
| Method lock | `eddb551cd0c2d2ccb4c0ccd464413e87d61f537b5b14bc7ad08c526f786e9ca1` |
| Graph scores | `d0ac0831fc975de02612a6616798195dcabb8ba973bd74bed1ebcc402344e72e` |
| Baseline smoke | `871e1979c07dbebca065645cb80fdd2f192f1706166621d9468feb32723a9840` |
| Baseline scores | `e57f216a7a6186c2a3841d53a36eb90bd3c246226a5b362478c01a8173d4b4ee` |
| Baseline manifest | `2b41c9f1095cdbfd8642d1ea43475bc37f1d0af6e4c20684e3420d77c9021154` |
| BPR-MF checkpoint | `549ac0340000c11370b4a843523add124ccb2bfa7a1cc46323e51272ee6fae6f` |
| SASRec checkpoint | `08eeb5aaf958cc51153599b8d2d5387e3b671123d6b279c1daeb0eedc7d61848` |
| Local smoke | `92b1f47564ca0eb60d20ebde919426f515685a39098973cb482f4f6a291e4813` |
| Local attempts | `70eb11b610656dbc79f5eb5ca72a0825178bc0faf6242309ed5a9716bf7c176b` |
| Local calls | `7551d630f40896e5ae3a99896f54003df5f97b088d85cdc4b7de7529b58303c6` |
| Local manifest | `257fc1a8cf42ca69a8e1fe111e9e56df7b9eb47deb7af3252645b251b5719a9f` |
| Pre-outcome score lock | `92b1025d7fb8e4604c0937166772c9e9869d81c5b14ccf364c3654f399d0f460` |
| Metrics | `66df8e555e0a507d9c0f68d6ee875a9bfc6fe255e40f4995795c988b1bf69a1b` |
| Evaluation manifest | `ec49efc8ce5206fa62a5983aa1fb40ef63cf3b361b45dfba41557477122bc26b` |

Outcome artifacts are sealed. The next work is thesis packaging and optional
analysis-only mechanism visualization; it is not another method-search loop.
