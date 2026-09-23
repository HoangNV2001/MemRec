# MovieLens learned graph-depth router protocol

> Status: **complete, sealed, stop depth routing**. Both deterministic
> out-of-fold gates failed. No feature, model, threshold or hyperparameter was
> changed after outcomes; no final model, fresh cohort, LLM request or GPU run
> was created.

## 0. Execution status

| Phase | Status | Result |
|---|---|---|
| Unit tests | Pass | Full suite 45/45 |
| Feature smoke | Pass | 20 events / 40 rows; deterministic and label-blind |
| Full feature lock | Pass | 400 rows hashed before router evaluation |
| Five-fold OOF evaluation | **Fail / sealed** | Exact -0,0004; session -0,0086 vs one-step |
| Final model | Not written | Both promotion gates failed |
| Fresh test / LLM / GPU | Not started | Stop rule applied |

## 1. Research question

On graph-hard development candidates, global PPR failed to beat one-step, but
the non-deployable per-event oracle retained +0,0747 exact and +0,0440 session
NDCG@5 headroom. The new question is:

> Can a learned, gold-agnostic graph-structure router predict when to use PPR
> instead of one-step without hand-tuned rules?

This is a CPU-only cross-validated development gate. It reuses the sealed M5
200-event cohort and score matrix. It does not issue LLM requests, use GPU or
touch a fresh primary cohort.

## 2. Absolute no-tuning contract

The following are locked before router outcomes:

- exactly 12 features listed below; no addition/removal after evaluation;
- one independent model for each preregistered graph view;
- deterministic five-fold split by hash of event key;
- ordinary least squares solved once with `numpy.linalg.lstsq`;
- train-fold z-standardization, with constant columns mapped to zero;
- no regularization, model family choice, grid search or early stopping;
- choose PPR iff predicted `PPR NDCG@5 - one-step NDCG@5 > 0`;
- threshold is semantically fixed at zero and cannot be swept;
- no feature importance-based pruning or fold/model selection.

Only model coefficients, intercept and train-fold normalization statistics are
learned. Any change to this contract creates a new study and cannot reuse its
cross-validation result as confirmation.

## 3. Gold-agnostic features

For each view and event, compute from the ten candidate score vectors only:

1. one-step nonzero fraction;
2. PPR nonzero fraction;
3. one-step maximum share after L1 normalization;
4. PPR maximum share;
5. one-step top-1 minus top-2 normalized margin;
6. PPR normalized margin;
7. one-step entropy divided by `log(10)`;
8. PPR normalized entropy;
9. half-L1 distance between normalized score distributions;
10. normalized Spearman footrule distance between rankings;
11. indicator that both rankings have the same top item;
12. seed count divided by the frozen cap of six.

Ties use the already frozen candidate order. Features cannot read gold ID,
ratings, target rank, item semantics, local LLM outputs or M5 metric values.

## 4. Training target and cross-validation

Within each training fold, the regression target is the per-event signed gain:

```text
y = NDCG@5(PPR) - NDCG@5(one-step)
```

For each held-out fold, fit OLS on the other four folds and route held-out
events using predicted gain `> 0`. Concatenate all held-out decisions into one
out-of-fold policy. No event is routed by a model trained on its label.

Report router NDCG@5/Hit@5, delta versus always-one-step, paired-bootstrap CI,
PPR selection rate, fold sizes and oracle gap.

## 5. Promotion gate

Apply independently to exact and session views:

- out-of-fold router delta versus one-step `>= +0,02`; and
- paired-bootstrap 95% CI lower bound `> 0`.

| Exact router | Session router | Decision |
|---|---|---|
| Pass | Pass | Lock both full-development models; draft fresh-primary protocol |
| Pass | Fail | Lock exact model only |
| Fail | Pass | Lock session model only |
| Fail | Fail | Stop learned depth routing |

Only a passing view may be refit once on all 200 development events. The final
coefficients are not a test result; they may only be carried unchanged into a
separately locked fresh cohort.

## 6. Smoke and leakage sequence

1. Unit-test feature formulas, OLS and hash folds on synthetic inputs.
2. Feature-smoke 20 sealed score rows; require deterministic, finite features
   and no prepared/gold artifact access.
3. Materialize all 200×2 feature rows and hash them before reading labels.
4. Revalidate M5 source SHA values.
5. Open development labels once for deterministic out-of-fold evaluation.
6. Apply the frozen gate; no manual adjustment after seeing results.

Configuration: `configs/temporal_movielens32m/m6_depth_router.yaml`.

## 7. Sealed cross-validation result

The deterministic fold sizes were 43 / 41 / 34 / 43 / 39. Every held-out
prediction came from an OLS model fitted on the other four folds. The design
rank was 12 in every fold because at least one locked feature was constant;
the implementation retained it and did not perform feature selection.

| View / policy | NDCG@5 | Hit@5 |
|---|---:|---:|
| Exact always one-step | 0,593659 | 0,725 |
| Exact always PPR | 0,613055 | 0,765 |
| **Exact OOF router** | **0,593283** | **0,735** |
| Session always one-step | 0,643045 | 0,755 |
| Session always PPR | 0,607746 | 0,760 |
| **Session OOF router** | **0,634396** | **0,750** |

Exact router versus one-step:

- delta NDCG@5: **-0,000376**;
- paired-bootstrap 95% CI: **[-0,026653; +0,025604]**;
- improved / worsened / unchanged: 16 / 22 / 162;
- PPR selected for 117/200 events (58,5%);
- sign accuracy on non-ties: 40,58%;
- gate: **fail**.

Session router versus one-step:

- delta NDCG@5: **-0,008650**;
- paired-bootstrap 95% CI: **[-0,021419; +0,002334]**;
- improved / worsened / unchanged: 2 / 9 / 189;
- PPR selected for 60/200 events (30,0%);
- sign accuracy on non-ties: 53,62%;
- gate: **fail**.

Decision: `stop_depth_routing`. Since neither view passed, the implementation
did not refit on all development rows and did not write
`m6_router_model-hnv.json`.

## 8. Interpretation

The oracle heterogeneity in M5 is real descriptively, but the frozen
gold-agnostic score-distribution features do not predict it out of fold. Exact
routing is effectively neutral and session routing is worse than one-step.
Consequently, the oracle cannot justify deployment or a fresh-primary run.

This closes the current MovieLens multi-hop line under the no-manual-tuning
constraint:

- fixed PPR does not reliably beat one-step on graph-hard negatives;
- a preregistered learned depth selector also does not beat one-step;
- changing features, model class, regularization, fold count or threshold now
  would be post-outcome tuning and is prohibited.

## 9. Sealed hashes

| Artifact | SHA256 |
|---|---|
| Config | `1a4718ac1127a21d86e48c9437a3a29832223099e387e3e96046e9330263ab5e` |
| Feature smoke | `15fae1fe08fbe0db8e9822e515ab96c6a663cabe3d5ddbadc65844f3c6dce839` |
| Feature matrix, 400 rows | `14f8a27a2e4b9151faf1ef21cb3c364c560d1a2b6d5ef05c3f72d9edc429ac0d` |
| Feature manifest | `dde6103ab9faea83cc93fbde226992508d27aef79a18837c331fd1f0fa5fa9b0` |
| CV metrics | `1ae2feb42cb82a1e5e7a6224f2fed7b05f4353f0f67645d81880d518ed1eeedb` |
| CV manifest | `19564f3432da536ee30981ebbd1a3fcc2ba1c576a7b6a5b4438257a48b6d71bc` |

Code and contract were committed at `479b6af` before feature smoke. The final
audit confirms `manual_tuning_performed=false`, `final_model.written=false`,
zero LLM requests and zero GPU use.
