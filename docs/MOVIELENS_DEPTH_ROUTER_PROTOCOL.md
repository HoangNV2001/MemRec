# MovieLens learned graph-depth router protocol

> Status: frozen before router feature/outcome evaluation. This study follows
> the sealed graph-hard result and obeys a strict no-manual-tuning contract.

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
