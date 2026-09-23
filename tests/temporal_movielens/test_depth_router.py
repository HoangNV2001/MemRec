import numpy as np

from src.temporal_movielens.depth_router import (
    FEATURE_NAMES,
    fit_standardized_ols,
    fold_for_key,
    predict_standardized_ols,
    router_features,
)


def toy_config():
    return {"model": {"fold_salt": "fixed", "folds": 5}}


def test_router_features_are_finite_gold_agnostic_and_ordered():
    candidates = [str(index) for index in range(10)]
    row = {
        "candidate_item_ids": candidates,
        "seed_item_ids": ["a", "b", "c"],
        "views": {
            "exact": {
                "one_step_scores": {item: float(index + 1) for index, item in enumerate(candidates)},
                "ppr_scores": {item: float(10 - index) for index, item in enumerate(candidates)},
            }
        },
    }
    first = router_features(row, "exact", seed_cap=6)
    row["gold_item_id"] = "9"
    second = router_features(row, "exact", seed_cap=6)
    assert first == second
    assert tuple(first) == FEATURE_NAMES
    assert all(np.isfinite(list(first.values())))
    assert first["seed_fraction"] == 0.5
    assert first["top1_agreement"] == 0.0
    assert first["normalized_rank_footrule"] == 1.0


def test_standardized_ols_has_no_tuned_parameter_and_recovers_linear_signal():
    x = np.arange(30, dtype=float).reshape(15, 2)
    y = 2.0 + 3.0 * x[:, 0] - 0.5 * x[:, 1]
    model = fit_standardized_ols(x, y)
    predicted = predict_standardized_ols(model, x)
    assert np.allclose(predicted, y)
    assert set(model) == {
        "feature_mean",
        "feature_scale",
        "intercept",
        "coefficients",
        "design_rank",
        "singular_values",
        "training_rows",
    }


def test_hash_folds_are_deterministic_without_search():
    values = [fold_for_key(f"u:{index}", toy_config()) for index in range(50)]
    assert values == [fold_for_key(f"u:{index}", toy_config()) for index in range(50)]
    assert set(values) == {0, 1, 2, 3, 4}
