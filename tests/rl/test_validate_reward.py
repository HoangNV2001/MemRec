"""
Unit tests for the M2 validation harness itself.

The harness decides whether the reward is trustworthy enough to spend 25 GPU-hour
on, so its own arithmetic needs checking -- a broken Spearman would either waive
the gate or block it for no reason.
"""
import math

import pytest

from src.rl.validate_reward import _sensitivity_ok, spearman


# --- Spearman ---------------------------------------------------------------

def test_spearman_perfect_positive():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_value_based():
    """Monotone but wildly non-linear must still be rho = 1."""
    assert spearman([1, 2, 3, 4], [1, 10, 1000, 100000]) == pytest.approx(1.0)


def test_spearman_handles_ties_with_average_ranks():
    # scipy.stats.spearmanr([1,2,2,3],[1,2,3,4]).statistic == 0.9486832980505138
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138, abs=1e-9)


def test_spearman_all_tied_is_nan_not_crash():
    """A constant proxy must surface as nan, not as a divide-by-zero traceback."""
    assert math.isnan(spearman([5, 5, 5, 5], [1, 2, 3, 4]))


def test_spearman_too_few_points_is_nan():
    assert math.isnan(spearman([1], [2]))


def test_spearman_matches_scipy_on_random_data():
    import numpy as np
    from scipy.stats import spearmanr

    rng = np.random.RandomState(0)
    for _ in range(5):
        xs = rng.rand(40).tolist()
        ys = (np.array(xs) * 0.5 + rng.rand(40) * 0.5).tolist()
        assert spearman(xs, ys) == pytest.approx(spearmanr(xs, ys).statistic, abs=1e-9)


# --- Validation B ordering --------------------------------------------------

def test_sensitivity_accepts_real_memory_beating_every_corrupted_arm():
    assert _sensitivity_ok(
        {"sample1": 0.60, "shuffled": 0.45, "lorem": 0.38, "empty": 0.37}
    )


def test_sensitivity_accepts_the_shape_actually_observed():
    """
    Measured with gpt-4o-mini on 149 val users: the corrupted arms cluster together
    because the real ranker ignores irrelevant memory rather than being misled.
    That must pass -- requiring shuffled > lorem would demand the proxy be more
    confusable than the model it stands in for.
    """
    assert _sensitivity_ok(
        {"sample1": 0.7204, "shuffled": 0.6090, "lorem": 0.6079, "empty": 0.6092}
    )


def test_sensitivity_rejects_insensitive_reward():
    """If garbage memory scores as well as real memory, the reward is useless."""
    assert not _sensitivity_ok(
        {"sample1": 0.40, "shuffled": 0.41, "lorem": 0.40, "empty": 0.40}
    )


def test_sensitivity_rejects_a_margin_too_small_to_trust():
    assert not _sensitivity_ok(
        {"sample1": 0.605, "shuffled": 0.600, "lorem": 0.599, "empty": 0.600}
    )


def test_sensitivity_rejects_missing_arm():
    assert not _sensitivity_ok({"sample1": 0.6, "shuffled": 0.4})
