"""Tests for the drift calculations (no database involved)."""

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    DriftSettings,
    bin_proportions,
    build_reference_profile,
    clean_values,
    compare_feature,
    detect_batch_drift,
    fit_psi_bin_edges,
    population_stability_index,
)

RNG_SEED = 123


def make_reference(values, name="x"):
    return build_reference_profile(pd.DataFrame({name: values}), features=[name])[name]


def by_test(results):
    return {r["test_name"]: r for r in results}


@pytest.fixture
def normal_reference():
    rng = np.random.default_rng(RNG_SEED)
    return make_reference(rng.normal(0, 1, 5000))


# --- identical and similar distributions -------------------------------------
def test_identical_distributions_show_no_drift(normal_reference):
    results = by_test(compare_feature(normal_reference, normal_reference.values.copy()))

    assert results["ks"]["score"] == 0.0
    assert results["ks"]["p_value"] == pytest.approx(1.0)
    assert results["ks"]["status"] == "no_drift"
    assert results["psi"]["score"] == pytest.approx(0.0, abs=1e-12)
    assert results["psi"]["status"] == "no_drift"


def test_new_sample_from_same_distribution_shows_no_drift(normal_reference):
    current = np.random.default_rng(RNG_SEED + 1).normal(0, 1, 2000)
    results = by_test(compare_feature(normal_reference, current))

    assert results["ks"]["status"] == "no_drift"
    assert results["psi"]["score"] < 0.10


# --- shifted distributions ----------------------------------------------------
def test_mean_shift_is_detected_by_both_tests(normal_reference):
    current = np.random.default_rng(RNG_SEED + 2).normal(1.0, 1, 2000)
    results = by_test(compare_feature(normal_reference, current))

    assert results["ks"]["status"] == "drift"
    assert results["ks"]["p_value"] < 0.05
    assert results["psi"]["status"] == "drift"
    assert results["psi"]["score"] >= 0.25


def test_spread_change_is_detected(normal_reference):
    current = np.random.default_rng(RNG_SEED + 3).normal(0, 3, 2000)
    results = by_test(compare_feature(normal_reference, current))

    assert results["ks"]["status"] == "drift"
    assert results["psi"]["status"] == "drift"


def test_small_shift_gives_psi_warning(normal_reference):
    current = np.random.default_rng(RNG_SEED + 4).normal(0.4, 1, 5000)
    psi = by_test(compare_feature(normal_reference, current))["psi"]
    assert psi["status"] == "warning"
    assert 0.10 <= psi["score"] < 0.25


# --- constant features --------------------------------------------------------
def test_constant_feature_unchanged_is_not_drift():
    reference = make_reference(np.zeros(500))
    results = by_test(compare_feature(reference, np.zeros(200)))

    assert results["ks"]["status"] == "no_drift"
    assert results["psi"]["score"] == pytest.approx(0.0, abs=1e-12)
    assert results["psi"]["status"] == "no_drift"


@pytest.mark.parametrize("new_value", [1.0, -1.0])
def test_constant_feature_that_changes_is_drift(new_value):
    reference = make_reference(np.zeros(500))
    results = by_test(compare_feature(reference, np.full(200, new_value)))

    assert results["ks"]["status"] == "drift"
    assert results["psi"]["status"] == "drift"
    assert np.isfinite(results["psi"]["score"])


# --- missing values and small samples -----------------------------------------
def test_missing_and_infinite_values_are_ignored_and_counted(normal_reference):
    current = np.random.default_rng(RNG_SEED + 5).normal(0, 1, 1000)
    current[:100] = np.nan
    current[100:110] = np.inf

    results = by_test(compare_feature(normal_reference, current))

    for result in results.values():
        assert result["n_missing"] == 110
        assert result["n_current"] == 890
        assert result["status"] in {"no_drift", "warning", "drift"}
        assert "110 missing" in result["detail"]


def test_all_missing_values_give_insufficient_data(normal_reference):
    results = compare_feature(normal_reference, np.full(500, np.nan))
    for result in results:
        assert result["status"] == "insufficient_data"
        assert result["score"] is None
        assert result["p_value"] is None


def test_small_sample_gives_insufficient_data(normal_reference):
    results = compare_feature(normal_reference, np.array([0.1, 0.2, 0.3]))
    assert {r["status"] for r in results} == {"insufficient_data"}


def test_reference_too_small_is_rejected():
    with pytest.raises(ValueError, match="usable values"):
        make_reference([1.0, 2.0, 3.0])


# --- PSI numerical stability --------------------------------------------------
def test_psi_handles_empty_bins_without_inf_or_nan():
    reference = np.array([0.5, 0.5, 0.0])
    current = np.array([0.0, 0.0, 1.0])
    psi = population_stability_index(reference, current)
    assert np.isfinite(psi)
    assert psi > 0.25


def test_psi_is_zero_for_identical_and_never_negative():
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(50):
        a = rng.dirichlet(np.ones(10))
        b = rng.dirichlet(np.ones(10))
        assert population_stability_index(a, a) == pytest.approx(0.0, abs=1e-12)
        assert population_stability_index(a, b) >= 0.0


def test_psi_when_all_current_values_fall_in_one_bin(normal_reference):
    results = by_test(compare_feature(normal_reference, np.full(500, 100.0)))
    assert np.isfinite(results["psi"]["score"])
    assert results["psi"]["status"] == "drift"


def test_psi_rejects_mismatched_bins():
    with pytest.raises(ValueError, match="same number of bins"):
        population_stability_index(np.array([0.5, 0.5]), np.array([1.0]))


# --- binning ------------------------------------------------------------------
def test_bins_are_reference_quantiles_and_cover_all_values():
    reference = np.arange(1000, dtype=float)
    edges = fit_psi_bin_edges(reference, n_bins=10)

    assert len(edges) == 9
    proportions = bin_proportions(reference, edges)
    np.testing.assert_allclose(proportions, 0.1, atol=0.002)

    # Values far outside the reference range still land in the end bins.
    outside = bin_proportions(np.array([-1e9, 1e9]), edges)
    assert outside[0] == 0.5 and outside[-1] == 0.5


def test_repeated_values_merge_bin_edges():
    reference = np.concatenate([np.zeros(900), np.arange(1, 101, dtype=float)])
    edges = fit_psi_bin_edges(reference, n_bins=10)
    assert len(edges) == len(np.unique(edges))
    assert len(edges) < 9


# --- invalid inputs and determinism ---------------------------------------------
def test_non_numeric_values_are_rejected():
    with pytest.raises(ValueError, match="numeric"):
        clean_values(["a", "b"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ks_alpha": 0},
        {"ks_alpha": 1.5},
        {"psi_n_bins": 1},
        {"psi_warning_threshold": 0.3, "psi_drift_threshold": 0.2},
        {"psi_epsilon": 0},
        {"min_sample_size": 0},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DriftSettings(**kwargs)


def test_batch_missing_a_feature_is_rejected(normal_reference):
    with pytest.raises(ValueError, match="missing feature columns"):
        detect_batch_drift({"x": normal_reference}, pd.DataFrame({"y": [1.0] * 100}))


def test_results_are_deterministic(normal_reference):
    current = np.random.default_rng(RNG_SEED + 6).normal(0.2, 1, 1000)
    assert compare_feature(normal_reference, current) == compare_feature(normal_reference, current)
