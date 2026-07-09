"""Tests for PSI / KS drift monitoring and the retraining-recommendation
policy."""

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    classify_drift,
    drift_report,
    ks_statistic,
    psi,
    retraining_recommendation,
)


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(50, 10, 5000)
    current = rng.normal(50, 10, 5000)
    assert psi(reference, current) < 0.02


def test_psi_increases_with_distribution_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(50, 10, 5000)
    small_shift = rng.normal(52, 10, 5000)
    large_shift = rng.normal(70, 10, 5000)
    assert psi(reference, small_shift) < psi(reference, large_shift)


def test_classify_drift_thresholds():
    assert classify_drift(0.05) == "stable"
    assert classify_drift(0.15) == "moderate"
    assert classify_drift(0.30) == "significant"
    assert classify_drift(0.10) == "moderate"
    assert classify_drift(0.25) == "significant"


def test_ks_statistic_zero_for_identical_arrays():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 2000)
    assert ks_statistic(reference, reference) == 0.0


def test_ks_statistic_detects_shift():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 3000)
    current_same = rng.normal(0, 1, 3000)
    current_shifted = rng.normal(3, 1, 3000)
    assert ks_statistic(reference, current_same) < ks_statistic(reference, current_shifted)


def test_ks_statistic_matches_scipy():
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 1000)
    current = rng.normal(0.5, 1.2, 1000)
    ours = ks_statistic(reference, current)
    scipy_stat = ks_2samp(reference, current).statistic
    assert abs(ours - scipy_stat) < 1e-9


def test_drift_report_flags_the_shifted_feature():
    rng = np.random.default_rng(5)
    n = 3000
    reference = pd.DataFrame({
        "stable_feature": rng.normal(50, 10, n),
        "shifted_feature": rng.normal(50, 10, n),
    })
    current = pd.DataFrame({
        "stable_feature": rng.normal(50, 10, n),
        "shifted_feature": rng.normal(90, 10, n),
    })
    report = drift_report(reference, current, feature_cols=["stable_feature", "shifted_feature"])
    assert report.iloc[0]["feature"] == "shifted_feature"
    assert report.iloc[0]["drift"] == "significant"
    stable_row = report[report["feature"] == "stable_feature"].iloc[0]
    assert stable_row["drift"] == "stable"


def test_retraining_recommendation_retrain_now_on_significant_score_drift():
    feature_report = pd.DataFrame({"feature": ["a", "b"], "drift": ["stable", "stable"]})
    rec = retraining_recommendation(score_psi=0.40, feature_report=feature_report)
    assert rec["recommended_action"] == "retrain_now"


def test_retraining_recommendation_no_action_when_everything_stable():
    feature_report = pd.DataFrame({"feature": ["a", "b"], "drift": ["stable", "stable"]})
    rec = retraining_recommendation(score_psi=0.02, feature_report=feature_report)
    assert rec["recommended_action"] == "no_action"


def test_retraining_recommendation_investigate_when_a_feature_is_significant():
    """Score itself is stable, but one feature has already drifted
    significantly -- catches a shift before it shows up in the score."""
    feature_report = pd.DataFrame({"feature": ["a"], "drift": ["significant"]})
    rec = retraining_recommendation(score_psi=0.02, feature_report=feature_report)
    assert rec["recommended_action"] == "investigate"


def test_retraining_recommendation_investigate_on_moderate_score_drift():
    feature_report_stable = pd.DataFrame({"feature": ["a"], "drift": ["stable"]})
    rec = retraining_recommendation(score_psi=0.15, feature_report=feature_report_stable)
    assert rec["recommended_action"] == "investigate"
