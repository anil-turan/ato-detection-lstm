"""Feature and score drift monitoring for the deployed ATO risk model, from
scratch.

Fraud is adversarial in a way churn isn't: attackers actively change
behaviour to evade a fixed model, and legitimate traffic (device mix,
transaction value, time-of-day patterns) also drifts on its own. A model
validated once on historical data can silently degrade on either front, with
no exception thrown to say so.

Two complementary tests, both comparing a fixed `reference` distribution
(the data the model was trained/validated on) against a `current` one (a
recent production window):

    PSI  — Population Stability Index. Bins both distributions (using the
           reference distribution's own quantile edges) and sums a
           symmetric, weighted log-ratio of bin proportions. Standard
           industry thresholds (Siddiqi, 2006, credit-scoring practice,
           unchanged since): < 0.10 stable, 0.10-0.25 moderate shift,
           > 0.25 significant shift requiring investigation/retraining.
    KS   — Kolmogorov-Smirnov statistic. The maximum absolute gap between
           the two empirical CDFs — more sensitive than PSI to a shift
           concentrated in one part of the distribution, and doesn't
           depend on a binning choice.

Both apply to any numeric series — engineered features (TransactionAmt,
amt_deviation, DeviceInfo_enc, ...) or the model's own risk scores (score
drift, which catches a shift even when no single input feature looks
alarming on its own).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_STABLE_MAX = 0.10
PSI_MODERATE_MAX = 0.25

_EPS = 1e-6  # avoids log(0)/div-by-0 when a bin is empty in one distribution


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI for a numeric feature. Bin edges come from the reference
    distribution's quantiles, so bins are equal-population in `reference` by
    construction — `current` proportions in the same bins are what reveal
    the shift."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / ref_counts.sum() + _EPS
    cur_pct = cur_counts / cur_counts.sum() + _EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample KS statistic: max gap between the empirical CDFs of
    `reference` and `current`, evaluated at every point either sample takes
    (the only points where the step-function CDFs can actually be maximally
    apart)."""
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    all_values = np.concatenate([reference, current])

    cdf_ref = np.searchsorted(reference, all_values, side="right") / len(reference)
    cdf_cur = np.searchsorted(current, all_values, side="right") / len(current)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def classify_drift(psi_value: float) -> str:
    """Standard PSI alarm bands (Siddiqi, 2006)."""
    if psi_value < PSI_STABLE_MAX:
        return "stable"
    if psi_value < PSI_MODERATE_MAX:
        return "moderate"
    return "significant"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    psi_bins: int = 10,
) -> pd.DataFrame:
    """Per-feature PSI + KS between `reference` and `current`, sorted by PSI
    descending — the features most likely driving any model degradation
    surface at the top."""
    rows = []
    for col in feature_cols:
        ref_values = reference[col].to_numpy(dtype=float)
        cur_values = current[col].to_numpy(dtype=float)
        psi_value = psi(ref_values, cur_values, bins=psi_bins)
        ks_value = ks_statistic(ref_values, cur_values)
        rows.append({
            "feature": col, "psi": psi_value, "ks": ks_value,
            "drift": classify_drift(psi_value),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )


def retraining_recommendation(score_psi: float, feature_report: pd.DataFrame) -> dict:
    """A concrete, deployable policy from the thresholds above:

    - score PSI significant (>= 0.25): retraining is overdue -- the model's
      own output distribution has shifted enough that its validated
      performance no longer applies.
    - score PSI moderate (0.10-0.25): investigate; retrain if it persists
      across consecutive monitoring windows.
    - otherwise: no action, but flag any individual feature that is itself
      significant, since a feature can drift before it shows up in the
      score (e.g. a still-compensated shift).
    """
    n_significant_features = int((feature_report["drift"] == "significant").sum())
    score_status = classify_drift(score_psi)

    if score_status == "significant":
        action = "retrain_now"
    elif score_status == "moderate" or n_significant_features > 0:
        action = "investigate"
    else:
        action = "no_action"

    return {
        "score_psi": score_psi,
        "score_drift": score_status,
        "n_features_significant": n_significant_features,
        "recommended_action": action,
    }
