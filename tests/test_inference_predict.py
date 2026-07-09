"""Tests for the two-tier decision policy in src/inference/predict.py."""

import numpy as np

from src.inference.predict import THRESHOLDS, decide


def test_decide_rf_backend_thresholds():
    verify = THRESHOLDS["rf"]["verify"] * 100
    block = THRESHOLDS["rf"]["block"] * 100
    risk = np.array([0.0, verify - 0.01, verify, block - 0.01, block, 100.0])
    actions = decide(risk, backend="rf")
    assert list(actions) == ["allow", "allow", "step_up_auth", "step_up_auth", "block", "block"]


def test_decide_lstm_backend_thresholds():
    verify = THRESHOLDS["lstm"]["verify"] * 100
    block = THRESHOLDS["lstm"]["block"] * 100
    risk = np.array([0.0, verify, block])
    actions = decide(risk, backend="lstm")
    assert list(actions) == ["allow", "step_up_auth", "block"]


def test_decide_unknown_backend_falls_back_to_rf():
    risk = np.array([0.0])
    assert decide(risk, backend="not_a_real_backend")[0] == decide(risk, backend="rf")[0]


def test_decide_is_monotonic_in_risk():
    """Higher risk should never map to a less severe action."""
    severity = {"allow": 0, "step_up_auth": 1, "block": 2}
    risk = np.linspace(0, 100, 50)
    actions = decide(risk, backend="rf")
    severities = [severity[a] for a in actions]
    assert all(a <= b for a, b in zip(severities, severities[1:]))
