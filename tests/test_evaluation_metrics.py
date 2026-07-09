"""Tests for src/evaluation/metrics.py -- score comparison, threshold
sweeps, and the two threshold-selection policies (best F1 / recall target)."""

import numpy as np

from src.evaluation.metrics import (
    best_f1_threshold,
    recall_target_threshold,
    score_table,
    threshold_sweep,
)


def _separable_scores(n=1000, seed=0):
    """Scores that actually separate the classes (unlike random noise),
    so thresholds/AUC behave the way a sane metric should."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    scores = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.5, n))
    return y, scores


def test_score_table_reports_roc_and_pr_auc_per_model():
    y, scores = _separable_scores()
    table = score_table(y, {"model_a": scores, "model_b": scores})
    assert set(table.keys()) == {"model_a", "model_b"}
    for entry in table.values():
        assert 0.9 < entry["roc_auc"] <= 1.0  # scores clearly separate the classes
        assert 0 < entry["pr_auc"] <= 1.0


def test_threshold_sweep_precision_increases_with_threshold_on_separable_scores():
    y, scores = _separable_scores()
    rows = threshold_sweep(y, scores, thresholds=(0.3, 0.5, 0.7))
    precisions = [r["precision"] for r in rows]
    # a higher bar should never make precision worse for well-separated scores
    assert precisions[-1] >= precisions[0]


def test_threshold_sweep_recall_decreases_with_threshold():
    y, scores = _separable_scores()
    rows = threshold_sweep(y, scores, thresholds=(0.3, 0.5, 0.7))
    recalls = [r["recall"] for r in rows]
    assert recalls[0] >= recalls[-1]


def test_best_f1_threshold_beats_a_bad_threshold():
    y, scores = _separable_scores()
    best = best_f1_threshold(y, scores)
    bad = threshold_sweep(y, scores, thresholds=(0.95,))[0]
    assert best["f1"] >= bad["f1"]


def test_recall_target_threshold_is_reachable_for_separable_scores():
    y, scores = _separable_scores()
    result = recall_target_threshold(y, scores, target_recall=0.90)
    assert result["reachable"] is True
    assert result["recall"] >= 0.90


def test_recall_target_threshold_unreachable_when_target_impossible():
    y, scores = _separable_scores()
    result = recall_target_threshold(y, scores, target_recall=1.01)
    assert result["reachable"] is False


def test_recall_target_threshold_lower_bar_gives_higher_precision():
    """Asking for less recall should never require a lower-precision
    operating point than asking for more recall."""
    y, scores = _separable_scores()
    high_recall = recall_target_threshold(y, scores, target_recall=0.95)
    low_recall = recall_target_threshold(y, scores, target_recall=0.50)
    assert low_recall["precision"] >= high_recall["precision"] - 1e-9
