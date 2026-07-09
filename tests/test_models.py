"""Tests for src/models/baselines.py, src/models/lstm.py, and the
undersample() helper in src/run_pipeline.py."""

import numpy as np

from src.models.baselines import isolation_forest, random_forest
from src.models.lstm import build_model, class_weights
from src.run_pipeline import undersample


def _toy_sequences(n=200, window=5, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window, n_features)).astype("float32")
    y = (rng.random(n) < 0.05).astype("int8")  # rare positive class, like real ATO rate
    return X, y


def test_class_weights_caps_at_25():
    y = np.array([1] + [0] * 10_000)  # extremely imbalanced
    weights = class_weights(y)
    assert weights[1] == 25.0
    assert weights[0] == 1.0


def test_class_weights_scales_with_imbalance_below_cap():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # 2 pos, 6 neg -> ratio 3, below the cap
    weights = class_weights(y)
    assert weights[1] == 3.0


def test_undersample_keeps_every_positive():
    X, y = _toy_sequences(n=500, seed=1)
    X_sub, y_sub = undersample(X, y, ratio=5, seed=42)
    assert y_sub.sum() == y.sum()  # every positive example is kept


def test_undersample_respects_the_ratio():
    X, y = _toy_sequences(n=2000, seed=2)
    ratio = 10
    X_sub, y_sub = undersample(X, y, ratio=ratio, seed=42)
    n_pos = int(y_sub.sum())
    n_neg = len(y_sub) - n_pos
    assert n_neg <= ratio * n_pos


def test_undersample_never_exceeds_original_size():
    X, y = _toy_sequences(n=50, seed=3)
    X_sub, y_sub = undersample(X, y, ratio=1000, seed=42)  # ratio far bigger than the data
    assert len(y_sub) <= len(y)


def test_random_forest_scores_are_valid_probabilities():
    X, y = _toy_sequences(n=300, seed=4)
    X_train, y_train = X[:200], y[:200]
    X_test = X[200:]
    _, scores = random_forest(X_train, y_train, X_test)
    assert scores.shape == (100,)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_isolation_forest_scores_are_scaled_to_unit_range():
    X, y = _toy_sequences(n=300, seed=5)
    X_train, X_test = X[:200], X[200:]
    contamination = float(y[:200].mean())
    _, scores = isolation_forest(X_train, X_test, contamination=contamination)
    assert scores.shape == (100,)
    assert scores.min() >= -1e-9 and scores.max() <= 1 + 1e-9


def test_build_model_output_shape_and_compiles():
    model = build_model(window_size=10, n_features=6)
    assert model.output_shape == (None, 1)
    dummy = np.zeros((4, 10, 6), dtype="float32")
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (4, 1)
    assert (preds >= 0).all() and (preds <= 1).all()  # sigmoid output
