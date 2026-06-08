"""
Baseline models to compare against the LSTM.

The LSTM only looks good if it beats simpler models. We use two baselines:
  - Random Forest    -> a strong classic model (needs flat input, so we
                        flatten each sequence into one long row).
  - Isolation Forest -> an anomaly detector that learns "normal" and flags
                        the odd ones out (unsupervised).

Both are set up to handle the heavy class imbalance fairly. The old project
ran Random Forest without balancing, so it predicted "never fraud" and scored
recall 0; class_weight="balanced" fixes that.
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from src import config


def _flatten(X: np.ndarray) -> np.ndarray:
    """Turn (samples, timesteps, features) into (samples, timesteps*features)."""
    return X.reshape(X.shape[0], -1)


def random_forest(X_train, y_train, X_test):
    """Train a balanced Random Forest and return its risk scores for the test set."""
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",   # make the rare attack class count more
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
    )
    clf.fit(_flatten(X_train), y_train)
    scores = clf.predict_proba(_flatten(X_test))[:, 1]
    return clf, scores


def isolation_forest(X_train, X_test, contamination):
    """Train an Isolation Forest on the flattened data and return anomaly scores.

    Higher score = more anomalous. We flip sklearn's sign so that bigger means
    riskier, matching the other models.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=min(max(contamination, 1e-4), 0.5),
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(_flatten(X_train))
    # score_samples: higher = more normal, so negate to get a risk score.
    scores = -iso.score_samples(_flatten(X_test))
    # Scale to 0-1 so it is comparable to probability outputs.
    lo, hi = scores.min(), scores.max()
    scores = (scores - lo) / (hi - lo + 1e-9)
    return iso, scores
