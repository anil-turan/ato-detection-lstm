"""
Turn a trained model into a usable ATO risk scorer.

This module does two things:
  1. Give each session/sequence a risk score from 0 to 100 (real-time scoring).
  2. Turn that score into an action using a two-tier policy:
        score >= BLOCK_AT   -> "block"
        score >= VERIFY_AT  -> "step_up_auth"  (ask for extra verification)
        otherwise           -> "allow"

It exports a small scored table that a downstream case-management system or
monitoring dashboard can read, joined on TransactionID.

There are two backends:
  - "rf"  : Random Forest (default). Works anywhere, no TensorFlow needed.
  - "lstm": the LSTM model. Used only if TensorFlow is available.
"""

import numpy as np
import pandas as pd

from src import config

# Decision thresholds on the 0-1 probability, one set per model. Each model has a
# different score distribution, so the thresholds must match the model.
#   - VERIFY = the "catch ~90% of attacks" threshold (step-up auth from here up)
#   - BLOCK  = the "best F1" threshold, where precision is highest (block from here up)
# Below VERIFY -> allow; between VERIFY and BLOCK -> step-up auth; >= BLOCK -> block.
# Values taken from the real evaluation JSON files.
THRESHOLDS = {
    # from evaluation_v2.json (LSTM)
    "lstm": {"verify": 0.45, "block": 0.95},
    # from evaluation_baselines_v2.json (Random Forest)
    "rf": {"verify": 0.055, "block": 0.16},
}


def _rf_scores(X: np.ndarray) -> np.ndarray:
    """Train the Random Forest and return its 0-1 attack probability per row."""
    from src.models import baselines
    from src.run_pipeline import undersample

    X_train = np.load(config.PROCESSED_DIR / "X_train.npy")
    y_train = np.load(config.PROCESSED_DIR / "y_train.npy")
    X_tr, y_tr = undersample(X_train, y_train, ratio=20, seed=config.RANDOM_SEED)
    _, scores = baselines.random_forest(X_tr, y_tr, X)
    return scores


def _lstm_scores(X: np.ndarray) -> np.ndarray:
    """Load the trained LSTM and return its 0-1 attack probability per row."""
    import tensorflow as tf  # imported lazily so RF backend needs no TensorFlow

    model = tf.keras.models.load_model(config.MODELS_DIR / "lstm_v2_best.keras")
    return model.predict(X, batch_size=1024, verbose=0).ravel()


def score_sequences(X: np.ndarray, backend: str = "rf") -> np.ndarray:
    """Return a 0-100 risk score for each input sequence (higher = riskier)."""
    probs = _lstm_scores(X) if backend == "lstm" else _rf_scores(X)
    return np.round(probs * 100, 2)


def decide(risk_0_100: np.ndarray, backend: str = "rf") -> np.ndarray:
    """Map each 0-100 risk score to an action using the model's two-tier policy."""
    t = THRESHOLDS.get(backend, THRESHOLDS["rf"])
    block = t["block"] * 100
    verify = t["verify"] * 100
    return np.where(risk_0_100 >= block, "block",
                    np.where(risk_0_100 >= verify, "step_up_auth", "allow"))


def export_risk_scores(backend: str = "rf", out_path=None) -> pd.DataFrame:
    """Score the held-out test set and save a table for downstream consumers
    (a case-management system or monitoring dashboard).

    TransactionID is the join key any downstream consumer would merge on.
    """
    X_test = np.load(config.PROCESSED_DIR / "X_test.npy")
    txn_ids = np.load(config.PROCESSED_DIR / "txn_ids_test.npy")

    risk = score_sequences(X_test, backend=backend)
    actions = decide(risk, backend=backend)

    out = pd.DataFrame({
        "TransactionID": txn_ids,        # join key for downstream consumers
        "ato_risk_score": risk,          # 0-100, higher = riskier
        "recommended_action": actions,   # allow / step_up_auth / block
    })

    out_path = out_path or (config.REPORTS_DIR / "ato_scores_for_dashboard.csv")
    out.to_csv(out_path, index=False)

    # Print how many sessions fall in each action band.
    counts = out["recommended_action"].value_counts().to_dict()
    print(f"[inference] backend={backend}  scored {len(out):,} sessions -> {counts}")
    print(f"[inference] saved -> {out_path}")
    return out


if __name__ == "__main__":
    export_risk_scores(backend="rf")
