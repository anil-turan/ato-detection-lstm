"""
Explain which features drive the predictions.

Feature importance matters here for the FCA/GDPR audit trail. We explain
a Random Forest because tree models are fast and exact to explain.

Each sequence is 10 time-steps x N features flattened into one long row, so a
flattened column means "feature F at step T". We add up the importance across all
10 steps to get one importance value per original feature.

Two methods are produced:
  1. SHAP TreeExplainer on a small sample (the audit-grade method).
  2. The Random Forest's built-in importances (always available, used as a
     fast fallback so the report never gets stuck).

A small, dedicated forest is trained here just for explaining, so SHAP stays
fast and does not hang.

Run from the project root:
    PYTHONPATH=. python3 -m src.evaluation.explain
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.run_pipeline import undersample


def _aggregate_over_timesteps(per_col: np.ndarray, n_feat: int) -> np.ndarray:
    """Turn one value per flattened column into one value per original feature.

    The flattened row is (10 timesteps x n_feat), so we reshape and sum the 10
    steps for each feature.
    """
    return per_col.reshape(config.WINDOW_SIZE, n_feat).sum(axis=0)


def _save_ranking(ranking, json_name, fig_title, fig_name):
    """Write a ranking to JSON and draw a top-12 bar chart."""
    with open(config.REPORTS_DIR / json_name, "w") as f:
        json.dump([{"feature": k, "importance": round(float(v), 6)} for k, v in ranking],
                  f, indent=2)
    top = ranking[:12][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh([k for k, _ in top], [v for _, v in top], color="#B85042")
    plt.xlabel("Importance (summed over 10 steps)")
    plt.title(fig_title)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / fig_name, dpi=120)
    plt.close()


def run(max_explain=400, n_trees=60):
    # Load data and the feature names.
    X_train = np.load(config.PROCESSED_DIR / "X_train.npy")
    y_train = np.load(config.PROCESSED_DIR / "y_train.npy")
    X_test = np.load(config.PROCESSED_DIR / "X_test.npy")
    feature_cols = (config.PROCESSED_DIR / "feature_cols.txt").read_text().strip().split("\n")
    n_feat = len(feature_cols)

    # Train a SMALL forest just for explaining (keeps SHAP fast).
    X_tr, y_tr = undersample(X_train, y_train, ratio=20, seed=config.RANDOM_SEED)
    clf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=config.RANDOM_SEED,
    )
    clf.fit(X_tr.reshape(len(X_tr), -1), y_tr)

    # --- Method 1: built-in importances (always works, never hangs) ---------
    builtin = _aggregate_over_timesteps(clf.feature_importances_, n_feat)
    builtin_rank = sorted(zip(feature_cols, builtin), key=lambda x: -x[1])
    _save_ranking(builtin_rank, "feature_importance_rf_builtin_v2.json",
                  "Top features driving ATO risk (RF built-in importance)",
                  "feature_importance_rf_builtin_v2.png")
    print("[explain] built-in RF importance top 8:")
    for k, v in builtin_rank[:8]:
        print(f"          {k:18s} {v:.5f}")

    # --- Method 2: SHAP on a small sample (audit-grade, kept light) ---------
    try:
        import shap
        sample = X_test[:max_explain].reshape(min(max_explain, len(X_test)), -1)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(sample, check_additivity=False)
        if isinstance(sv, list):           # [class0, class1] -> take attack class
            sv = sv[1]
        if getattr(sv, "ndim", 2) == 3:    # (rows, cols, classes)
            sv = sv[:, :, -1]
        shap_imp = _aggregate_over_timesteps(np.abs(sv).mean(axis=0), n_feat)
        shap_rank = sorted(zip(feature_cols, shap_imp), key=lambda x: -x[1])
        _save_ranking(shap_rank, "shap_feature_importance_v2.json",
                      "Top features driving ATO risk (SHAP, Random Forest)",
                      "shap_bar_v2.png")
        print("[explain] SHAP top 8:")
        for k, v in shap_rank[:8]:
            print(f"          {k:18s} {v:.5f}")
        print("[explain] saved SHAP json + figure")
    except Exception as e:
        print(f"[explain] SHAP step skipped ({type(e).__name__}: {e}); "
              f"use the built-in importance instead.")

    print("[explain] done")


if __name__ == "__main__":
    run()
