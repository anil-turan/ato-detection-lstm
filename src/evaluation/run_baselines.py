"""
Evaluate the sklearn baselines only (no TensorFlow needed).

This trains Random Forest + Isolation Forest on the leak-free, card-split data
and writes the real metrics. It is kept separate from run_pipeline so we can get
honest baseline numbers even when TensorFlow is broken or missing.

Run from the project root:
    PYTHONPATH=. python3 -m src.evaluation.run_baselines
"""

import numpy as np

from src import config
from src.models import baselines
from src.evaluation import metrics
from src.run_pipeline import undersample


def main():
    X_train = np.load(config.PROCESSED_DIR / "X_train.npy")
    y_train = np.load(config.PROCESSED_DIR / "y_train.npy")
    X_test = np.load(config.PROCESSED_DIR / "X_test.npy")
    y_test = np.load(config.PROCESSED_DIR / "y_test.npy")
    print(f"[baselines] train {X_train.shape} | test {X_test.shape} "
          f"| test pos {int(y_test.sum())}")

    # Undersample the normal class in TRAIN only (keeps every attack).
    X_tr, y_tr = undersample(X_train, y_train, ratio=20, seed=config.RANDOM_SEED)
    print(f"[baselines] train after undersample {X_tr.shape} pos={int(y_tr.sum())}")

    print("[baselines] Random Forest ...")
    _, rf_scores = baselines.random_forest(X_tr, y_tr, X_test)
    print("[baselines] Isolation Forest ...")
    _, iso_scores = baselines.isolation_forest(X_tr, X_test, contamination=float(y_train.mean()))

    scores = {"Random Forest": rf_scores, "Isolation Forest": iso_scores}

    report = {
        "note": "Baselines only (sklearn). LSTM not included - see run_pipeline for the LSTM.",
        "test_size": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "model_scores": metrics.score_table(y_test, scores),
        "random_forest_threshold_sweep": metrics.threshold_sweep(y_test, rf_scores),
        "random_forest_best_f1": metrics.best_f1_threshold(y_test, rf_scores),
        "random_forest_at_recall_0.90": metrics.recall_target_threshold(y_test, rf_scores, 0.90),
    }
    metrics.plot_roc(y_test, scores, config.FIGURES_DIR / "roc_v2_baselines.png")
    metrics.plot_pr(y_test, scores, config.FIGURES_DIR / "pr_v2_baselines.png")
    metrics.save_report(report, config.REPORTS_DIR / "evaluation_baselines_v2.json")

    print("\n=== REAL BASELINE RESULTS ===")
    for name, vals in report["model_scores"].items():
        print(f"  {name:18s} ROC-AUC={vals['roc_auc']:.4f}  PR-AUC={vals['pr_auc']:.4f}")
    print("  RF best-F1:", report["random_forest_best_f1"])
    print("  RF at 90% recall:", report["random_forest_at_recall_0.90"])


if __name__ == "__main__":
    main()
