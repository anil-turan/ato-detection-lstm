"""
Run the full ATO detection pipeline end to end.

Order of steps:
  1. labeling  -> build the ato_proxy label (skipped if already built)
  2. features  -> leak-free, card-split, scaled sequences (skipped if built)
  3. train     -> the LSTM model
  4. baselines -> Random Forest + Isolation Forest
  5. evaluate  -> ROC-AUC, PR-AUC, threshold sweep, plots, JSON + markdown table

Run from the project root:
    PYTHONPATH=. python3 -m src.run_pipeline
Use --rebuild to force the labeling/feature steps to run again.
"""

import argparse

import numpy as np

from src import config
from src.data import labeling
from src.features import build as features
from src.models import baselines, lstm
from src.evaluation import metrics


def undersample(X, y, ratio=20, seed=42):
    """Keep every attack example, plus `ratio` times as many normal examples.

    Attacks are only ~0.4% of the data, so training on all of it is slow and
    very imbalanced. Undersampling the normal class keeps all the rare attacks,
    speeds up training a lot, and helps the model focus on the minority class.
    The TEST set is never undersampled, so the reported scores stay realistic.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), ratio * len(pos_idx)), replace=False)
    idx = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _markdown_table(report: dict) -> str:
    """Make a small markdown results table for the report write-up."""
    lines = ["| Model | ROC-AUC | PR-AUC |", "|---|---|---|"]
    for name, vals in report["model_scores"].items():
        lines.append(f"| {name} | {vals['roc_auc']} | {vals['pr_auc']} |")
    return "\n".join(lines)


def main(rebuild=False):
    # --- Step 1: labels ----------------------------------------------------
    labeled = config.PROCESSED_DIR / "df_labeled_v2.parquet"
    if rebuild or not labeled.exists():
        labeling.run()
    else:
        print("[pipeline] labels already exist, skipping labeling")

    # --- Step 2: features --------------------------------------------------
    x_train_path = config.PROCESSED_DIR / "X_train.npy"
    if rebuild or not x_train_path.exists():
        features.run()
    else:
        print("[pipeline] feature arrays already exist, skipping feature build")

    X_train = np.load(config.PROCESSED_DIR / "X_train.npy")
    y_train = np.load(config.PROCESSED_DIR / "y_train.npy")
    X_test = np.load(config.PROCESSED_DIR / "X_test.npy")
    y_test = np.load(config.PROCESSED_DIR / "y_test.npy")
    print(f"[pipeline] train {X_train.shape} | test {X_test.shape}")

    # Undersample the normal class in TRAINING only (keeps all attacks).
    X_tr, y_tr = undersample(X_train, y_train, ratio=20, seed=config.RANDOM_SEED)
    print(f"[pipeline] training set after undersampling: {X_tr.shape} "
          f"pos={int(y_tr.sum())} ({y_tr.mean()*100:.2f}%)")

    # --- Step 3: LSTM ------------------------------------------------------
    model, _ = lstm.train(X_tr, y_tr, epochs=15, batch_size=512)
    lstm_scores = model.predict(X_test, batch_size=1024, verbose=0).ravel()

    # --- Step 4: baselines -------------------------------------------------
    print("[pipeline] training Random Forest baseline ...")
    _, rf_scores = baselines.random_forest(X_tr, y_tr, X_test)
    print("[pipeline] training Isolation Forest baseline ...")
    # contamination = the real attack rate (not the undersampled rate).
    _, iso_scores = baselines.isolation_forest(X_tr, X_test, contamination=float(y_train.mean()))

    scores_by_model = {
        "LSTM (ours)": lstm_scores,
        "Random Forest": rf_scores,
        "Isolation Forest": iso_scores,
    }

    # --- Step 5: evaluate --------------------------------------------------
    report = {
        "test_size": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "model_scores": metrics.score_table(y_test, scores_by_model),
        "lstm_threshold_sweep": metrics.threshold_sweep(y_test, lstm_scores),
        "lstm_best_f1": metrics.best_f1_threshold(y_test, lstm_scores),
        "lstm_at_recall_0.90": metrics.recall_target_threshold(y_test, lstm_scores, 0.90),
    }
    metrics.plot_roc(y_test, scores_by_model, config.FIGURES_DIR / "roc_v2_all_models.png")
    metrics.plot_pr(y_test, scores_by_model, config.FIGURES_DIR / "pr_v2_all_models.png")
    metrics.save_report(report)

    table_md = _markdown_table(report)
    with open(config.REPORTS_DIR / "results_table_v2.md", "w") as f:
        f.write("# ATO detection results (leak-free pipeline)\n\n")
        f.write(table_md + "\n")
    print("\n=== RESULTS ===")
    print(table_md)
    print("\nLSTM best-F1 point:", report["lstm_best_f1"])
    print("LSTM at 90% recall:", report["lstm_at_recall_0.90"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="force re-run of labeling and feature steps")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
