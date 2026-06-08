"""
Measure and compare the models, and pick an operating threshold.

For very imbalanced problems, ROC-AUC alone is misleading, so we also report
PR-AUC (average precision) and the precision/recall/F1 at several thresholds.
We also suggest a simple two-threshold policy for real use:
  - high score  -> block the session
  - medium score-> ask for extra verification (step-up auth)
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures without needing a screen
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

from src import config


def score_table(y_true, scores_by_model: dict) -> dict:
    """Build one results table: ROC-AUC and PR-AUC for every model."""
    table = {}
    for name, scores in scores_by_model.items():
        table[name] = {
            "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
            "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        }
    return table


def threshold_sweep(y_true, scores, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list:
    """Show precision, recall and F1 at a range of cut-off thresholds."""
    rows = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        })
    return rows


def best_f1_threshold(y_true, scores) -> dict:
    """Find the threshold that gives the best F1 score."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    t = float(thr[best]) if len(thr) else 0.5
    pred = (scores >= t).astype(int)
    return {
        "threshold": round(t, 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def recall_target_threshold(y_true, scores, target_recall=0.90) -> dict:
    """Find the highest threshold that still reaches the recall target.

    This answers the business question: "if we must catch 90% of attacks,
    what precision do we get?" It makes the recall/precision trade-off explicit.
    """
    prec, rec, thr = precision_recall_curve(y_true, scores)
    # rec is decreasing as threshold rises; find points meeting the target.
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return {"target_recall": target_recall, "reachable": False}
    idx = ok[-1]  # highest threshold that still meets the recall target
    t = float(thr[idx])
    pred = (scores >= t).astype(int)
    return {
        "target_recall": target_recall,
        "reachable": True,
        "threshold": round(t, 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
    }


def plot_roc(y_true, scores_by_model: dict, path):
    """Save one ROC curve figure with all models on it."""
    plt.figure(figsize=(7, 6))
    for name, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, scores):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - ATO detection (leak-free)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_pr(y_true, scores_by_model: dict, path):
    """Save one precision-recall curve figure (better for rare-class problems)."""
    plt.figure(figsize=(7, 6))
    for name, scores in scores_by_model.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve - ATO detection (leak-free)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_report(report: dict, path=None):
    """Write all numbers to a JSON file so the report write-up can quote them."""
    path = path or (config.REPORTS_DIR / "evaluation_v2.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[metrics] saved -> {path}")
