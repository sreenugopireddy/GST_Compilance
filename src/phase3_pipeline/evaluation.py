"""
Phase 3 — Evaluation.

Classifier metrics, confusion matrix / ROC curve plotting, and
precision@k / recall@k for anomaly-ranked GSTINs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
from src import config


def evaluate_gnn_classifier(y_true, y_probs, title_suffix="", save_plot=True,
                             plot_name="gnn_evaluation.png"):
    """
    Print accuracy/precision/recall/F1/ROC-AUC and plot confusion matrix + ROC.

    Inputs:
        y_true: np.ndarray (0/1)
        y_probs: np.ndarray - predicted P(risk_flag == 1)
        title_suffix: str - appended to the printed header
        save_plot: bool - save figure to outputs/plots/
        plot_name: str

    Returns:
        metrics: dict of the computed scores
    """
    y_pred = (y_probs >= config.RISK_LABEL_THRESHOLD).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs),
    }

    print("=" * 70)
    print(f"GNN CLASSIFIER EVALUATION {title_suffix}")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    plt.tight_layout()

    if save_plot:
        path = os.path.join(config.PLOTS_DIR, plot_name)
        plt.savefig(path, dpi=120)
        print(f"Saved plot: {path}\n")
    plt.close()

    return metrics


def precision_recall_at_k(y_true, scores, k=50):
    """
    Precision@k and recall@k for the top-k highest-scoring GSTINs.

    Inputs:
        y_true: np.ndarray (0/1)
        scores: np.ndarray - higher = riskier
        k: int

    Returns:
        (precision_at_k, recall_at_k)
    """
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    top_labels = y_true[order]
    precision_at_k = top_labels.sum() / k
    recall_at_k = top_labels.sum() / max(y_true.sum(), 1)
    print(f"Precision@{k}: {precision_at_k:.4f} | Recall@{k}: {recall_at_k:.4f}")
    return precision_at_k, recall_at_k
