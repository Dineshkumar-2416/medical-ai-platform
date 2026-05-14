import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix


def compute_metrics(all_labels: list, all_probs: list, threshold: float = 0.5) -> dict:
    """
    Compute all metrics for medical binary classification.

    Args:
        all_labels : ground truth labels (0=NORMAL, 1=PNEUMONIA)
        all_probs  : predicted probabilities for PNEUMONIA class
        threshold  : decision boundary (default 0.5, tune later)

    Returns dict with: accuracy, auc, sensitivity, specificity, f1, tp, fp, tn, fn
    """
    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = (probs >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall for PNEUMONIA
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # recall for NORMAL
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    auc         = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

    return {
        "accuracy":    round(accuracy, 4),
        "auc":         round(auc, 4),
        "sensitivity": round(sensitivity, 4),   # most important — did we catch all sick patients?
        "specificity": round(specificity, 4),   # did we avoid false alarms?
        "f1":          round(f1, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def format_metrics(metrics: dict, prefix: str = "") -> str:
    p = f"{prefix}_" if prefix else ""
    return (
        f"AUC={metrics['auc']:.4f} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"Sens={metrics['sensitivity']:.4f} | "
        f"Spec={metrics['specificity']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )
