"""Evaluation metrics for seizure detection."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-12)
    sensitivity = tp / (tp + fn + 1e-12)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, np.asarray(y_score)))
        except ValueError:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")
    return out


def format_percent(x: float) -> float:
    return round(100.0 * float(x), 4)
