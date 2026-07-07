"""High-resolution plotting utilities.

All plots are generated from CSV metric tables, not screen captures. This avoids
pixelation and directly addresses figure-quality concerns during review.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc


def _save(fig, path: Path, dpi: int = 600, save_svg: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_loso_metrics(metrics_csv: Path, out_path: Path, dpi: int = 600, save_svg: bool = True) -> None:
    df = pd.read_csv(metrics_csv)
    if df.empty:
        raise ValueError(f"No rows in {metrics_csv}")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    ax.bar(x - width, df["accuracy"] * 100, width, label="Accuracy")
    ax.bar(x, df["f1"] * 100, width, label="F1-score")
    ax.bar(x + width, df["auc"] * 100, width, label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["target_subject"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Value (%)")
    ax.set_xlabel("Held-out subject")
    ax.set_title("Cross-subject LOSO seizure detection performance")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path, dpi=dpi, save_svg=save_svg)


def plot_training_curve(log_csv: Path, out_path: Path, dpi: int = 600, save_svg: bool = True) -> None:
    df = pd.read_csv(log_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["d_loss"], marker="o", label="Discriminator loss")
    ax.plot(df["epoch"], df["c_loss"], marker="s", label="Classifier loss")
    ax.plot(df["epoch"], df["g_loss"], marker="^", label="Generator loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DGANM adversarial training convergence")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_path, dpi=dpi, save_svg=save_svg)


def plot_confusion_from_predictions(pred_csv: Path, out_path: Path, dpi: int = 600, save_svg: bool = True) -> None:
    df = pd.read_csv(pred_csv)
    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Interictal", "Preictal"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("DGANM confusion matrix")
    _save(fig, out_path, dpi=dpi, save_svg=save_svg)


def plot_roc_from_predictions(pred_csv: Path, out_path: Path, dpi: int = 600, save_svg: bool = True) -> None:
    df = pd.read_csv(pred_csv)
    if len(df["y_true"].unique()) < 2:
        raise ValueError("ROC cannot be drawn because only one class is present in predictions.")
    fpr, tpr, _ = roc_curve(df["y_true"], df["score_preictal"])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("DGANM ROC curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_path, dpi=dpi, save_svg=save_svg)
