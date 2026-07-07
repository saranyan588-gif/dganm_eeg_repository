#!/usr/bin/env python
"""Aggregate saved DGANM predictions into publication-ready metric tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ensure_subdirs, load_config, output_dir
from src.metrics import binary_metrics, format_percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DGANM predictions.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    cfg = load_config(parse_args().config)
    out = output_dir(cfg)
    dirs = ensure_subdirs(out)
    pred_path = dirs["predictions"] / "predictions_all_folds.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Run train.py first. Missing: {pred_path}")
    pred = pd.read_csv(pred_path)
    rows = []
    for fold, fold_df in pred.groupby("fold"):
        m = binary_metrics(fold_df["y_true"], fold_df["y_pred"], fold_df["score_preictal"])
        row = {"fold": fold, **m}
        rows.append(row)
    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(dirs["metrics"] / "fold_metrics_from_predictions.csv", index=False)

    overall = binary_metrics(pred["y_true"], pred["y_pred"], pred["score_preictal"])
    overall_df = pd.DataFrame([{k: v for k, v in overall.items()}])
    overall_df.to_csv(dirs["metrics"] / "overall_metrics.csv", index=False)

    percent_table = fold_metrics.copy()
    for col in ["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]:
        percent_table[col] = percent_table[col].apply(format_percent)
    percent_table.to_csv(dirs["metrics"] / "publication_percent_metrics.csv", index=False)

    print("Overall metrics")
    for key, value in overall.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
