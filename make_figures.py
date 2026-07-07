#!/usr/bin/env python
"""Generate high-resolution manuscript figures from reproducible CSV outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ensure_subdirs, load_config, output_dir
from src.visualization import plot_confusion_from_predictions, plot_loso_metrics, plot_roc_from_predictions, plot_training_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create high-resolution DGANM figures.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    cfg = load_config(parse_args().config)
    out = output_dir(cfg)
    dirs = ensure_subdirs(out)
    dpi = int(cfg.get("figures", {}).get("dpi", 600))
    save_svg = bool(cfg.get("figures", {}).get("save_svg", True))
    fmt = str(cfg.get("figures", {}).get("format", "png")).lstrip(".")

    metrics_csv = dirs["metrics"] / "loso_metrics.csv"
    pred_csv = dirs["predictions"] / "predictions_all_folds.csv"
    if metrics_csv.exists():
        plot_loso_metrics(metrics_csv, dirs["figures"] / f"figure_loso_cross_subject_metrics.{fmt}", dpi=dpi, save_svg=save_svg)
    if pred_csv.exists():
        plot_confusion_from_predictions(pred_csv, dirs["figures"] / f"figure_confusion_matrix.{fmt}", dpi=dpi, save_svg=save_svg)
        try:
            plot_roc_from_predictions(pred_csv, dirs["figures"] / f"figure_roc_curve.{fmt}", dpi=dpi, save_svg=save_svg)
        except ValueError as exc:
            print(f"Skipping ROC: {exc}")

    for log_csv in sorted((dirs["logs"]).glob("training_*.csv")):
        name = log_csv.stem.replace("training_", "")
        plot_training_curve(log_csv, dirs["figures"] / f"figure_training_curve_{name}.{fmt}", dpi=dpi, save_svg=save_svg)

    print(f"Figures saved to {dirs['figures']}")


if __name__ == "__main__":
    main()
