#!/usr/bin/env python
"""Train DGANM under a leave-one-subject-out domain adaptation protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.config import ensure_subdirs, get_device, load_config, output_dir, seed_everything
from src.dataset import EEGWindowDataset, load_dataset_from_config, make_domain_pair_indices, subject_ids
from src.model import DGANM, count_parameters
from src.train_utils import evaluate_model, make_loader, predict_dataframe, save_json, train_one_loso_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DGANM for cross-subject EEG seizure detection.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny deterministic run for repository verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("project", {}).get("seed", 42))
    seed_everything(seed)

    if args.smoke_test:
        cfg.setdefault("data", {})["mode"] = "synthetic"
        cfg["data"]["synthetic_subjects"] = 2
        cfg["data"]["synthetic_windows_per_subject"] = 4
        cfg["data"]["sampling_rate"] = 16
        cfg["data"]["window_seconds"] = 1
        cfg["data"]["n_channels"] = 2
        cfg.setdefault("model", {})["generator_filters"] = [4]
        cfg["model"]["discriminator_filters"] = [4]
        cfg["model"]["classifier_filters"] = [4]
        cfg["model"]["classifier_hidden"] = 8
        cfg.setdefault("training", {})["epochs"] = 1
        cfg["training"]["batch_size"] = 4

    out = output_dir(cfg)
    dirs = ensure_subdirs(out)
    arrays = load_dataset_from_config(cfg)
    device = get_device(cfg)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    subjects = data_cfg.get("loso_subjects") or subject_ids(arrays)
    batch_size = int(train_cfg.get("batch_size", 72))
    num_workers = int(train_cfg.get("num_workers", 0))
    val_fraction = float(train_cfg.get("validation_fraction", 0.15))

    all_fold_metrics = []
    all_predictions = []

    print(f"Loaded dataset: windows={arrays.windows.shape}, labels={arrays.labels.shape}, subjects={len(subject_ids(arrays))}")
    print(f"Using device: {device}")

    for target_subject in subjects:
        fold_name = f"target_{target_subject}".replace("/", "_").replace(" ", "_")
        split = make_domain_pair_indices(arrays, target_subject=target_subject, val_fraction=val_fraction, seed=seed)
        source_ds = EEGWindowDataset(arrays, split["source_train"])
        val_ds = EEGWindowDataset(arrays, split["source_val"])
        target_adapt_ds = EEGWindowDataset(arrays, split["target_adapt"])
        target_test_ds = EEGWindowDataset(arrays, split["target_test"])

        source_loader = make_loader(source_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        target_loader = make_loader(target_adapt_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = make_loader(target_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = DGANM(cfg).to(device)
        if target_subject == subjects[0]:
            save_json(dirs["logs"] / "model_parameter_count.json", {"trainable_parameters": count_parameters(model)})

        print(f"\n=== LOSO fold: target subject {target_subject} ===")
        print(f"source_train={len(source_ds)}, source_val={len(val_ds)}, target_unlabeled={len(target_adapt_ds)}, target_test={len(target_test_ds)}")

        train_info = train_one_loso_fold(
            model=model,
            source_loader=source_loader,
            target_loader=target_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            fold_name=fold_name,
            out_dir=out,
        )
        test_metrics = evaluate_model(model, test_loader, device=device)
        pred_df = predict_dataframe(model, test_loader, device=device)
        pred_df.insert(0, "fold", fold_name)
        pred_df.to_csv(dirs["predictions"] / f"predictions_{fold_name}.csv", index=False)
        all_predictions.append(pred_df)

        row = {
            "fold": fold_name,
            "target_subject": str(target_subject),
            "source_train_windows": len(source_ds),
            "source_validation_windows": len(val_ds),
            "target_unlabeled_windows": len(target_adapt_ds),
            "target_test_windows": len(target_test_ds),
            "checkpoint": train_info.get("checkpoint", ""),
            **test_metrics,
        }
        all_fold_metrics.append(row)
        pd.DataFrame(all_fold_metrics).to_csv(dirs["metrics"] / "loso_metrics.csv", index=False)
        print({k: row[k] for k in ["accuracy", "f1", "auc", "sensitivity", "specificity"]})

    metrics_df = pd.DataFrame(all_fold_metrics)
    pred_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    pred_all.to_csv(dirs["predictions"] / "predictions_all_folds.csv", index=False)

    summary = metrics_df[["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]].agg(["mean", "std"]).T
    summary.to_csv(dirs["metrics"] / "summary_metrics.csv")
    print("\nSummary metrics")
    print(summary)
    print(f"\nSaved outputs to: {out}")


if __name__ == "__main__":
    main()
