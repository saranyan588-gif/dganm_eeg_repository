"""Training and evaluation loops for DGANM."""
from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import binary_metrics
from .model import DGANM, gaussian_mmd


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False, pin_memory=torch.cuda.is_available())


def _batch_to_device(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return batch["x"].to(device, non_blocking=True), batch["y"].to(device, non_blocking=True)


def train_one_loso_fold(
    model: DGANM,
    source_loader: DataLoader,
    target_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    fold_name: str,
    out_dir: Path,
) -> Dict[str, float]:
    train_cfg = cfg.get("training", {})
    lr = float(train_cfg.get("learning_rate", 0.001))
    momentum = float(train_cfg.get("momentum", 0.9))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 8))
    patience = int(train_cfg.get("patience", 10))
    lambda_adv = float(train_cfg.get("lambda_adv", 1.0))
    lambda_cls = float(train_cfg.get("lambda_cls", 1.0))
    lambda_mmd = float(train_cfg.get("lambda_mmd", 0.05))
    real_value = float(train_cfg.get("real_label_smooth", 0.9))
    fake_value = float(train_cfg.get("fake_label_value", -0.9))

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    opt_g = torch.optim.SGD(model.generator.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    opt_d = torch.optim.SGD(model.discriminator.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    opt_c = torch.optim.SGD(model.classifier.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    best_f1 = -1.0
    best_path = out_dir / "models" / f"dganm_{fold_name}.pt"
    log_rows = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        d_losses, c_losses, g_losses = [], [], []
        target_iter = cycle(target_loader)
        pbar = tqdm(source_loader, desc=f"{fold_name} epoch {epoch}/{epochs}", leave=False)
        for source_batch in pbar:
            target_batch = next(target_iter)
            xs, ys = _batch_to_device(source_batch, device)
            xt, _ = _batch_to_device(target_batch, device)

            bs = xs.size(0)
            bt = xt.size(0)
            real_labels = torch.full((bt,), real_value, device=device)
            fake_labels = torch.full((bs,), fake_value, device=device)

            # Step 1: discriminator update
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                xg_detached = model.generator(xs)
            d_real = model.discriminator(xt)
            d_fake = model.discriminator(xg_detached)
            d_loss = 0.5 * (mse(d_real, real_labels) + mse(d_fake, fake_labels))
            d_loss.backward()
            nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=5.0)
            opt_d.step()

            # Step 2: classifier update on source and generated source-like labels
            opt_c.zero_grad(set_to_none=True)
            with torch.no_grad():
                xg_for_c = model.generator(xs)
            logits_src = model.classifier(xs)
            logits_gen = model.classifier(xg_for_c)
            c_loss = 0.5 * (ce(logits_src, ys) + ce(logits_gen, ys))
            c_loss.backward()
            nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=5.0)
            opt_c.step()

            # Step 3: generator update guided by discriminator and classifier
            opt_g.zero_grad(set_to_none=True)
            xg = model.generator(xs)
            dg = model.discriminator(xg)
            logits_g, feat_g = model.classifier(xg, return_features=True)
            with torch.no_grad():
                _, feat_t = model.classifier(xt, return_features=True)
            adv_loss = mse(dg, torch.full((bs,), real_value, device=device))
            cls_loss = ce(logits_g, ys)
            mmd_loss = gaussian_mmd(feat_g, feat_t)
            g_loss = lambda_adv * adv_loss + lambda_cls * cls_loss + lambda_mmd * mmd_loss
            g_loss.backward()
            nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=5.0)
            opt_g.step()

            d_losses.append(float(d_loss.detach().cpu()))
            c_losses.append(float(c_loss.detach().cpu()))
            g_losses.append(float(g_loss.detach().cpu()))
            pbar.set_postfix(d=np.mean(d_losses), c=np.mean(c_losses), g=np.mean(g_losses))

        val_metrics = evaluate_model(model, val_loader, device=device)
        log_row = {
            "fold": fold_name,
            "epoch": epoch,
            "d_loss": float(np.mean(d_losses)),
            "c_loss": float(np.mean(c_losses)),
            "g_loss": float(np.mean(g_losses)),
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))},
        }
        log_rows.append(log_row)
        pd.DataFrame(log_rows).to_csv(out_dir / "logs" / f"training_{fold_name}.csv", index=False)

        current = float(val_metrics.get("f1", 0.0))
        if current > best_f1:
            best_f1 = current
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "fold": fold_name, "best_f1": best_f1}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    return {"best_val_f1": float(best_f1), "checkpoint": str(best_path)}


@torch.no_grad()
def evaluate_model(model: DGANM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_y, all_pred, all_score = [], [], []
    for batch in loader:
        x, y = _batch_to_device(batch, device)
        logits = model.classifier(x)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_score.extend(prob.detach().cpu().numpy().tolist())
    return binary_metrics(all_y, all_pred, all_score)


@torch.no_grad()
def predict_dataframe(model: DGANM, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        x, y = _batch_to_device(batch, device)
        logits = model.classifier(x)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        subjects = batch["subject_id"]
        names = batch["dataset_name"]
        for i in range(len(y_np)):
            rows.append({
                "subject_id": subjects[i],
                "dataset_name": names[i],
                "y_true": int(y_np[i]),
                "y_pred": int(pred[i]),
                "score_preictal": float(prob[i]),
            })
    return pd.DataFrame(rows)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
