"""Configuration and reproducibility helpers for the DGANM EEG repository."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a YAML configuration file and normalize common paths."""
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg is None:
        raise ValueError(f"Configuration file is empty: {path}")
    root = Path(path).resolve().parent
    cfg.setdefault("_root", str(root))
    return cfg


def resolve_path(cfg: Dict[str, Any], path: str | os.PathLike[str]) -> Path:
    """Resolve relative paths against the repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(cfg.get("_root", ".")).resolve() / p


def output_dir(cfg: Dict[str, Any]) -> Path:
    out = resolve_path(cfg, cfg.get("project", {}).get("output_dir", "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def seed_everything(seed: int = 42) -> None:
    """Make CPU/GPU runs as deterministic as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg.get("training", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(requested)


def ensure_subdirs(out: Path) -> Dict[str, Path]:
    dirs = {
        "models": out / "models",
        "metrics": out / "metrics",
        "figures": out / "figures",
        "predictions": out / "predictions",
        "logs": out / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs
