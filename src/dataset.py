"""Dataset loading and LOSO split helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .config import resolve_path


@dataclass
class EEGArrays:
    windows: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    dataset_names: np.ndarray


class EEGWindowDataset(Dataset):
    """Torch dataset for fixed-length EEG windows shaped [channels, time]."""

    def __init__(self, arrays: EEGArrays, indices: Sequence[int] | None = None):
        self.arrays = arrays
        self.indices = np.asarray(indices if indices is not None else np.arange(len(arrays.labels)), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor | str]:
        idx = int(self.indices[item])
        x = torch.tensor(self.arrays.windows[idx], dtype=torch.float32)
        y = torch.tensor(int(self.arrays.labels[idx]), dtype=torch.long)
        return {
            "x": x,
            "y": y,
            "subject_id": str(self.arrays.subject_ids[idx]),
            "dataset_name": str(self.arrays.dataset_names[idx]),
        }


def load_npz_dataset(path: str | Path) -> EEGArrays:
    """Load a processed EEG NPZ file.

    Required arrays: windows [N, C, T], labels [N], subject_ids [N].
    Optional: dataset_names [N].
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Processed dataset not found: {p}")
    data = np.load(p, allow_pickle=True)
    required = ["windows", "labels", "subject_ids"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"NPZ is missing required arrays: {missing}")
    windows = data["windows"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    subject_ids = data["subject_ids"].astype(str)
    dataset_names = data["dataset_names"].astype(str) if "dataset_names" in data else np.asarray(["unknown"] * len(labels))
    validate_arrays(windows, labels, subject_ids)
    return EEGArrays(windows=windows, labels=labels, subject_ids=subject_ids, dataset_names=dataset_names)


def validate_arrays(windows: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray) -> None:
    if windows.ndim != 3:
        raise ValueError("windows must be shaped [N, channels, time].")
    n = windows.shape[0]
    if len(labels) != n or len(subject_ids) != n:
        raise ValueError("windows, labels, and subject_ids must contain the same number of samples.")
    if np.any(~np.isfinite(windows)):
        raise ValueError("windows contain NaN or infinite values.")
    unique_labels = set(labels.astype(int).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError("This implementation expects binary labels: 0=interictal, 1=preictal/seizure.")


def create_synthetic_eeg_dataset(
    n_subjects: int = 6,
    windows_per_subject: int = 90,
    n_channels: int = 19,
    time_points: int = 2560,
    seed: int = 42,
) -> EEGArrays:
    """Create deterministic EEG-like data for smoke testing the repository.

    The generated data are not clinical data. They only verify that the DGANM
    pipeline, metrics, LOSO split, and figure generation run end-to-end.
    """
    rng = np.random.default_rng(seed)
    windows, labels, subjects, names = [], [], [], []
    t = np.linspace(0, 10, time_points, dtype=np.float32)
    base_freqs = np.asarray([3.0, 6.0, 10.0, 18.0, 30.0], dtype=np.float32)
    for sid in range(n_subjects):
        subject_shift = rng.normal(0.0, 0.35, size=(n_channels, 1)).astype(np.float32)
        subject_scale = rng.uniform(0.85, 1.25, size=(n_channels, 1)).astype(np.float32)
        for i in range(windows_per_subject):
            y = int(rng.random() < 0.42)
            signal = np.zeros((n_channels, time_points), dtype=np.float32)
            for ch in range(n_channels):
                mixture = np.zeros(time_points, dtype=np.float32)
                for f in base_freqs:
                    amp = rng.uniform(0.02, 0.18)
                    phase = rng.uniform(0, 2 * np.pi)
                    mixture += amp * np.sin(2 * np.pi * (f + 0.06 * sid) * t + phase)
                if y == 1:
                    burst_center = rng.uniform(2.0, 8.0)
                    burst = np.exp(-0.5 * ((t - burst_center) / 0.45) ** 2)
                    mixture += rng.uniform(0.35, 0.70) * burst * np.sin(2 * np.pi * rng.uniform(8.0, 16.0) * t)
                noise = rng.normal(0.0, 0.08, size=time_points).astype(np.float32)
                signal[ch] = mixture + noise
            signal = signal * subject_scale + subject_shift
            # within-window min-max normalization per channel
            mn = signal.min(axis=-1, keepdims=True)
            mx = signal.max(axis=-1, keepdims=True)
            signal = 2.0 * (signal - mn) / (mx - mn + 1e-8) - 1.0
            windows.append(signal.astype(np.float32))
            labels.append(y)
            subjects.append(f"S{sid + 1:02d}")
            names.append("synthetic-eeg")
    arrays = EEGArrays(
        windows=np.stack(windows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subjects),
        dataset_names=np.asarray(names),
    )
    validate_arrays(arrays.windows, arrays.labels, arrays.subject_ids)
    return arrays


def load_dataset_from_config(cfg: Dict) -> EEGArrays:
    mode = str(cfg.get("data", {}).get("mode", "synthetic")).lower()
    data_cfg = cfg.get("data", {})
    if mode == "npz":
        return load_npz_dataset(resolve_path(cfg, data_cfg.get("npz_path", "data/processed/eeg_windows.npz")))
    if mode == "synthetic":
        fs = int(data_cfg.get("sampling_rate", 256))
        seconds = int(data_cfg.get("window_seconds", 10))
        return create_synthetic_eeg_dataset(
            n_subjects=int(data_cfg.get("synthetic_subjects", 6)),
            windows_per_subject=int(data_cfg.get("synthetic_windows_per_subject", 90)),
            n_channels=int(data_cfg.get("n_channels", 19)),
            time_points=fs * seconds,
            seed=int(cfg.get("project", {}).get("seed", 42)),
        )
    raise ValueError(f"Unsupported data.mode: {mode}")


def subject_ids(arrays: EEGArrays) -> List[str]:
    return sorted(np.unique(arrays.subject_ids).astype(str).tolist())


def loso_indices(arrays: EEGArrays, target_subject: str, val_fraction: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation, and test indices for one target subject."""
    all_idx = np.arange(len(arrays.labels))
    target_mask = arrays.subject_ids.astype(str) == str(target_subject)
    test_idx = all_idx[target_mask]
    source_idx = all_idx[~target_mask]
    if len(test_idx) == 0:
        raise ValueError(f"No samples found for target subject: {target_subject}")
    if len(source_idx) == 0:
        raise ValueError("LOSO requires at least two subjects.")
    rng = np.random.default_rng(seed)
    rng.shuffle(source_idx)
    n_val = max(1, int(round(len(source_idx) * val_fraction)))
    val_idx = source_idx[:n_val]
    train_idx = source_idx[n_val:]
    return train_idx, val_idx, test_idx


def make_domain_pair_indices(arrays: EEGArrays, target_subject: str, val_fraction: float, seed: int) -> Dict[str, np.ndarray]:
    train_idx, val_idx, test_idx = loso_indices(arrays, target_subject, val_fraction=val_fraction, seed=seed)
    source_train_idx = train_idx
    # During unsupervised adaptation, target windows are available without labels; labels are never used in loss.
    target_adapt_idx = test_idx.copy()
    return {
        "source_train": source_train_idx,
        "source_val": val_idx,
        "target_adapt": target_adapt_idx,
        "target_test": test_idx,
    }
