"""EEG preprocessing utilities for CHB-MIT/SIENA style seizure detection.

The functions keep preprocessing explicit and reproducible: notch filtering,
band-pass filtering, fixed-window segmentation, and within-window min-max
normalization. The manuscript can cite this module as the executable form of the
preprocessing pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


@dataclass(frozen=True)
class WindowRecord:
    """Metadata for one segmented EEG window."""

    subject_id: str
    start_sample: int
    end_sample: int
    label: int
    source_file: str = ""


def notch_filter(x: np.ndarray, fs: float, notch_hz: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Remove power-line noise using a zero-phase IIR notch filter."""
    if notch_hz <= 0 or notch_hz >= fs / 2:
        return x
    b, a = iirnotch(w0=notch_hz, Q=quality, fs=fs)
    return filtfilt(b, a, x, axis=-1)


def bandpass_filter(x: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 70.0, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth band-pass filtering along the time axis."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        raise ValueError(f"Invalid bandpass range: {low_hz}-{high_hz} Hz for fs={fs}")
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)


def minmax_normalize_window(x: np.ndarray, feature_range: Tuple[float, float] = (-1.0, 1.0), eps: float = 1e-8) -> np.ndarray:
    """Normalize each EEG channel inside one window to a fixed range."""
    lo, hi = feature_range
    x_min = x.min(axis=-1, keepdims=True)
    x_max = x.max(axis=-1, keepdims=True)
    scaled = (x - x_min) / (x_max - x_min + eps)
    return scaled * (hi - lo) + lo


def preprocess_signal(
    eeg: np.ndarray,
    fs: float,
    notch_hz: float = 50.0,
    low_hz: float = 0.5,
    high_hz: float = 70.0,
    feature_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Filter and normalize an EEG array shaped [channels, samples]."""
    if eeg.ndim != 2:
        raise ValueError("Expected EEG array shaped [channels, samples].")
    y = notch_filter(eeg.astype(np.float32), fs=fs, notch_hz=notch_hz)
    y = bandpass_filter(y, fs=fs, low_hz=low_hz, high_hz=high_hz)
    return minmax_normalize_window(y, feature_range=feature_range).astype(np.float32)


def segment_windows(
    eeg: np.ndarray,
    fs: float,
    window_seconds: float = 10.0,
    annotations: Optional[Sequence[Tuple[float, float, int]]] = None,
    subject_id: str = "subject",
    source_file: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[WindowRecord]]:
    """Segment EEG into non-overlapping windows.

    Parameters
    ----------
    eeg:
        Array shaped [channels, samples].
    annotations:
        Optional list of seizure/preictal intervals as (start_seconds, end_seconds, label).
        A window receives the positive label when its center falls inside a positive interval.
    """
    window_len = int(round(window_seconds * fs))
    if window_len <= 0:
        raise ValueError("window_seconds must be positive.")
    n_samples = eeg.shape[-1]
    windows, labels, records = [], [], []
    annotations = annotations or []
    for start in range(0, n_samples - window_len + 1, window_len):
        end = start + window_len
        center_sec = (start + end) / 2.0 / fs
        label = 0
        for ann_start, ann_end, ann_label in annotations:
            if ann_start <= center_sec <= ann_end:
                label = int(ann_label)
                break
        windows.append(eeg[:, start:end])
        labels.append(label)
        records.append(WindowRecord(subject_id=subject_id, start_sample=start, end_sample=end, label=label, source_file=source_file))
    if not windows:
        raise ValueError("Signal is shorter than one configured window.")
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64), records


def load_edf_with_mne(path: str | Path, pick_channels: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, float, List[str]]:
    """Load an EDF file using MNE when raw CHB-MIT/SIENA files are available."""
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for EDF loading. Install with `pip install mne`.") from exc
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if pick_channels:
        raw.pick_channels(list(pick_channels))
    data = raw.get_data().astype(np.float32)
    return data, float(raw.info["sfreq"]), list(raw.ch_names)
