# Dataset Preparation Instructions

This repository does not include raw EEG recordings. Use the official CHB-MIT and SIENA dataset sources and prepare the recordings locally.

## Required Processed File Format

Create a processed file:

```text
data/processed/eeg_windows.npz
```

It must contain these arrays:

```text
windows      float32, shape [N, C, T]
labels       int64,   shape [N]
subject_ids  string,  shape [N]
dataset_names string, shape [N] optional
```

Label convention:

```text
0 = interictal / non-seizure
1 = preictal or seizure-related positive class, depending on the manuscript protocol
```

## Preprocessing Protocol Implemented

The preprocessing code in `src/preprocessing.py` supports the following steps:

1. EDF loading through MNE when raw EDF files are available.
2. 50 Hz notch filtering for power-line interference removal.
3. 0.5–70 Hz band-pass filtering.
4. Non-overlapping 10-second EEG window segmentation.
5. Per-channel min-max normalization inside each window.
6. Subject-wise indexing for leave-one-subject-out evaluation.

## Recommended Preparation Workflow

1. Download the public EEG dataset files.
2. Use consistent EEG channel selection across subjects.
3. Apply the same filtering and windowing settings from `config.yaml`.
4. Assign labels using the expert seizure annotations provided with each dataset.
5. Save all windows into one NPZ file with the required arrays.
6. Set `data.mode: npz` in `config.yaml`.
7. Run `bash run_reproducibility.sh`.

## Preventing Data Leakage

The repository creates LOSO folds by holding out one subject as the target subject. Source labels are used for supervised classification. Target labels are used only for final evaluation, not for optimization. During unsupervised domain adaptation, the target EEG windows are used without labels.
