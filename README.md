# DGANM EEG Domain Adaptation Repository

This repository provides the executable code package for the study titled **EEG-Based Seizure Detection Using Deep Generative Models with Domain Adaptation for Cross-Subject Generalization**.

The implementation reproduces the computational workflow described in the paper: EEG preprocessing, leave-one-subject-out cross-subject evaluation, classifier-guided adversarial domain adaptation, metric reporting, and high-resolution figure generation.

## Repository Structure

```text
DGANM_EEG_Domain_Adaptation/
├── README.md
├── config.yaml
├── requirements.txt
├── run_reproducibility.sh
├── train.py
├── evaluate.py
├── make_figures.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train_utils.py
│   ├── metrics.py
│   ├── baselines.py
│   └── visualization.py
└── assets/
    ├── CODE_AVAILABILITY.md
    ├── DATASET_INSTRUCTIONS.md
    ├── REVIEWER_RESPONSE_CODE.md
    ├── REPRODUCIBILITY_CHECKLIST.md
    └── ZENODO_ARCHIVAL_STEPS.md
```

Only two folders are used: `src/` for executable implementation files and `assets/` for reviewer-facing documentation.

## Core Method Implemented

The proposed Deep Generative Adversarial Network Model (DGANM) contains three jointly trained modules:

1. **Generator**: maps source-subject EEG windows into target-aligned feature representations.
2. **Discriminator**: distinguishes real target-subject EEG representations from generated target-like representations.
3. **Classifier**: performs seizure/non-seizure prediction and guides the generator so that label-specific seizure information is preserved during domain alignment.

The training loop uses alternating optimization:

1. update the discriminator using real target windows and generated source windows;
2. update the classifier using labeled source and generated source windows;
3. update the generator using adversarial loss, classifier consistency loss, and a soft MMD feature-alignment term.

This implements the classifier-in-the-loop adversarial alignment described in the manuscript.

## Installation

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Verification Run

A deterministic EEG-like synthetic dataset is included through code generation only. It is not clinical data; it is used to verify that the repository runs end-to-end.

```bash
python train.py --config config.yaml --smoke-test
python evaluate.py --config config.yaml
python make_figures.py --config config.yaml
```

## Full Reproducibility Run

```bash
bash run_reproducibility.sh
```

Outputs are written to:

```text
outputs/
├── figures/
├── logs/
├── metrics/
├── models/
└── predictions/
```

The generated figures are exported at 600 dpi and also saved as SVG files when enabled in `config.yaml`. This avoids screen-captured or distorted plots.

## Using CHB-MIT or SIENA Data

Raw public EEG recordings are not included in this repository. After preparing the datasets according to the instructions in `assets/DATASET_INSTRUCTIONS.md`, save the processed array file as:

```text
data/processed/eeg_windows.npz
```

The NPZ file must contain:

```text
windows      shape [N, channels, time]
labels       shape [N], binary: 0 = interictal, 1 = preictal/seizure
subject_ids  shape [N]
```

Then update `config.yaml`:

```yaml
data:
  mode: npz
  npz_path: data/processed/eeg_windows.npz
```

Run:

```bash
bash run_reproducibility.sh
```

## Metrics Reported

The evaluation scripts report:

- Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- F1-score
- AUC
- TP, TN, FP, FN counts

The primary cross-subject evaluation uses leave-one-subject-out testing. Target-subject labels are not used for optimization during training.

## Figure Generation

`make_figures.py` regenerates high-resolution figures from saved CSV files. It creates:

- Cross-subject LOSO metric comparison
- Confusion matrix
- ROC curve
- Adversarial training loss curves

These figures are produced directly from numerical results and are suitable for manuscript replacement because they are not resized screenshots.

## Code Availability Statement for Manuscript

A reviewer-facing statement is provided in:

```text
assets/CODE_AVAILABILITY.md
```

After uploading this repository to GitHub, create a versioned release and archive it through a DOI-assigning repository such as Zenodo. Replace the placeholder in the code availability statement with the generated repository DOI.

## Important Reproducibility Note

The synthetic mode validates software execution only. Manuscript-level numerical claims must be generated with the public CHB-MIT and SIENA datasets using the same preprocessing and LOSO evaluation protocol.
