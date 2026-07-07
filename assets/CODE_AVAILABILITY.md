# Code Availability Statement

The complete source code used to implement the proposed Deep Generative Adversarial Network Model (DGANM), including preprocessing utilities, model definitions, classifier-guided adversarial training, leave-one-subject-out evaluation, metric computation, and high-resolution figure generation, is provided in the accompanying public repository.

The repository contains scripts for reproducing the full computational workflow:

```text
train.py
evaluate.py
make_figures.py
run_reproducibility.sh
```

The implementation supports public EEG datasets after local preparation into the documented NPZ format. Raw CHB-MIT and SIENA EEG recordings are not redistributed with the code package and must be obtained from their official public dataset sources according to their terms of use.

Repository DOI: **to be inserted after Zenodo or another DOI-assigning repository archives the released code version.**

Recommended manuscript wording after archival:

> The source code for preprocessing, model training, leave-one-subject-out evaluation, metric computation, and figure generation is publicly available in the archived repository at [insert repository DOI]. The repository contains the exact DGANM implementation used in this study, together with configuration files and instructions for reproducing the experiments from locally prepared CHB-MIT and SIENA EEG data.
