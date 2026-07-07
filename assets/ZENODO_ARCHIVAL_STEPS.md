# DOI-Assigning Repository Archival Steps

A GitHub repository alone does not normally satisfy the reviewer's DOI-deposition request. After finalizing the code, archive the exact release in a DOI-assigning repository.

## Recommended Steps

1. Create a GitHub repository and upload this complete code package.
2. Confirm that `README.md`, `requirements.txt`, `config.yaml`, and the reproducibility scripts are visible.
3. Run the smoke test and commit the final version.
4. Create a versioned GitHub release, for example `v1.0.0`.
5. Archive the release in Zenodo or another recognized DOI-assigning repository.
6. Copy the generated DOI.
7. Replace the placeholder text in `assets/CODE_AVAILABILITY.md` and the manuscript Code Availability section.
8. Mention the archived DOI in the response letter.

## Suggested Release Title

DGANM EEG Domain Adaptation Code for Cross-Subject Seizure Detection

## Suggested Description

This archived code release contains the implementation of the Deep Generative Adversarial Network Model for EEG-based seizure detection with cross-subject domain adaptation. It includes preprocessing utilities, DGANM model components, LOSO evaluation, metric computation, and high-resolution figure generation scripts.
