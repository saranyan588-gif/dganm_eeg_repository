# Reviewer-Facing Code Response

The reviewer requested deposition of the underlying code because the manuscript reports a custom computational pipeline and a proposed algorithm. This repository directly addresses that requirement by providing the executable DGANM implementation and the complete reproducibility workflow.

## What Has Been Added

1. Full PyTorch implementation of the proposed DGANM architecture:
   - Generator
   - Discriminator
   - Classifier
   - classifier-guided adversarial optimization
   - MMD-based feature alignment term

2. Reproducible preprocessing utilities:
   - notch filtering
   - band-pass filtering
   - 10-second window segmentation
   - per-channel min-max normalization

3. Cross-subject evaluation:
   - leave-one-subject-out fold creation
   - target-subject adaptation without target labels
   - final target-subject testing

4. Metric reporting:
   - accuracy
   - precision
   - sensitivity
   - specificity
   - F1-score
   - AUC
   - confusion matrix values

5. Figure regeneration:
   - high-resolution PNG figures
   - SVG vector copies
   - plots generated directly from CSV outputs

## Recommended Response Letter Text

> We thank the reviewer for pointing out the requirement for code deposition. We have prepared a dedicated reproducibility repository containing the implementation of the proposed DGANM pipeline, including preprocessing, model architecture, classifier-guided adversarial domain adaptation, leave-one-subject-out evaluation, metric computation, and high-resolution figure generation. The repository has been organized to allow independent execution from locally prepared CHB-MIT and SIENA EEG data. After creating a stable release, the repository will be archived in a DOI-assigning repository and the generated DOI will be inserted in the revised Code Availability section.

## Figure-Quality Action

The plotting scripts regenerate figures from numerical CSV files at 600 dpi and optionally as SVG vector images. The revised manuscript figures should be replaced using these generated files rather than screenshots or low-resolution resized images.

## Reference-Consistency Action

The missing references identified by the reviewer must be added to the manuscript reference section itself. This repository does not fabricate or infer bibliographic entries. The exact works corresponding to Batista et al., Zerbe et al., Doležalová et al., and Rijal et al. should be verified against the cited manuscript text before final insertion.
