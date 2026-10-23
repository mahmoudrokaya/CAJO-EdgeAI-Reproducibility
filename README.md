# CAJO Edge AI — Evidence-filtered reproducibility repository

Companion to *Constraint-Aware Configuration Search for Edge AI: A
Multi-Dataset Study of Architecture Selection, Pruning, Knowledge
Distillation, and Quantization*.

## Repository structure

- `src/original/`: selected original experimental Python scripts.
- `protocol/`: original configuration and dataset-verification records.
- `data/split_manifests/`: available split metadata, not raw datasets.
- `results/section_4_1/`: architecture screening and controlled training.
- `results/section_4_2/`: quantization, deployment measurements and paired comparisons.
- `results/section_4_3/`: provisional Pareto and historical policy evidence.
- `figures/`: original figure files found during collection.
- `provenance/`: SHA256 manifest, omissions and build report.

## Large files: external companion archive

Files **strictly larger than 25 MiB** (26,214,400 bytes) are
excluded from the GitHub folder and placed in the sibling folder
`CAJO-EdgeAI-Reproducibility-External/`, preserving their **exact repository-relative paths**.

These large files will be shared at the following Taif University
SharePoint/OneDrive location:

https://taifedusa-my.sharepoint.com/:f:/r/personal/mahmoudrokaya_tu_edu_sa/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmahmoudrokaya%5Ftu%5Fedu%5Fsa%2FDocuments%2F48%2F481%2FToward%5FLightweight%5FEdge%5FIntelligence&share=cgrNdZT7rYK5Try9UdQPyicCEgUCWEoYAe03SdEM5rkjN713dg

After uploading the contents of `CAJO-EdgeAI-Reproducibility-External/` to the linked SharePoint
folder, restore them by copying the external tree **into the root of this
GitHub repository**, preserving directory names. Only paths that contain
files are created; empty directories are not included.

For the exact external-file list, original sizes and SHA256 checksums, see
`provenance/external_files.csv`. For all selected files see
`provenance/file_manifest.csv`. Note: the link above is a *planned storage
location*, not a claim that the files have already been uploaded or that
public access has been tested.

## Scientific scope

- Section 4.1: 24 FP32 five-epoch screening configurations and 54
  controlled 20-epoch training runs.
- Section 4.2: 36 precision-specific deployment records forming 18
  matched FP32/INT8 pairs.
- Section 4.3: 11 provisional three-objective nondominated candidates
  and 18 historical policy comparisons.
- Validation metrics are **not held-out test results**.
- Native peak inference memory and per-inference CPU energy are not
  fully certified; three-objective results are **not five-objective Pareto claims**.
- Historical policy differences do not prove causal search efficiency.

## Reproducibility and licensing

This is an evidence-filtered snapshot, not an independently validated
end-to-end reproduction environment. Original scripts may require
additional dependencies, teacher checkpoints, data splits and original
directory layouts. Review `provenance/omissions.csv`.

Raw datasets and model exports are excluded by default; do not
redistribute datasets or model weights without checking their licenses.
The repository's `requirements.txt` is a starting point, not a frozen
environment lockfile.
