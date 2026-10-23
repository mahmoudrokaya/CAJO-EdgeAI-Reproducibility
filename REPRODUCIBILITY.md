# Reproducibility Guide

> **Archived software release:** [CAJO Edge AI v1.0.0](https://github.com/mahmoudrokaya/CAJO-EdgeAI-Reproducibility/releases/tag/v1.0.0) · **Version DOI:** [10.5281/zenodo.23602180](https://doi.org/10.5281/zenodo.23602180). Documentation changes made after the release are not part of the frozen v1.0.0 archive.

**Project:** CAJO Edge AI Reproducibility Repository  
**Study:** *Constraint-Aware Configuration Search for Edge AI: A Multi-Dataset Study of Architecture Selection, Pruning, Knowledge Distillation, and Quantization*  
**Scope:** Evidence underlying manuscript Sections 4.1–4.3  
**License:** CC BY 4.0 for original materials released under this repository's license; third-party materials retain their own terms.

## 1. Purpose and reproducibility status

This repository is an evidence-filtered record of the experiments reported in the manuscript. It is designed to make the configuration choices, measurements, matched comparisons, and limitations inspectable. **It is not yet a verified, one-command, end-to-end reproduction package.** Some original scripts may depend on unbundled datasets, pretrained teachers, checkpoints, local paths, or environment-specific packages. The repository does not claim to reproduce experiments merely because their scripts and summary CSV files are present.

The principal experiments are:

| Manuscript | Experiment | Reported experimental scope |
| --- | --- | --- |
| §4.1 | FP32 architecture/compression screening | 24 configurations, five epochs, seed 42 |
| §4.1 | Controlled recovery and ordering comparisons | 54 training runs, 20 epochs, seeds 42, 43, 44 |
| §4.2 | FP32 and INT8 CPU deployment evaluation | 36 precision-specific deployment artifacts forming 18 checkpoint-matched pairs |
| §4.3 | Constraint-based configuration evaluation | Provisional nondominated configurations over validation accuracy, serialized storage, and mean CPU latency |
| §4.3 | Historical search-policy comparisons | 18 matched historical comparisons; not a causal demonstration of search-efficiency improvement |

The study uses **CIFAR-10, CIFAR-100, and Speech Commands v0.02**. The evaluation unit and matching keys should be preserved when analyzing repeated seeds or precision pairs. Do not mix validation and held-out test metrics.

## 2. Repository layout

```text
src/original/                  Original, selected experimental scripts
protocol/                      Experiment configuration and verification records
data/split_manifests/          Available dataset/split manifests
results/section_4_1/          Screening, training and controlled comparisons
results/section_4_2/          Precision pairs, deployment metrics and export metadata
results/section_4_3/          Feasibility, provisional frontier and policy comparisons
figures/                       Original figure assets, if available
models/                        Optional measured model exports
provenance/file_manifest.csv   Selected files and SHA256 checksums
provenance/omissions.csv       Missing/unavailable inputs
provenance/external_files.csv  Files supplied separately due to size
provenance/build_report.json  Collection report and unresolved warnings
```

A directory is absent when it has no selected files. The manifest, rather than this illustrative tree, is authoritative for the deposited release.

## 3. Retrieve the complete release

The GitHub repository contains files up to the collection threshold of **25 MiB** (26,214,400 bytes). Selected larger files are staged in a separate companion directory, with **identical repository-relative paths**.

The intended SharePoint/OneDrive location for the companion files is:

<https://taifedusa-my.sharepoint.com/:f:/r/personal/mahmoudrokaya_tu_edu_sa/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmahmoudrokaya%5Ftu%5Fedu%5Fsa%2FDocuments%2F48%2F481%2FToward%5FLightweight%5FEdge%5FIntelligence&share=cgrNdZT7rYK5Try9UdQPyicCEgUCWEoYAe03SdEM5rkjN713dg>

**Availability caveat:** This link is the intended distribution location. Availability, permissions, file completeness, and long-term persistence must be checked before release. An archived GitHub DOI does not automatically archive externally hosted files.

To reconstruct the complete directory, copy the **contents** of `CAJO-EdgeAI-Reproducibility-External/` into the root of `CAJO-EdgeAI-Reproducibility/`, preserving subdirectories. Do not rename files. Verify every external file against `provenance/external_files.csv` (path, byte count, SHA256).

Example integrity check, from the repository root:

```bash
python - <<'PY'
import csv, hashlib
from pathlib import Path
manifest = Path('provenance/external_files.csv')
if not manifest.exists():
    raise SystemExit('Missing external_files.csv; check release completeness')
with manifest.open(newline='', encoding='utf-8') as stream:
    records = list(csv.DictReader(stream))
errors = []
for record in records:
    path = Path(record['relative_path'])
    if not path.is_file():
        errors.append(f'MISSING {path}')
        continue
    expected_size = int(record['size'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != expected_size or digest.hexdigest().lower() != record['sha256'].lower():
        errors.append(f'MISMATCH {path}')
print(f'Checked {len(records)} external files; errors: {len(errors)}')
for error in errors:
    print(error)
if errors:
    raise SystemExit(1)
PY
```

The verification above checks the **external files only**; `provenance/file_manifest.csv` lists the remaining selected evidence.

## 4. Environment and dependencies

The source experiments use Python and the PyTorch ecosystem. A minimal starting environment is:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Do not treat `requirements.txt` as a tested lockfile.** Verify the Python version, exact PyTorch/torchvision/torchaudio versions, GPU/CUDA compatibility, and any additional imports in `src/original/`. Before claiming full reproduction, publish a tested lockfile and a hardware/software inventory in `ENVIRONMENT.md`.

For the reported CPU inference measurements, the experimental environment was **Intel Core Ultra 7 265KF**, batch size **1**, **one CPU thread**, **50 warm-up runs**, and **100 timed runs**. The consolidated results report **mean** CPU latency. They should not be relabeled as median latency.

## 5. Datasets and split integrity

Use the official distributors for CIFAR-10, CIFAR-100, and Speech Commands v0.02 and follow their individual licenses and terms. Raw datasets are not included by default. Any available split manifests and dataset verification records are located in `data/split_manifests/` and `protocol/`.

Before rerunning:

1. Verify dataset release/version, file checksums, labels and class counts against the recorded dataset verification evidence.
2. Preserve the original train/validation/test separation and the run's seed.
3. Ensure teacher training, student recovery, compression and calibration do not use held-out test information.
4. Record any substituted preprocessing, augmentations, split definitions or checkpoint sources; a modified protocol is a **new experiment**, not an exact reproduction.

The reported accuracy figures are **validation accuracy** unless explicitly labeled otherwise. This repository does not certify held-out test performance.

## 6. Reproduce Section 4.1: screening and controlled training

### 6.1 Five-epoch screening

Review `protocol/` and `results/section_4_1/` for the locked screening definitions and per-candidate evidence. The reported screening contains 24 FP32 combinations across three datasets, two architecture-width choices, two pruning levels, and two recovery objectives, with seed 42 and five epochs. The original scripts are collected under `src/original/`.

To verify the reported screening results without retraining, inspect the screening summary and individual contracts in `results/section_4_1/`. To retrain, first resolve the precise original script entrypoint, configuration paths, teacher checkpoint dependencies and data locations. **No generic execution command is asserted here because the complete dependency closure has not been verified.**

### 6.2 Controlled training

The controlled evaluation comprises 54 runs (three datasets × six configurations × three seeds) with 20 epochs. Retain seed-level records before computing means or paired contrasts. The manuscript compares cross-entropy (CE) and knowledge-distillation (KD) recovery, and joint versus sequential operation ordering under the evaluated configurations.

To audit a reported comparison, locate the individual seed records and paired contrast table in `results/section_4_1/`, match on dataset/seed/experimental context, then recompute differences. Do not treat repeated-seed observations as independent architectures or combine unmatched configurations.

## 7. Reproduce Section 4.2: precision and deployment

The measured deployment pool consists of **36 precision-specific artifacts**, representing **18 matched FP32/INT8 pairs**. Match records by candidate/checkpoint, dataset and seed, and vary precision within each pair. The reported aggregate validation accuracy, serialized artifact size and mean CPU latency are:

| Dataset | FP32 validation accuracy (%) | INT8 validation accuracy (%) | FP32 storage (MiB) | INT8 storage (MiB) | FP32 mean CPU latency (ms) | INT8 mean CPU latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 89.5467 | 89.4933 | 0.960 | 0.288 | 1.251476 | 0.581984 |
| CIFAR-100 | 77.4800 | 77.3733 | 32.483 | 8.267 | 10.477525 | 2.912774 |
| Speech Commands | 94.1956 | 93.8884 | 0.122 | 0.076 | 1.098501 | 0.705385 |

These are **aggregate descriptive values**, not per-run targets. At least one Speech Commands INT8 pair had worse latency than its FP32 counterpart; therefore, do not assert that every quantized model runs faster.

Storage refers to the serialized model artifact, **not** native peak inference memory. Reliable per-inference energy and native peak memory were not certified. Preserve the original timing statistic: the consolidated deployment evidence uses **mean latency**, even though an earlier protocol specified median latency.

## 8. Reproduce Section 4.3: feasibility and provisional Pareto analysis

The reported feasibility analysis used three measured objectives with dataset-specific thresholds:

| Dataset | Minimum validation accuracy | Maximum serialized storage | Maximum mean CPU latency |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 85% | 8 MiB | 30 ms |
| CIFAR-100 | 70% | 64 MiB | 100 ms |
| Speech Commands | 90% | 2 MiB | 20 ms |

For each dataset, retain measured candidates meeting all three thresholds. Candidate A dominates candidate B only if A is **no worse** on all three objectives (higher accuracy; lower storage and latency) and **strictly better** on at least one. Apply dominance **within each dataset**, not across datasets.

The reported **11 provisional nondominated candidates** were distributed as follows:

| Dataset | FP32 | INT8 | Total |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 1 | 3 | 4 |
| CIFAR-100 | 0 | 3 | 3 |
| Speech Commands | 2 | 2 | 4 |
| **Total** | **3** | **8** | **11** |

This is a candidate-pool-relative **three-objective** frontier, not a globally optimal or certified five-objective Pareto set. Do not add energy or peak-memory claims without independently verified measurements.

The historical search-policy comparisons include 18 matched contrasts. They are descriptive: differing upstream budgets and execution histories prevent causal attribution of search superiority. Cached contextual gating/replay is evidence of filtering consistency, **not** a demonstration of faster adaptive search or fewer training evaluations.

## 9. Manuscript-to-evidence audit

For each table, figure or numerical claim in Sections 4.1–4.3:

1. Identify its exact source CSV/JSON and the original experiment context in the repository.
2. Verify source file hashes using `provenance/file_manifest.csv`.
3. Recompute the statistic from individual records where available, preserving dataset, seed, candidate, precision and checkpoint matching.
4. Compare units carefully: bytes versus MiB, accuracy fraction versus percentage, and mean versus median latency.
5. Confirm that the figure is generated from the measured data. **Illustrative or AI-generated figures are not scientific measurement evidence.**
6. Record any discrepancy in a release issue or audit log before updating the manuscript.

The available figure files should be cross-checked against their numerical inputs. A figure's presence in `figures/` does not itself establish that it was derived from the deposited measurements.

## 10. Known limitations and exclusions

- Restricted datasets, architectures, pruning levels, recovery objectives and hardware environment.
- Three seeds for controlled training, limiting precision of small-effect estimates.
- Missing or uncertified energy and native peak inference-memory measurements.
- Mean deployment latency differs from the median statistic specified in an earlier protocol.
- No verified causal advantage over matched-budget adaptive search baselines.
- Original source scripts may require unpublished absolute paths, external checkpoints or additional packages.
- Third-party datasets and model weights may not be redistributable under this repository's CC BY 4.0 license.
- SharePoint availability and access permissions must be independently checked.

## 11. Citation, license and archived version

Use `CITATION.cff` for citation metadata. The archived v1.0.0 release has version DOI **https://doi.org/10.5281/zenodo.23602180**. Include the release tag and commit SHA in any reproduction report. Subsequent documentation changes do not retroactively modify this archive.

The repository's original research materials are designated CC BY 4.0 where the authors have rights to license them. Third-party code, data, and weights are governed by their own applicable licenses. For exact included files and any missing evidence, consult the release's provenance reports.

## 12. Reproduction reporting checklist

- [ ] Recorded archived release DOI, tag and commit SHA.
- [ ] Downloaded and SHA256-verified all required external files.
- [ ] Resolved all relevant omissions in `provenance/omissions.csv`.
- [ ] Recorded complete dependency versions and hardware details.
- [ ] Verified dataset versions and unchanged splits.
- [ ] Preserved seeds and matched run identifiers.
- [ ] Recomputed validation, storage and CPU latency metrics from individual records.
- [ ] Recomputed the dataset-specific three-objective nondominated sets.
- [ ] Compared each manuscript figure/table with the underlying evidence.
- [ ] Documented all deviations, failed reproductions and unresolved uncertainties.

---

**Release note:** This document describes the currently supported evidence and the steps needed for independent verification. It intentionally does not claim unverified executable entrypoints, energy results, full five-objective feasibility, or a successful fresh training reproduction.
