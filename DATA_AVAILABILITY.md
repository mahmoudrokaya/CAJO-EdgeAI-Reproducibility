# Data Availability

> **Archived software release:** [CAJO Edge AI v1.0.0](https://github.com/mahmoudrokaya/CAJO-EdgeAI-Reproducibility/releases/tag/v1.0.0) · **Version DOI:** [10.5281/zenodo.23602180](https://doi.org/10.5281/zenodo.23602180). Documentation changes made after the release are not part of the frozen v1.0.0 archive.

This document describes the datasets, experimental records, and large-file companion archive associated with **CAJO Edge AI Reproducibility Repository**, the research companion to *Constraint-Aware Configuration Search for Edge AI: A Multi-Dataset Study of Architecture Selection, Pruning, Knowledge Distillation, and Quantization*.

## 1. Availability overview

The repository is intended to provide the **selected code, experiment configurations, result tables, figure-supporting evidence, and provenance records** underlying the manuscript's reported experiments. It is not intended to redistribute every intermediate experiment, failed run, or historical artifact.

| Material | Intended location | Notes |
| --- | --- | --- |
| Original experimental scripts selected for the paper | `src/original/` | Scripts may depend on local paths or assets not included in the repository. |
| Configuration and dataset verification records | `protocol/` | Availability depends on the evidence collected by the repository builder. |
| Dataset split manifests, if present | `data/split_manifests/` | Manifests are not substitutes for the original datasets. |
| Architecture screening and controlled-training evidence | `results/section_4_1/` | Five-epoch screening and 20-epoch controlled training. |
| FP32/INT8 deployment measurements and export metadata | `results/section_4_2/` | Validation accuracy, serialized storage, and measured CPU latency. |
| Provisional Pareto and historical policy comparisons | `results/section_4_3/` | Relative to the evaluated candidates and measured objectives. |
| Available original manuscript figures | `figures/` | Figures must be checked against underlying numerical results. |
| File inventories, omissions, and integrity checks | `provenance/` | See manifests and build report described below. |
| Selected files larger than 25 MiB | External SharePoint/OneDrive companion folder | See Section 3. |

**Important:** This table describes the *planned repository layout*. The files actually included in a given release are determined by its manifests. A named directory or experiment is not evidence that every underlying artifact is available.

## 2. Original datasets

The experiments used three publicly known benchmark datasets:

- **CIFAR-10** — image classification benchmark. Official source: <https://www.cs.toronto.edu/~kriz/cifar.html>.
- **CIFAR-100** — image classification benchmark. Official source: <https://www.cs.toronto.edu/~kriz/cifar.html>.
- **Speech Commands v0.02** — speech classification benchmark. Official TensorFlow dataset information: <https://www.tensorflow.org/datasets/catalog/speech_commands>.

The source datasets are **not included by default** in the GitHub repository. Researchers should obtain them from their original distributors and follow the applicable terms of use and redistribution conditions. Dataset identifiers, preprocessing assumptions, and any available split records should be checked against `protocol/` and `data/split_manifests/` before attempting reproduction. The existence of a dataset download does not itself guarantee reproduction of the paper's exact training/validation partition.

## 3. External archive for large files

To keep the GitHub repository within the project's file-size policy, the repository-building script routes files **strictly larger than 25 MiB (26,214,400 bytes)** to a separate local companion directory:

```text
CAJO-EdgeAI-Reproducibility-External/
```

The companion directory preserves each file's **exact path relative to the GitHub repository root**. Only directories containing selected files are created. For example, a file stored externally at:

```text
CAJO-EdgeAI-Reproducibility-External/results/section_4_2/example.bin
```

should be restored to:

```text
CAJO-EdgeAI-Reproducibility/results/section_4_2/example.bin
```

**Planned SharePoint/OneDrive location for the external files:**

<https://taifedusa-my.sharepoint.com/:f:/r/personal/mahmoudrokaya_tu_edu_sa/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmahmoudrokaya%5Ftu%5Fedu%5Fsa%2FDocuments%2F48%2F481%2FToward%5FLightweight%5FEdge%5FIntelligence&share=cgrNdZT7rYK5Try9UdQPyicCEgUCWEoYAe03SdEM5rkjN713dg>

**Access status:** The URL is supplied as the intended hosting location. This document does **not** assert that all companion files have already been uploaded, that anonymous/public access is enabled, or that the link will remain permanently accessible. Users who cannot access it should contact the repository maintainers through the GitHub repository's issue tracker or other published contact channel.

### Restoring the external files

1. Download the external companion directory from the link above, once published.
2. Preserve its internal folder structure.
3. Copy the **contents** of `CAJO-EdgeAI-Reproducibility-External/` into the root of `CAJO-EdgeAI-Reproducibility/`.
4. Verify the restored files using `provenance/external_files.csv`, which records repository-relative paths, sizes, and SHA256 hashes for the external files that were successfully copied during the repository build.

Example verification using Python:

```python
import csv
import hashlib
from pathlib import Path

root = Path("CAJO-EdgeAI-Reproducibility")
manifest = root / "provenance" / "external_files.csv"

with manifest.open(newline="", encoding="utf-8") as handle:
    for record in csv.DictReader(handle):
        relative = Path(record["relative_path"])
        file_path = root / relative
        expected = record["sha256"].strip().lower()
        if not file_path.is_file():
            print("MISSING", relative)
            continue
        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        print("OK" if digest.hexdigest() == expected else "MISMATCH", relative)
```

A GitHub or Zenodo DOI assigned to the **GitHub release** does **not automatically archive or preserve** files hosted separately on SharePoint. The external files should be retained, access-tested, and ideally deposited in a separately citable preservation archive if long-term independent reproducibility is required.

## 4. Relationship to the reported experiments

The curated evidence is organized around the following manuscript sections:

| Manuscript section | Experimental scope | Interpretation limits |
| --- | --- | --- |
| Section 4.1 | 24 five-epoch FP32 screening configurations; 54 controlled 20-epoch training runs | Restricted search space and training budgets; controlled comparisons used three seeds. |
| Section 4.2 | 36 deployment artifacts forming 18 matched FP32/INT8 pairs | Measurements include validation accuracy, serialized storage, and mean CPU latency. |
| Section 4.3 | Provisional three-objective candidate selection; historical policy comparisons | The 11 nondominated candidates are relative to the measured pool, not globally Pareto-optimal. |

The study's reported **validation accuracy is not held-out test accuracy**. The consolidated deployment evidence uses **mean CPU latency**; it must not be silently relabeled as median latency. Native peak inference memory and per-inference energy were not fully certified in the consolidated evidence. Accordingly, the available results do not establish complete five-objective feasibility. Historical policy comparisons do not establish a causal advantage in search efficiency.

## 5. Provenance and integrity

The repository builder generates these files in `provenance/`:

- `file_manifest.csv` — selected and successfully copied/verified files, repository-relative paths, destinations, source references, sizes, and SHA256 checksums.
- `external_files.csv` — the subset of successfully copied/verified files routed to the external companion archive.
- `omissions.csv` — expected source files not found during collection or other documented omissions.
- `build_report.json` — build configuration, selected-file counts, missing-file counts, copying errors, and possible stale copies from earlier builds.

Before citing the archive as complete, inspect `build_report.json` for errors and `omissions.csv` for evidence gaps. The manifests document what was collected; they do not independently certify the correctness of the experimental implementation or scientific claims.

## 6. Licenses and reuse

Original repository research materials and documentation are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, **only to the extent that the repository maintainers own or are authorized to license those materials**. See the root `LICENSE` file.

Third-party datasets, pretrained models, external code, and other third-party assets remain subject to their own licenses and redistribution terms. Inclusion of a reference or manifest does not change those terms. Users must check the original source conditions before redistributing such materials.

## 7. Citation and versioning

Cite the archived v1.0.0 release using `CITATION.cff` and **https://doi.org/10.5281/zenodo.23602180**. Record the release tag and access date when using external companion files. If an external archive receives a separate persistent identifier, cite that identifier as well.

For experimental reproduction procedures, consult [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). For the scope and contents of the release, consult [`README.md`](README.md) and the `provenance/` manifests.

---

*This document should be reviewed and updated at each release, especially after the external archive is uploaded, its access permissions are verified, or a new DOI is assigned.*
