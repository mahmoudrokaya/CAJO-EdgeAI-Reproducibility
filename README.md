# CAJO Edge AI Reproducibility Repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23602180.svg)](https://doi.org/10.5281/zenodo.23602180)

**Archived release:** [v1.0.0](https://github.com/mahmoudrokaya/CAJO-EdgeAI-Reproducibility/releases/tag/v1.0.0)  
**Version DOI:** [10.5281/zenodo.23602180](https://doi.org/10.5281/zenodo.23602180)  
**License:** [CC BY 4.0](LICENSE) for original material covered by the repository license; third-party content retains its own terms.

This repository accompanies the research study *Constraint-Aware Configuration Search for Edge AI: A Multi-Dataset Study of Architecture Selection, Pruning, Knowledge Distillation, and Quantization*. It provides a curated collection of source code, configuration records, deployment measurements, and evidence supporting experiments on CIFAR-10, CIFAR-100, and Speech Commands v0.02.

## Citation

Rokaya, M. (2026). *CAJO Edge AI Reproducibility Repository* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23602180

For machine-readable citation metadata, see [`CITATION.cff`](CITATION.cff).

## Repository organization

| Path | Contents |
|---|---|
| `src/original/` | Selected original experimental scripts |
| `protocol/` | Experiment configuration and verification records |
| `data/split_manifests/` | Available dataset split metadata |
| `results/section_4_1/` | Architecture screening and controlled training |
| `results/section_4_2/` | FP32/INT8 deployment and matched comparisons |
| `results/section_4_3/` | Provisional three-objective Pareto and historical policy analysis |
| `figures/` | Available original manuscript figures |
| `provenance/` | Build report, source manifest, omissions, and external-file checksums |

Actual availability depends on the files included in the archived release and its provenance manifests; the directory table is not a completeness guarantee.

## Experimental scope

- **Section 4.1:** 24 five-epoch FP32 screening configurations; 54 controlled 20-epoch training runs.
- **Section 4.2:** 36 FP32/INT8 deployment artifacts comprising 18 matched checkpoint pairs.
- **Section 4.3:** 11 provisional nondominated candidates using validation accuracy, serialized storage, and mean CPU latency; 18 historical policy comparisons.

These are **validation and measured deployment results**, not held-out test performance or proof of global five-objective Pareto optimality. Native peak inference memory and per-inference CPU energy were not fully certified. Historical policy comparisons do not establish causal search-efficiency improvements.

## Large-file companion archive

Files strictly larger than **25 MiB (26,214,400 bytes)** were assigned to a separate, mirrored folder by the repository builder. The planned Taif University SharePoint/OneDrive distribution location is:

<https://taifedusa-my.sharepoint.com/:f:/r/personal/mahmoudrokaya_tu_edu_sa/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmahmoudrokaya%5Ftu%5Fedu%5Fsa%2FDocuments%2F48%2F481%2FToward%5FLightweight%5FEdge%5FIntelligence&share=cgrNdZT7rYK5Try9UdQPyicCEgUCWEoYAe03SdEM5rkjN713dg>

The external folder is named `CAJO-EdgeAI-Reproducibility-External`. Restore its contents into the GitHub repository root, retaining their relative paths. Consult `provenance/external_files.csv` for file sizes, paths, and SHA256 checksums where available. **The Zenodo DOI archives the GitHub release, not the separately hosted SharePoint files.** SharePoint upload and access permissions must be verified independently.

## Documentation

- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — experimental scope and reproduction procedures.
- [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) — original datasets, large-file access, integrity, and licensing.
- [`ENVIRONMENT.md`](ENVIRONMENT.md) — recorded hardware and software environment; unverified details identified.
- [`CHANGELOG.md`](CHANGELOG.md) — documentation and release history.

## Reuse and limitations

The [CC BY 4.0 license](LICENSE) covers only materials that the rights holder is authorized to license. Original benchmark datasets, third-party code, pretrained models, and external assets may have different terms. The curated snapshot is not represented as a one-command, independently verified reproduction pipeline.

**Release integrity:** Documentation commits after `v1.0.0` do not alter the frozen Zenodo archive for that version. If these updates need to be archived, publish a new release and use its new version DOI for that release.
