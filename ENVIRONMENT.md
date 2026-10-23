# Experimental Environment and Reproducibility

**Project:** CAJO Edge AI — Constraint-Aware Configuration Search for Edge AI  
**Repository release:** v1.0.0 (planned; confirm before archiving)  
**Status:** Partially documented. Values not preserved in the available experimental evidence are explicitly marked **not verified**.

## 1. Scope

This document records the computing environment relevant to the experiments reported in Sections 4.1–4.3: architecture screening, controlled pruning and knowledge-distillation training, FP32/INT8 deployment measurements, and post hoc constraint-aware candidate analysis. It distinguishes **recorded experimental conditions** from **requirements that still need verification**. It is not a claim that the original execution environment has been fully reconstructed.

## 2. Recorded hardware and measurement protocol

| Item | Recorded setting | Verification status |
|---|---|---|
| CPU for reported deployment measurements | Intel Core Ultra 7 265KF | Reported in experiment summary; confirm against original benchmark logs |
| Inference device | CPU | Reported |
| Inference batch size | 1 | Reported |
| CPU inference threads | 1 | Reported |
| Warm-up iterations | 50 | Reported |
| Timed inference iterations | 100 | Reported |
| Reported inference latency | Mean latency in milliseconds | Reported; **not** the median specified in the original selection protocol |
| GPU model used for training | **Not verified** | Extract from original run metadata |
| System RAM / CPU core allocation | **Not verified** | Extract from execution machine |
| Operating system and build | **Not verified** | Extract from execution machine |
| Power/energy instrumentation | **Not certified** | Do not claim measured CPU energy |
| Native peak inference memory | **Not certified** | Serialized model size is not peak RAM usage |

**Measurement comparability:** Reported latency comparisons apply to the measured CPU, software runtime, exported model representation, and timing protocol. They must not be generalized to other processors or accelerators without new measurements. The one-thread setting refers to inference benchmarking, not necessarily training.

## 3. Software environment

The available evidence does not establish exact versions for the following dependencies. Do not substitute current installed versions as though they were the versions used in the original experiments.

| Component | Required information | Verified original version |
|---|---|---|
| Python | Interpreter version and architecture | **Not verified** |
| PyTorch | Version and build identifier | **Not verified** |
| torchvision | Version | **Not verified** |
| torchaudio | Version, if used by Speech Commands pipeline | **Not verified** |
| CUDA toolkit / PyTorch CUDA build | Version if GPU training was used | **Not verified** |
| NVIDIA driver / cuDNN | Version if applicable | **Not verified** |
| NumPy | Version | **Not verified** |
| pandas | Version | **Not verified** |
| Quantization backend and engine | Backend and runtime used for INT8 inference | **Not verified** |
| Threading libraries | OpenMP / MKL settings, if applicable | **Not verified** |

The repository's `requirements.txt` is a **dependency starting point**, not a validated lockfile. A new installation from it may produce different accuracy, storage, or latency values.

## 4. Experimental configurations and datasets

| Experiment | Recorded configuration |
|---|---|
| Section 4.1 screening | 24 FP32 configurations; 5 epochs; seed 42 |
| Section 4.1 controlled training | 54 runs; 20 epochs; seeds 42, 43, and 44 |
| Section 4.2 deployment | 36 precision-specific artifacts; 18 matched FP32/INT8 pairs |
| Section 4.3 selection | Provisional three-objective accuracy–storage–mean-latency analysis |
| Datasets | CIFAR-10, CIFAR-100, Speech Commands v0.02 |

Original dataset downloads, preprocessing, split manifests, teacher checkpoints, training hyperparameters, and candidate IDs must be obtained from the accompanying source code, protocol files, and evidence records. Dataset versions alone do not guarantee identical train/validation splits or preprocessing.

## 5. Capturing the environment on the original execution machine

Run the following commands **on the machine that actually produced the reported results**, or recover the equivalent historical outputs from logs. If run on a new machine, label the output *current reproduction environment*, not *original experimental environment*.

### Windows PowerShell

```powershell
python --version
python -m pip freeze > environment-pip-freeze.txt
python -m pip check > environment-pip-check.txt
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber,CsTotalPhysicalMemory | Format-List > environment-system.txt
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-List > environment-cpu.txt
nvidia-smi -q > environment-nvidia-smi.txt  # only if NVIDIA GPU and nvidia-smi available
```

Capture PyTorch's build and runtime information:

```powershell
python -c "import sys,platform,torch; print('Python:',sys.version); print('OS:',platform.platform()); print('Torch:',torch.__version__); print('Torch CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('Quantized engine:',torch.backends.quantized.engine); print('Torch threads:',torch.get_num_threads()); print('CUDA devices:',[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" > environment-pytorch.txt
```

**Note:** `nvidia-smi` and `Get-ComputerInfo` may be unavailable on some systems. Do not treat missing output as evidence that a GPU or Windows configuration was absent during the experiments.

### Optional reproducible environment specification

After the original versions have been confirmed, pin them in `requirements-lock.txt` or a platform-specific Conda environment file. Record the operating system, accelerator, quantization engine, and dataset provenance separately; Python package pinning alone does not reproduce hardware-dependent latency.

## 6. Measurement verification checklist

- [ ] Confirm CPU and benchmark runtime from the original execution logs.
- [ ] Recover exact Python, PyTorch, torchvision, torchaudio, and quantization backend versions.
- [ ] Confirm training GPU, driver, CUDA, and cuDNN versions where applicable.
- [ ] Verify original dataset checksums, split manifests, preprocessing, and seed handling.
- [ ] Match the 36 deployment records to the corresponding exported artifacts and hashes.
- [ ] Verify batch size, CPU thread count, warm-ups, repetitions, and latency statistic.
- [ ] Resolve the **mean-versus-median latency** protocol discrepancy before claiming full protocol compliance.
- [ ] Do not claim measured energy or native peak RAM without independent instrumented evidence.
- [ ] Record the final environment snapshot and its SHA256 checksum in `provenance/`.

## 7. Related repository documentation

- [`README.md`](README.md): repository overview and large-file archive link.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): experiment-specific reproduction guidance.
- [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md): dataset and externally hosted file access.
- `provenance/file_manifest.csv`: selected file paths and SHA256 checksums.
- `provenance/build_report.json`: collection status and omissions.

## 8. Reporting limitations

The published CPU inference numbers reflect **mean** latency, while the original protocol called for **median** latency. Serialized artifact size must not be described as native peak inference memory. The evidence does not establish a fully measured five-objective optimization outcome, nor does the retrospective policy comparison establish causal search-efficiency improvements. This document will be updated only when original execution evidence supports additional environment details.
