# ClinImCL

ClinImCL is a GPU-accelerated machine learning pipeline for longitudinal MRI representation learning using contrastive 3D CNN encoders.

The system processes OASIS-3 MRI data, constructs temporal embeddings across patient timepoints, and evaluates representation quality under high-dimensional volumetric constraints.

## System Overview

ClinImCL operates as a GPU-accelerated ML pipeline:

- Data processing — preprocesses longitudinal MRI scans into model-ready tensors
- Training — learns representations using a contrastive 3D CNN encoder
- Embedding generation — produces latent representations across timepoints
- Evaluation — analyzes embedding quality using downstream metrics and projections

The system is designed for scalable training on high-dimensional volumetric data while managing memory, batching, and temporal alignment constraints.

## Architecture

```text
MRI Data (OASIS-3)
        ↓
Preprocessing Pipeline
        ↓
3D CNN Encoder (PyTorch)
        ↓
Contrastive Learning Objective
        ↓
Embedding Space
        ↓
Evaluation / Visualization
```

## System Constraints

- High memory requirements for 3D MRI volumes during GPU training
- Limited batch sizes due to volumetric data dimensionality
- Temporal alignment challenges across longitudinal scans
- Preprocessing overhead for large-scale medical imaging datasets

## Key Properties

- GPU-accelerated 3D CNN training (PyTorch)
- Longitudinal representation learning across time-series MRI data
- Contrastive learning framework for embedding construction
- End-to-end pipeline from preprocessing to evaluation
- Designed for medical imaging workflows and structured datasets

## Why This Matters

Longitudinal medical imaging data presents challenges in capturing temporal structure and variability across scans. ClinImCL explores how contrastive learning can be applied to learn meaningful representations across time, enabling improved analysis of disease progression and patient trajectories in real-world clinical data pipelines.

## Repository Layout

```
ClinImCL/
├── download.sh
├── preprocess.py
├── model.ipynb
├── visualize.py
├── figures/
│   ├── epoch1_projections.png
│   ├── epoch20_projections.png
│   ├── epoch40_projections.png
│   ├── linearprobe_cm.png
│   ├── linearprobe_roc.png
│   ├── test_projections.png
│   ├── test_confusion.png
│   └── oasisbrains.png
├── Makefile
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- Install dependencies with `pip install -r requirements.txt`. For **GPU training**, install **PyTorch** for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install the rest of the requirements (the default `torch` from PyPI is often CPU-only).

## Standard Commands

```bash
make all
make clean
```

## Data and Training Notes

- The notebook workflow uses the following cloud references:
  - token `google_default`
  - project `clinimcl`
  - preprocessed path `gs://clinimcl-data/OASIS3/preprocessed/`
  - checkpoint path `gs://clinimcl-data/checkpoints/`
- `preprocess.py` handles MRI preprocessing.
- **Training** is implemented in `model.ipynb` (GPU, GCS-backed tensors and checkpoints). **`visualize.py`** loads checkpoints for embedding plots, stability checks, and linear-probe metrics—it does not run the training loop.
- **OASIS-3:** Imaging must be obtained and used under [OASIS](https://www.oasis-brains.org/) terms and any applicable agreements. This repository does **not** redistribute scans or patient data; it only documents pipeline code and static figure assets.

## Project Notes

- Figure assets live under `figures/`; keep filenames stable if external docs reference them.
