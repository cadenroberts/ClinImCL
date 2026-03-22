# ClinImCL

ClinImCL is a GPU-accelerated machine learning pipeline for longitudinal MRI representation learning using contrastive learning.

The system processes OASIS-3 MRI data, constructs temporal embeddings via 3D CNN encoders, and evaluates representation quality across longitudinal patient scans.

## System Overview

ClinImCL operates as a structured ML pipeline:

- **Data processing** — preprocesses longitudinal MRI scans into model-ready tensors
- **Training** — learns representations using a 3D CNN contrastive learning framework
- **Embedding generation** — produces latent representations across timepoints
- **Evaluation** — analyzes embedding quality using downstream metrics and projections

The pipeline is designed for GPU-accelerated training and scalable processing of longitudinal medical imaging data.

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
- Dataset preprocessing cost for large-scale medical imaging data

## Key Properties

- GPU-accelerated 3D CNN training (PyTorch)
- Longitudinal representation learning across time-series MRI data
- Contrastive learning framework for embedding construction
- End-to-end pipeline from preprocessing to evaluation
- Designed for medical imaging workflows and structured datasets

## Why This Matters

Longitudinal medical imaging data presents challenges in capturing temporal structure and variability across scans. ClinImCL explores how contrastive learning can be applied to learn meaningful representations across time, enabling improved analysis of disease progression and patient trajectories.

## Repository Layout

```
ClinImCL/
├── download.sh
├── preprocess.py
├── model.ipynb
├── visualize.py
├── report/
│   ├── report.tex
│   ├── report.pdf
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
- TeX Live with `pdflatex`

## Standard Commands

Use `make` targets for repeatable runs:

```bash
make pdf
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
- `model.ipynb` and `visualize.py` are the main training and analysis paths.

## Project Notes

- Do not rename figure files without updating `report/report.tex`.
- Use `make clean` after compiling to remove LaTeX temporary files.
- Keep `report/report.pdf` aligned with `report/report.tex`.
