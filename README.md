# ClinImCL

Distributed self-supervised representation learning pipeline for longitudinal MRI analysis.

## Purpose

ClinImCL trains a 3D convolutional encoder on longitudinal brain MRI scans using temporal contrastive learning. Same-subject scans across time points form positive pairs; the model learns stable embeddings without manual annotations. The pipeline processes 3,000+ T1-weighted scans from the OASIS-3 dataset on GPU clusters using MONAI preprocessing and PyTorch training infrastructure.

## Architecture

```
OASIS-3 dataset
    │
    ├─ data_download.sh / download_oasis_scans.sh
    │  (bulk download via XNAT API)
    │
    ▼
preprocess_t1w.py
    │  Intensity normalization, reorientation,
    │  cropping, resizing via MONAI transforms
    │
    ▼
Splits_Creation.py
    │  Subject-level train/val/test splits
    │  (no subject leaks across splits)
    │
    ▼
model_train.ipynb
    │  3D CNN encoder + temporal contrastive loss
    │  GPU-accelerated training (A100)
    │
    ▼
Testing.py
    │  Evaluate on held-out test set
    │
    ├─ Linear_Probe.py         → linear classifier on frozen embeddings
    ├─ PCA_Visualization.py    → PCA embedding projections
    ├─ TSNE_visualization.py   → t-SNE embedding projections
    └─ Scan_Distribution.py    → dataset statistics
```

## Key design decisions

| Decision | Tradeoff | Rationale |
|----------|----------|-----------|
| Temporal contrastive loss (same-subject pairs) | Requires longitudinal data; not applicable to cross-sectional datasets | Exploits temporal consistency for label-free learning; same brain across time provides natural positive pairs |
| 3D CNN over slice-based 2D models | Higher memory cost per sample | Captures volumetric spatial relationships critical for structural brain analysis |
| Subject-level split strategy | Reduces effective training set size | Prevents data leakage: a subject's scans appear in exactly one split |
| MONAI preprocessing | Adds dependency | Standardized medical imaging transforms with GPU acceleration; handles orientation, spacing, and intensity normalization consistently |

## Evaluation

| Metric | Definition |
|--------|-----------|
| Embedding stability | Cosine similarity between embeddings of same-subject scans across time points |
| Representation quality | Linear probe accuracy on CDR (Clinical Dementia Rating) classification |
| Cluster separation | Visual assessment via PCA, t-SNE, and UMAP projections colored by diagnosis |
| Downstream transfer | Accuracy of frozen-embedding classifiers on disease progression tasks |

## Repo structure

```
ClinImCL/
├── data_download.sh           Bulk OASIS-3 download
├── download_oasis_scans.sh    Per-scan download helper
├── preprocess_t1w.py          T1-weighted MRI preprocessing (MONAI)
├── Preprocessing.py           General preprocessing utilities
├── Splits_Creation.py         Subject-level train/val/test splits
├── model_train.ipynb          Training notebook (3D CNN + contrastive loss)
├── Testing.py                 Test set evaluation
├── Linear_Probe.py            Linear probing on frozen embeddings
├── PCA_Visualization.py       PCA embedding visualization
├── TSNE_visualization.py      t-SNE embedding visualization
├── Scan_Distribution.py       Dataset distribution analysis
├── Example_MRI_Slice.py       MRI slice visualization
└── clinimcl.pdf               Technical report
```

## Usage

```bash
# 1. Download OASIS-3 dataset
./data_download.sh

# 2. Preprocess T1-weighted scans
python preprocess_t1w.py

# 3. Create subject-level splits
python Splits_Creation.py

# 4. Train (open model_train.ipynb, configure hyperparameters, execute cells)

# 5. Evaluate
python Testing.py
python Linear_Probe.py
python PCA_Visualization.py
python TSNE_visualization.py
```

## License

MIT License (see LICENSE file).
