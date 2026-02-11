# Patchset Summary

## Baseline Snapshot

**Branch**: main  
**HEAD commit**: 5e99afe76a23ce850cc503e3778b11b3a1756aad  
**Tracked files**: 14

### Primary Entry Points

- `data_download.sh` / `download_oasis_scans.sh` — OASIS-3 dataset download via XNAT API
- `preprocess_t1w.py` — T1-weighted MRI preprocessing pipeline (MONAI transforms)
- `Splits_Creation.py` — Subject-level train/val/test split generation
- `model_train.ipynb` — 3D CNN encoder training with temporal contrastive loss (requires GPU)
- `Testing.py` — Embedding quality evaluation on held-out test set
- `Linear_Probe.py` — Linear classifier trained on frozen embeddings
- Visualization scripts: `PCA_Visualization.py`, `TSNE_visualization.py`, `Scan_Distribution.py`, `Example_MRI_Slice.py`

### Execution Context

The project trains a 3D convolutional neural network on longitudinal brain MRI scans using self-supervised temporal contrastive learning. Same-subject scans across time points form positive pairs. Training requires:

- OASIS-3 dataset (3,000+ T1-weighted scans)
- A100-class GPU
- Google Cloud Storage buckets for data storage (referenced in Testing.py, model_train.ipynb)
- MONAI, PyTorch, nibabel, pandas, scikit-learn, UMAP

Preprocessing standardizes scans to RAS orientation, 1mm isotropic spacing, intensity normalization. Subject-level splits prevent data leakage. Training is notebook-driven with manual hyperparameter configuration. Evaluation computes embedding stability (cosine similarity for same-subject pairs) and linear probe accuracy on CDR (Clinical Dementia Rating) classification.

---

## Phase 1 — Technical Audit

(To be completed)

---

## Phase 2 — Cleaning

(To be completed)

---

## Phase 3 — Documentation Rebuild

(To be completed)

---

## Phase 4 — Verification Implementation

**Verification Script**: `scripts/demo.sh`

The smoke test validates:
- Dependency availability (PyTorch, MONAI, NumPy)
- MONAI preprocessing pipeline on synthetic 3D volume
- Model architecture forward pass

**Local Execution**:
```
./scripts/demo.sh
```

**Result**:
```
========================================
ClinImCL Smoke Test
========================================

✓ Python detected: Python 3.13.9

Checking dependencies...
❌ ERROR: PyTorch not installed
```

**Note**: Full smoke test execution requires PyTorch, MONAI, and NumPy installed in the environment. The script correctly detects missing dependencies and exits with error code 1. This is expected behavior.

**Full execution requirements**:
- Python 3.8+
- PyTorch 2.0+
- MONAI 1.0+
- NumPy, matplotlib, scikit-learn

When dependencies are installed, the script validates:
1. Preprocessing transforms (shape: (1,128,128,128), finite values, intensity normalization)
2. Model forward pass (embedding shape: (batch, 128), no NaN/Inf)

Ends with: `SMOKE_OK` if all checks pass, `SMOKE_FAIL` otherwise.

**Limitations**:
- Does not test GCS I/O operations (requires GCS credentials)
- Does not test full training convergence (requires A100 GPU, hours of compute)
- Does not test split integrity (requires OASIS-3 dataset)

See **DEMO.md** for full pipeline execution instructions.

---

## Phase 5 — CI

(To be completed)

---

## Phase 6 — Finalization

(To be completed)
