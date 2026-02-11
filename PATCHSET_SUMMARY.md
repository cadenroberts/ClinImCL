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

(To be completed)

---

## Phase 5 — CI

(To be completed)

---

## Phase 6 — Finalization

(To be completed)
