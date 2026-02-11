# Architecture

## Overview

ClinImCL implements self-supervised representation learning for longitudinal brain MRI analysis. The system consists of four main stages: data acquisition, preprocessing, model training, and evaluation.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Acquisition Layer                       │
├─────────────────────────────────────────────────────────────────────┤
│  data_download.sh                                                    │
│    ├─ Authenticates to NITRC IR via XNAT API                        │
│    ├─ Fetches OASIS-3 experiment metadata (CSV)                     │
│    ├─ Provisions GCP VM (c2-standard-8, 3TB disk)                   │
│    └─ Spawns 8 parallel tmux sessions                               │
│         └─> download_oasis_scans.sh (per-session)                   │
│               ├─ Authenticated curl requests                        │
│               ├─ Downloads scans as ZIP archives                    │
│               ├─ Extracts and rearranges files                      │
│               └─> Outputs: /data/OASIS3/{experiment_id}/{scan}/     │
│                                                                      │
│  Output: ~4,000 raw T1-weighted NIfTI volumes in GCS bucket         │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       Preprocessing Layer                            │
├─────────────────────────────────────────────────────────────────────┤
│  Preprocessing.py (GCS-based) OR preprocess_t1w.py (filesystem)     │
│                                                                      │
│  Input: Raw NIfTI (.nii.gz) files                                   │
│  Transforms (MONAI pipeline):                                       │
│    1. EnsureChannelFirst → add channel dimension                    │
│    2. Orientation → standardize to RAS orientation                  │
│    3. Spacing → resample to 1mm isotropic voxels (bilinear)         │
│    4. ScaleIntensityRange → clip [0, 5000] → normalize [0, 1]       │
│    5. CropForeground → remove background padding                    │
│    6. Resize → 128×128×128 volume                                   │
│    7. NormalizeIntensity → mean=0, std=1 (non-zero voxels)          │
│                                                                      │
│  Validation:                                                         │
│    - Assert output shape: (1, 128, 128, 128)                        │
│    - Assert all values finite (no NaN/Inf)                          │
│                                                                      │
│  Output: PyTorch tensors (.pt) stored in GCS bucket                 │
│          gs://clinimcl-data/OASIS3/preprocessed_new/                │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Split Creation Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│  Splits_Creation.py                                                  │
│                                                                      │
│  Input: All preprocessed tensors from GCS bucket                    │
│  Process:                                                            │
│    1. List all .pt files from GCS (expected: ≥4116 scans)           │
│    2. Shuffle with fixed seed (np.random.seed(42))                  │
│    3. Split:                                                         │
│         - Train: 2800 scans (68%)                                   │
│         - Test: 840 scans (20%)                                     │
│         - Val: 476 scans (12%)                                      │
│    4. Partition each split into 14 GCS folders (for parallelism)   │
│                                                                      │
│  Output: Three GCS directories:                                     │
│    - gs://clinimcl-data/OASIS3/splits_new/train_new/part_{01..14}/  │
│    - gs://clinimcl-data/OASIS3/splits_new/test_new/part_{01..14}/   │
│    - gs://clinimcl-data/OASIS3/splits_new/validation_new/part_{..}/ │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          Training Layer                              │
├─────────────────────────────────────────────────────────────────────┤
│  model_train.ipynb                                                   │
│                                                                      │
│  Model Architecture (SimpleCNN):                                    │
│    Input: (batch, 1, 128, 128, 128) tensor                          │
│    │                                                                 │
│    ├─ Conv3d(1→8, kernel=3) → ReLU → MaxPool3d(2)                   │
│    ├─ Conv3d(8→16, kernel=3) → ReLU → AdaptiveAvgPool3d(1)          │
│    └─ Flatten → Linear(16→128) → embedding vector                   │
│                                                                      │
│  Training Loop:                                                      │
│    - Loss: Temporal contrastive loss (same-subject pairs positive)  │
│    - Optimizer: Adam (learning rate, batch size in notebook)        │
│    - Hardware: Google Colab A100 GPU                                │
│    - Data loading: Reads from GCS train split via gcsfs             │
│                                                                      │
│  Output: Model weights (not saved in repo)                          │
│          Embeddings per scan stored in GCS:                         │
│          gs://.../train_new_outputs/part_XX/subject_id/epoch_XXX/   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        Evaluation Layer                              │
├─────────────────────────────────────────────────────────────────────┤
│  Testing.py                                                          │
│    ├─ Loads test embeddings from GCS                                │
│    ├─ Computes embedding stability (cosine similarity)              │
│    ├─ Generates PCA, t-SNE, UMAP visualizations                     │
│    └─ Outputs confusion matrix, ROC curve to GCS                    │
│                                                                      │
│  Linear_Probe.py                                                     │
│    ├─ Loads frozen embeddings from train set                        │
│    ├─ Trains logistic regression classifier                         │
│    ├─ Evaluates ROC-AUC, confusion matrix                           │
│    └─ Note: Uses synthetic labels (hash(subj) % 2)                  │
│                                                                      │
│  PCA_Visualization.py, TSNE_visualization.py                        │
│    └─ 2D projections of embedding space for visual inspection       │
│                                                                      │
│  Scan_Distribution.py                                                │
│    └─ Histogram of scans per subject                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Execution Flow

### 1. Data Acquisition

**Entry**: `data_download.sh`

- Authenticates to NITRC IR using credentials (`NITRC_USER`, `NITRC_PASS`)
- Fetches OASIS-3 experiment list (CSV format from XNAT API)
- Filters for MRI sessions only (excludes PET, other modalities)
- Provisions GCP VM with 3TB disk
- Splits experiment list into 8 chunks for parallel processing
- Launches 8 tmux sessions, each running `download_oasis_scans.sh`
- Each session downloads scans via authenticated curl, extracts ZIP archives
- Final upload to GCS bucket: `gs://clinimcl-data/OASIS3/raw/`

**Dependencies**: NITRC IR account with OASIS-3 access approval, GCP project with billing enabled

**Failure modes**:
- Authentication failure: Invalid credentials → curl returns 401
- Network interruption: Partial downloads → script supports resume via `continueDownload()`
- Storage exhaustion: 3TB disk fills → script fails with no-space-left error
- Rate limiting: NITRC API throttles requests → downloads stall

### 2. Preprocessing

**Entry**: `Preprocessing.py` (GCS-based) or `preprocess_t1w.py` (local filesystem)

- Reads raw NIfTI files from GCS or filesystem
- Applies MONAI transform pipeline (7 transforms)
- Validates output tensor shape and finite values
- Saves preprocessed tensor to GCS bucket

**Dependencies**: MONAI, PyTorch, nibabel, gcsfs (for GCS variant)

**Failure modes**:
- 4D NIfTI files: Script extracts first volume automatically
- Non-3D data: Raises `ValueError` and skips scan
- Non-finite values: Assertion failure → scan skipped
- GCS write failure: Network error → scan skipped, error printed

### 3. Split Creation

**Entry**: `Splits_Creation.py`

- Lists all preprocessed tensors from GCS
- Asserts count ≥4116 (fails if dataset incomplete)
- Shuffles with fixed seed (reproducibility)
- Splits into train/test/val
- Copies files to split-specific GCS folders

**Dependencies**: gcsfs, NumPy

**Failure modes**:
- Insufficient scans: Assertion failure → script exits
- GCS API order nondeterminism: `fs.ls()` order may vary → split assignment changes (despite fixed seed)
- GCS write failure: Partial upload → incomplete splits

### 4. Training

**Entry**: `model_train.ipynb`

- Loads data from GCS train split
- Initializes 3D CNN encoder
- Trains with temporal contrastive loss
- Saves embeddings to GCS per epoch

**Dependencies**: PyTorch, MONAI, gcsfs, Google Colab A100 GPU

**Failure modes**:
- OOM: Batch size too large → reduce batch size or use gradient accumulation
- GCS quota exhaustion: Too many reads → script stalls
- Non-deterministic training: No manual seed set → results not reproducible

### 5. Evaluation

**Entry**: `Testing.py`, `Linear_Probe.py`, visualization scripts

- Loads embeddings from GCS
- Computes metrics and visualizations
- Writes outputs back to GCS

**Dependencies**: scikit-learn, matplotlib, seaborn, UMAP

**Failure modes**:
- Missing embeddings: If training incomplete, scripts fail reading expected paths
- Synthetic labels: `Linear_Probe.py` uses `hash(subj) % 2` instead of real CDR labels → metrics are not clinically meaningful

## Contracts Between Components

| Producer | Consumer | Contract |
|----------|----------|----------|
| `data_download.sh` | `Preprocessing.py` | Raw NIfTI files in GCS bucket `raw/`, organized as `{experiment_id}/{scan_type}/*.nii.gz` |
| `Preprocessing.py` | `Splits_Creation.py` | Tensors in GCS bucket `preprocessed_new/`, filenames: `{subject_id}_{scan_type}.pt`, shape: `(1,128,128,128)` |
| `Splits_Creation.py` | `model_train.ipynb` | Train/val splits in GCS folders `train_new/part_{01..14}/`, `validation_new/part_{01..14}/` |
| `model_train.ipynb` | Evaluation scripts | Embeddings in GCS folders `train_new_outputs/part_XX/{subject_id}/epoch_XXX/embedding.npy`, shape: `(128,)` |
| `Splits_Creation.py` | `Testing.py` | Test split in GCS folders `test_new/part_{01..14}/` |

## Observability

### Logging

- Print statements with progress indicators (`tqdm` for long-running loops)
- No structured logging (no `logging` module)
- No centralized log aggregation

### Monitoring

- No instrumentation for GCS API calls (request count, latency, errors)
- No training metrics tracking (no MLflow, Weights & Biases integration)
- No alerting on failures

### Debugging

- Preprocessing errors caught per-scan; script continues with remaining scans
- Error messages printed to stdout; no log files
- No stack traces preserved for batch processing

## Failure Modes Summary

| Layer | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| Data download | Network interruption | Download incomplete (small file size) | Resume via `continueDownload()` |
| Data download | Auth failure | curl 401 error | User must provide valid credentials |
| Preprocessing | Non-3D data | `ValueError` raised | Scan skipped; processing continues |
| Preprocessing | Non-finite values | Assertion failure | Scan skipped; processing continues |
| Split creation | Insufficient scans | Assertion failure | Script exits; user must complete download |
| Training | OOM | CUDA OOM exception | Reduce batch size or use smaller model |
| Evaluation | Missing embeddings | FileNotFoundError | User must complete training first |
