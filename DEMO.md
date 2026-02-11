# Demo

This document provides instructions for running ClinImCL end-to-end. Due to external dependencies (NITRC credentials, GCS buckets, A100 GPU), a **smoke test** is provided that validates core preprocessing logic locally.

---

## Prerequisites

### Required

- Python 3.8+
- PyTorch 2.0+
- MONAI 1.0+
- nibabel, numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm

### Optional (for full pipeline)

- NITRC IR account with OASIS-3 access approval
- Google Cloud Platform account with billing enabled
- Google Cloud Storage bucket (`gs://clinimcl-data/`)
- Google Colab account (for A100 GPU access)
- gcloud CLI configured with authentication

---

## Smoke Test (Local Execution)

The smoke test validates preprocessing transforms on synthetic data without requiring OASIS-3 dataset or GCS access.

### Create Smoke Test Script

Save as `smoke_test.py`:

```python
import torch
import numpy as np
from monai.transforms import (
    EnsureChannelFirst, Orientation, Spacing,
    ScaleIntensityRange, CropForeground, Resize, NormalizeIntensity, Compose
)

# Define preprocessing pipeline (same as Preprocessing.py)
preprocess = Compose([
    EnsureChannelFirst(channel_dim='no_channel'),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRange(a_min=0, a_max=5000, b_min=0, b_max=1, clip=True),
    CropForeground(),
    Resize((128, 128, 128)),
    NormalizeIntensity(nonzero=True)
])

# Create synthetic 3D brain-like volume
np.random.seed(42)
volume = np.random.randn(160, 200, 180) * 1000 + 2500  # Simulate MRI intensity range
volume[volume < 0] = 0  # No negative intensities
volume[10:150, 20:180, 15:165] += 500  # Add "brain" region

# Preprocess
try:
    processed = preprocess(volume)
    assert processed.shape == (1, 128, 128, 128), f"Shape: {processed.shape}"
    assert torch.isfinite(processed).all(), "Non-finite values detected"
    assert processed.dtype == torch.float32, f"Dtype: {processed.dtype}"
    print("✅ SMOKE_OK: Preprocessing pipeline validated")
except Exception as e:
    print(f"❌ SMOKE_FAIL: {e}")
    exit(1)
```

### Run Smoke Test

```bash
python smoke_test.py
```

**Expected output**:
```
✅ SMOKE_OK: Preprocessing pipeline validated
```

---

## Full Pipeline Execution

The full pipeline requires external infrastructure. Steps below document the complete workflow.

### 1. Data Download (Requires NITRC Credentials)

```bash
# Set credentials (replace with your NITRC account)
export NITRC_USER="your_username"
export NITRC_PASS="your_password"

# Run download script (provisions GCP VM, launches parallel downloads)
./data_download.sh

# Expected duration: 6-12 hours for ~4,000 scans
# Expected output: gs://clinimcl-data/OASIS3/raw/ populated with ~500GB of NIfTI files
```

**Troubleshooting**:
- `Authentication failed`: Verify NITRC credentials are correct
- `Permission denied (GCS)`: Verify GCP project has Storage Admin IAM role
- `VM creation failed`: Verify GCP quota for c2-standard-8 machines

### 2. Preprocessing

```bash
# GCS-based preprocessing (recommended)
python Preprocessing.py

# OR: Filesystem-based preprocessing (if data stored locally)
python preprocess_t1w.py

# Expected duration: 4-8 hours (depends on VM CPU count)
# Expected output: gs://clinimcl-data/OASIS3/preprocessed_new/ with ~4,116 .pt files
```

**Validation**:
```bash
# Check file count
gsutil ls gs://clinimcl-data/OASIS3/preprocessed_new/*.pt | wc -l
# Expected: ≥4116

# Spot-check tensor shape
python -c "
import torch, gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
with fs.open('clinimcl-data/OASIS3/preprocessed_new/OAS30001_anat1.pt', 'rb') as f:
    t = torch.load(f)
    print(f'Shape: {t.shape}, Dtype: {t.dtype}')
"
# Expected: Shape: torch.Size([1, 128, 128, 128]), Dtype: torch.float32
```

### 3. Split Creation

```bash
python Splits_Creation.py

# Expected duration: ~1 hour (GCS copy operations)
# Expected output: gs://clinimcl-data/OASIS3/splits_new/{train,test,validation}_new/part_{01..14}/
```

**Validation**:
```bash
# Check split counts
gsutil ls gs://clinimcl-data/OASIS3/splits_new/train_new/part_01/*.pt | wc -l
# Expected: ~200 files per train part

gsutil ls gs://clinimcl-data/OASIS3/splits_new/test_new/part_01/*.pt | wc -l
# Expected: ~60 files per test part
```

### 4. Model Training (Requires A100 GPU)

```bash
# Open model_train.ipynb in Google Colab
# Runtime → Change runtime type → GPU → A100

# Execute cells sequentially:
# 1. Install dependencies (%pip install monai)
# 2. Authenticate to GCS (if not using default credentials)
# 3. Define model architecture
# 4. Define contrastive loss
# 5. Training loop (monitor loss decrease)
# 6. Save embeddings to GCS per epoch

# Expected duration: 8-24 hours (depends on epoch count, batch size)
# Expected output: gs://clinimcl-data/OASIS3/train_new_outputs/part_XX/{subject}/epoch_XXX/embedding.npy
```

**Troubleshooting**:
- `CUDA out of memory`: Reduce batch size from 16 → 8 → 4
- `GCS quota exceeded`: Enable requester pays or increase quota in GCP console
- `Training loss not decreasing`: Check learning rate (try 1e-4, 1e-3, 1e-2)

### 5. Evaluation

```bash
# Embedding quality metrics
python Testing.py

# Linear probe (note: synthetic labels)
python Linear_Probe.py

# Visualizations
python PCA_Visualization.py
python TSNE_visualization.py

# Dataset statistics
python Scan_Distribution.py

# Expected outputs:
# - ROC curves, confusion matrices in GCS bucket
# - t-SNE plots saved locally or to GCS
# - Embedding stability metrics printed to stdout
```

---

## Expected Outputs Summary

| Stage | Output | Location |
|-------|--------|----------|
| Download | Raw NIfTI files (~500GB) | `gs://clinimcl-data/OASIS3/raw/` |
| Preprocessing | Preprocessed tensors (~50GB) | `gs://clinimcl-data/OASIS3/preprocessed_new/` |
| Splits | Train/val/test folders | `gs://clinimcl-data/OASIS3/splits_new/` |
| Training | Model weights, embeddings | `gs://clinimcl-data/OASIS3/train_new_outputs/` |
| Evaluation | Metrics, visualizations | Local files + GCS paths |

---

## Limitations

### Why Full Demo is Not Feasible Locally

1. **OASIS-3 dataset access**: Requires NITRC IR account with approved data use agreement. Dataset is not public.
2. **GCS dependency**: All scripts assume GCS bucket paths. Refactoring to local filesystem requires modifying 6+ Python files.
3. **GPU requirement**: Training requires A100 GPU (40GB VRAM). Consumer GPUs (e.g., RTX 3090 24GB) may OOM with current batch size.
4. **Compute time**: Full pipeline (download + preprocess + train + eval) takes 20-40 hours wall-clock time.

### Smoke Test Scope

The smoke test validates:
- MONAI preprocessing pipeline (transforms, shape assertions)
- Finite value checks
- Tensor dtype correctness

The smoke test does **not** validate:
- GCS I/O operations
- Model training convergence
- Embedding quality metrics
- Split integrity

For full validation, use the evaluation commands in **EVAL.md**.
