# ClinImCL

Self-supervised representation learning for longitudinal brain MRI analysis.

## Summary

ClinImCL trains a 3D convolutional encoder on brain MRI scans using temporal contrastive learning. Scans from the same subject at different time points form positive pairs; the model learns stable subject-specific embeddings without manual labels. The system processes 4,000+ T1-weighted volumes from OASIS-3 using MONAI preprocessing and PyTorch training on A100 GPUs.

## What It Does

- Downloads OASIS-3 longitudinal brain MRI dataset (3,000+ subjects, 4,000+ scans) via NITRC API
- Preprocesses T1-weighted NIfTI volumes: orientation standardization, isotropic resampling, intensity normalization, cropping to 128³
- Creates subject-level train/val/test splits (no data leakage)
- Trains 3D CNN encoder with temporal contrastive loss: same-subject scans → high cosine similarity, different-subject scans → low similarity
- Evaluates embedding quality: stability metrics, linear probe accuracy, PCA/t-SNE/UMAP visualizations

## Architecture

### Staged Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Data Acquisition                                       │
├─────────────────────────────────────────────────────────────────┤
│  data_download.sh → NITRC XNAT API → Raw NIfTI volumes         │
│  - Authenticated curl requests                                  │
│  - Parallel downloads (8 tmux sessions)                         │
│  - GCP VM provisioning (c2-standard-8, 3TB disk)                │
│  - Output: gs://clinimcl-data/OASIS3/raw/ (~500GB)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Preprocessing                                          │
├─────────────────────────────────────────────────────────────────┤
│  Preprocessing.py → MONAI transforms → Preprocessed tensors    │
│  Pipeline:                                                      │
│    1. Orientation → RAS                                         │
│    2. Spacing → 1mm isotropic (bilinear interpolation)          │
│    3. Intensity → clip [0,5000], normalize [0,1]                │
│    4. CropForeground → remove padding                           │
│    5. Resize → 128×128×128                                      │
│    6. NormalizeIntensity → mean=0, std=1 (non-zero voxels)      │
│  - Output: gs://...preprocessed_new/ (~50GB, 4116 .pt files)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Split Creation                                         │
├─────────────────────────────────────────────────────────────────┤
│  Splits_Creation.py → Subject-level splits                      │
│  - Shuffle subjects with fixed seed (np.random.seed(42))        │
│  - Split: 2800 train / 840 test / 476 val                      │
│  - Partition into 14 GCS folders per split (parallelism)        │
│  - Output: gs://...splits_new/{train,test,validation}_new/     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Training                                               │
├─────────────────────────────────────────────────────────────────┤
│  model_train.ipynb → 3D CNN encoder + contrastive loss         │
│  Model: SimpleCNN                                               │
│    Conv3d(1→8, k=3) → ReLU → MaxPool3d(2)                       │
│    Conv3d(8→16, k=3) → ReLU → AdaptiveAvgPool3d(1)              │
│    Flatten → Linear(16→128) → embedding                         │
│  Loss: Temporal contrastive (same-subject pairs positive)       │
│  Hardware: Google Colab A100 (40GB VRAM)                        │
│  - Output: gs://...train_new_outputs/ (embeddings per epoch)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: Evaluation                                             │
├─────────────────────────────────────────────────────────────────┤
│  Testing.py → Embedding quality metrics                         │
│  - Cosine similarity: same-subject vs different-subject pairs   │
│  - PCA, t-SNE, UMAP projections                                 │
│  - ROC curves, confusion matrices                               │
│                                                                 │
│  Linear_Probe.py → Logistic regression on frozen embeddings     │
│  - Note: Uses synthetic labels (hash(subj) % 2) for demo        │
│  - Production: Load actual CDR labels from OASIS metadata       │
└─────────────────────────────────────────────────────────────────┘
```

### Component Contracts

| Producer | Consumer | Data Contract |
|----------|----------|---------------|
| `data_download.sh` | `Preprocessing.py` | NIfTI files in GCS: `{experiment_id}/{scan_type}/*.nii.gz` |
| `Preprocessing.py` | `Splits_Creation.py` | Tensors: `{subject}_{scan_type}.pt`, shape `(1,128,128,128)` |
| `Splits_Creation.py` | `model_train.ipynb` | Train/val splits in GCS folders `part_{01..14}/` |
| `model_train.ipynb` | Evaluation | Embeddings: `part_XX/{subject}/epoch_XXX/embedding.npy`, shape `(128,)` |

## Design Tradeoffs

| Decision | Rationale | Consequence |
|----------|-----------|-------------|
| Temporal contrastive learning | Exploits longitudinal structure; no manual labels required | Requires multi-timepoint data; not applicable to cross-sectional datasets |
| 3D CNN | Captures volumetric spatial relationships | High memory cost; requires A100-class GPU |
| Subject-level splits | Prevents data leakage | Fewer training subjects (vs scan-level splitting) |
| MONAI preprocessing | Standardized medical imaging transforms | Adds dependency; abstracts low-level operations |
| GCS storage | Decouples storage from compute; accessible from Colab | Requires GCP billing; code not portable to local filesystem |
| Notebook training | Interactive development; inline visualizations | Not reproducible by default; harder to integrate into CI/CD |

See **DESIGN_DECISIONS.md** for detailed ADRs.

## Evaluation

### Correctness Criteria

1. **Preprocessing**: All volumes `(1,128,128,128)`, finite values, RAS orientation
2. **Splits**: No subject overlap between train/val/test
3. **Embeddings**: Same-subject cosine similarity > 0.7, different-subject < 0.3
4. **Linear probe** (with real labels): ROC-AUC > 0.6

### Evaluation Commands

```bash
# Smoke test (validates preprocessing without GCS)
python smoke_test.py
# Expected: ✅ SMOKE_OK

# Full evaluation (requires trained model)
python Testing.py           # Embedding metrics
python Linear_Probe.py      # Classification metrics (synthetic labels)
python PCA_Visualization.py # 2D projection
```

See **EVAL.md** for detailed verification protocol.

## Demo

### Smoke Test (No External Dependencies)

```bash
# Create synthetic volume, validate preprocessing pipeline
python smoke_test.py
# Expected output: ✅ SMOKE_OK: Preprocessing pipeline validated
```

### Full Pipeline (Requires NITRC + GCS + A100)

```bash
# 1. Download OASIS-3 (6-12 hours)
export NITRC_USER="your_username"
export NITRC_PASS="your_password"
./data_download.sh

# 2. Preprocess (4-8 hours)
python Preprocessing.py

# 3. Create splits (~1 hour)
python Splits_Creation.py

# 4. Train (8-24 hours, Google Colab A100)
# Open model_train.ipynb, execute cells

# 5. Evaluate
python Testing.py
python Linear_Probe.py
python PCA_Visualization.py
```

See **DEMO.md** for troubleshooting and expected outputs.

## Repository Layout

```
ClinImCL/
├── ARCHITECTURE.md           System design, component contracts, failure modes
├── DESIGN_DECISIONS.md       ADRs (10 decision records)
├── EVAL.md                   Correctness definition, verification commands
├── DEMO.md                   Execution instructions, smoke test
├── REPO_AUDIT.md             Technical audit (dependencies, config, risks)
├── PATCHSET_SUMMARY.md       Commit history and verification results
├── README.md                 This file
│
├── data_download.sh          OASIS-3 download orchestration (GCP VM, parallel)
├── download_oasis_scans.sh   Per-scan download helper (NITRC curl)
├── Preprocessing.py          GCS-based preprocessing (MONAI)
├── preprocess_t1w.py         Filesystem-based preprocessing (MONAI)
├── Splits_Creation.py        Subject-level train/val/test splits
├── model_train.ipynb         3D CNN training (Google Colab)
├── Testing.py                Embedding quality evaluation
├── Linear_Probe.py           Logistic regression on frozen embeddings
├── PCA_Visualization.py      PCA projection of embeddings
├── TSNE_visualization.py     t-SNE projection
├── Scan_Distribution.py      Dataset statistics (scans per subject)
├── Example_MRI_Slice.py      MRI slice visualization
│
├── clinimcl.pdf              Technical report
├── .env.example              Environment variable template
├── .gitignore                Git exclusions
└── sync.sh                   Commit helper script
```

## Limitations

### Reproducibility Blockers

1. **GCS dependency**: All scripts require Google Cloud Storage access. Cannot run locally without refactoring.
2. **OASIS-3 access**: Requires NITRC IR account with approved data use agreement. Dataset is not public.
3. **GPU requirement**: Training requires A100 GPU (40GB VRAM). Consumer GPUs may OOM.
4. **No dependency pinning**: No `requirements.txt` with versions. Install latest MONAI/PyTorch → potential compatibility issues.
5. **Nondeterministic training**: No manual seed set in training notebook → results not reproducible.
6. **Synthetic labels**: `Linear_Probe.py` uses `hash(subj) % 2` instead of real CDR labels → metrics not clinically valid.

### Known Issues

- **GCS file listing order**: `fs.ls()` order not guaranteed stable → split assignments may vary despite fixed seed
- **No model checkpointing**: Training notebook does not save model weights; only embeddings saved
- **No automated testing**: No unit tests, integration tests, or CI pipeline
- **Hardcoded paths**: GCS bucket paths repeated across 6+ scripts; not configurable

See **REPO_AUDIT.md** for prioritized improvement list.

## License

MIT License (see LICENSE file).
