# Repository Audit

## 1. Purpose

ClinImCL implements self-supervised representation learning for longitudinal brain MRI analysis. The system trains a 3D convolutional encoder on temporal scan pairs from the same subject using contrastive learning. The encoder produces stable embeddings without manual labels, exploiting the temporal consistency of same-brain scans across time points.

## 2. Entry Points

| File | Purpose | Execution Context |
|------|---------|-------------------|
| `data_download.sh` | Orchestrates OASIS-3 dataset bulk download from NITRC via XNAT API | Requires NITRC credentials, GCP VM provisioning, spawns 8 parallel tmux sessions |
| `download_oasis_scans.sh` | Third-party script for per-experiment download via authenticated curl | Called by `data_download.sh`; handles authentication, zip extraction, file rearrangement |
| `Preprocessing.py` | Loads raw NIfTI files from GCS, applies MONAI transforms, saves preprocessed tensors | Requires Google Cloud Storage access with `gcsfs`, writes to `gs://clinimcl-data/OASIS3/preprocessed_new/` |
| `preprocess_t1w.py` | Local preprocessing variant using filesystem paths instead of GCS | Expects `/data/OASIS3/**/**/*T1w.nii.gz` file structure, writes to `/data/OASIS3/preprocessed_t1w/` |
| `Splits_Creation.py` | Creates subject-level train/val/test splits, partitions into GCS folders | Requires GCS bucket access; shuffles 4116+ scans, distributes into 14-part splits |
| `model_train.ipynb` | Training notebook: 3D CNN encoder + temporal contrastive loss | Requires A100 GPU, Google Colab environment, reads from GCS splits |
| `Testing.py` | Evaluates embedding quality on test set: stability metrics, t-SNE/PCA/UMAP visualizations, ROC/confusion matrix | Reads from GCS test split, writes visualizations back to GCS |
| `Linear_Probe.py` | Trains logistic regression classifier on frozen embeddings | Loads embeddings from GCS, uses synthetic labels (`hash(subj) % 2`) for demonstration |
| `PCA_Visualization.py` | PCA projection of embeddings | Reads 50 embeddings from GCS, generates 2D scatter plot |
| `TSNE_visualization.py` | t-SNE projection of embeddings | (Not read in detail; likely similar to PCA script) |
| `Scan_Distribution.py` | Computes dataset statistics: scans per subject histogram | Reads preprocessed file list from GCS |
| `Example_MRI_Slice.py` | Visualizes MRI slice from volume | (Not read; likely for debugging/documentation) |

## 3. Dependency Surface

### Runtime Dependencies

- **Core**: `torch`, `numpy`, `pandas`, `nibabel` (NIfTI I/O)
- **Medical imaging**: `monai` (transforms, data loading)
- **Cloud storage**: `gcsfs` (Google Cloud Storage filesystem interface)
- **Visualization**: `matplotlib`, `seaborn`
- **Dimensionality reduction**: `scikit-learn` (PCA, logistic regression), `umap-learn`

### Development/Infrastructure Dependencies

- **Authentication**: NITRC IR credentials (environment variables: `NITRC_USER`, `NITRC_PASS`)
- **Compute**: Google Cloud Platform (Compute Engine VM, Cloud Storage buckets)
- **Notebook environment**: Google Colab with A100 GPU
- **Shell utilities**: `curl`, `zip`, `tmux`, `gcloud`

### Missing Dependency Specification

No `requirements.txt`, `pyproject.toml`, or `environment.yml` file exists. Dependency versions are not pinned. The notebook installs `monai` via `%pip install monai` without version constraint.

## 4. Configuration Surface

### Hardcoded Configuration

- **GCS bucket paths**: `clinimcl-data/OASIS3` appears in 6+ scripts (Preprocessing.py, Splits_Creation.py, Testing.py, Linear_Probe.py, PCA_Visualization.py, Scan_Distribution.py)
- **Filesystem paths**: `/data/OASIS3` in `preprocess_t1w.py`, `data_download.sh`
- **Split ratios**: 2800 train / 840 test / 476 validation (hardcoded in Splits_Creation.py)
- **Model architecture**: Conv3d layer dimensions, kernel sizes, pooling strategy in Testing.py and model_train.ipynb
- **Preprocessing parameters**: Intensity ranges (0-5000 → 0-1), spatial resolution (1mm isotropic), crop size (128³), orientation (RAS)
- **Random seed**: `np.random.seed(42)` in Splits_Creation.py

### Environment Variables

- `.env.example` exists but is empty (contains only a comment)
- `NITRC_USER`, `NITRC_PASS` used in `data_download.sh`

### Missing Configuration

- No centralized config file
- No command-line argument parsing in any script
- Hyperparameters (learning rate, batch size, epochs) embedded in notebook cells

## 5. Data Flow

```
NITRC XNAT API
    ↓ (authenticated curl, parallel downloads)
Raw NIfTI scans (.nii.gz)
    ↓ (Preprocessing.py or preprocess_t1w.py)
Preprocessed tensors (.pt, 128³ volumes, normalized intensity)
    ↓ (Splits_Creation.py)
Train/val/test splits (GCS folders: train_new, test_new, validation_new)
    ↓ (model_train.ipynb: 3D CNN + contrastive loss)
Trained model weights + embeddings per scan
    ↓ (Testing.py, Linear_Probe.py, PCA/t-SNE scripts)
Evaluation outputs: metrics, visualizations
```

**Critical dependencies**:
- Preprocessing depends on GCS bucket structure matching expected paths
- Splits_Creation.py asserts `>= 4116` scans; fails if dataset incomplete
- Model training reads from specific GCS split paths
- All evaluation scripts assume embeddings have been precomputed and stored in GCS with specific directory structure (`part_XX/subject_id/epoch_XXX/embedding.npy`)

## 6. Determinism Risks

| Risk | Location | Impact |
|------|----------|--------|
| **Non-reproducible random splits** | `Splits_Creation.py` uses `np.random.seed(42)` but shuffles file list from `fs.ls()` which may not guarantee consistent ordering across GCS API calls | Train/val/test subject assignment could vary |
| **Nondeterministic CUDA operations** | No `torch.backends.cudnn.deterministic = True` or `torch.use_deterministic_algorithms(True)` in training notebook | Training run outputs not reproducible |
| **Random initialization** | No manual seed set in `model_train.ipynb` | Model weights differ across runs |
| **GCS file listing order** | `fs.ls()` order not guaranteed stable | File iteration order varies |
| **Floating-point precision** | No explicit dtype specification in many operations | Minor numerical drift across hardware |
| **t-SNE randomness** | t-SNE visualization uses random initialization | Plots not reproducible |

## 7. Observability

### Logging

- Print statements throughout (e.g., `print(f"Found {len(t1w_files)} T1w scans")`)
- `tqdm` progress bars in preprocessing and split creation
- No structured logging (no `logging` module usage)
- No log levels (debug/info/warn/error)

### Error Handling

- `try/except` blocks in `Preprocessing.py` for per-scan failures; errors printed but processing continues
- No centralized error collection
- Scripts fail silently or with print statements; no exit codes except shell script defaults

### Metrics

- Embedding quality metrics computed in Testing.py: cosine similarity, ROC-AUC, confusion matrix
- Metrics not persisted to structured format (no CSV, JSON, MLflow, Weights & Biases)
- Linear probe accuracy computed but not logged to file

### Missing Observability

- No training metrics logging (loss curves not saved)
- No checkpointing metadata
- No data quality validation (e.g., verify all preprocessed scans are valid)
- No monitoring of GCS API quotas or download failures

## 8. Test State

**No automated tests exist.**

- No `test_*.py` files
- No `pytest`, `unittest`, or other testing framework
- No CI/CD configuration
- Validation is manual: run scripts, inspect outputs

**Implied validation**:
- Preprocessing includes `assert image.shape == (1,128,128,128)`
- Preprocessing includes `assert torch.isfinite(image).all()`
- Splits creation includes `assert total_scans >= 4116`

## 9. Reproducibility

### Pinned Dependencies

None. No lockfile (`requirements.txt`, `Pipfile.lock`, `poetry.lock`, `environment.yml` with pinned versions).

### Reproducibility Blockers

1. **Cloud storage dependency**: All scripts require Google Cloud Storage access with authentication. Cannot run locally without GCS buckets.
2. **OASIS-3 dataset access**: Requires NITRC IR account with approved OASIS access. Dataset is not public; download requires credentials.
3. **Compute requirements**: Training requires A100 GPU. Preprocessing of 4116 scans is computationally expensive.
4. **Hardcoded paths**: GCS bucket paths are not configurable; scripts will fail if run by another user without identical GCS setup.
5. **Missing model weights**: Trained model weights not included in repo; cannot reproduce evaluation without retraining.
6. **Manual notebook execution**: Training is notebook-based with manual cell execution; no automated training script.

## 10. Security Surface

| Component | Risk | Mitigation |
|-----------|------|------------|
| NITRC credentials | Environment variables `NITRC_USER`, `NITRC_PASS` in `data_download.sh` | Not committed; user must set manually |
| GCS authentication | `gcsfs` uses `token='google_default'` (ADC or service account) | Relies on GCP IAM; no keys in repo |
| VM provisioning | `gcloud compute` commands in `data_download.sh` create 3TB disk VM | No resource limits; cost exposure |
| Third-party script | `download_oasis_scans.sh` from external GitHub repo (NrgXnat/oasis-scripts) | Trusted source (OASIS project), but not vendored or version-pinned |
| Data exfiltration | All data stored in GCS; bucket access controls critical | Bucket permissions not documented |

## 11. Ranked Improvement List

### P0 (Blocks Basic Reproducibility)

1. **Add dependency specification**: Create `requirements.txt` with pinned versions for all Python dependencies.
2. **Document GCS setup**: Provide instructions for creating GCS buckets, setting IAM permissions, and authenticating with `gcsfs`.
3. **Add deterministic training script**: Convert notebook to `.py` script with manual seed setting, deterministic CUDA ops, CLI args.
4. **Document OASIS-3 access**: Explain how to request NITRC IR account, approve OASIS access, obtain credentials.

### P1 (Improves Reliability and Clarity)

5. **Centralize configuration**: Create `config.yaml` for all paths, hyperparameters, split ratios.
6. **Add smoke test**: Create minimal script that verifies preprocessing on a single scan without GCS dependency.
7. **Add data validation**: After preprocessing and splits creation, verify file counts, scan shapes, intensity ranges.
8. **Document compute requirements**: Specify memory, GPU type, disk space, estimated runtime for each script.
9. **Add checkpoint strategy**: Save model weights at regular intervals with metadata (epoch, loss, timestamp).
10. **Replace print statements with structured logging**: Use `logging` module with configurable log levels.

### P2 (Nice to Have)

11. **Containerization**: Create Dockerfile for reproducible environment.
12. **Add unit tests**: Test preprocessing transforms, split creation logic, model forward pass.
13. **Add linear probe ground truth**: Use actual CDR labels instead of `hash(subj) % 2` synthetic labels.
14. **Add embedding cache invalidation**: Track preprocessing/model version; regenerate embeddings when stale.
15. **Document failure modes**: Common errors (GCS auth failure, OOM, missing scans) with troubleshooting steps.
