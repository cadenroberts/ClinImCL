# Design Decisions

This document records key architectural decisions made in ClinImCL, following the Architectural Decision Record (ADR) format.

---

## ADR-001: Temporal Contrastive Learning Over Supervised Classification

**Context**

Brain MRI datasets often lack dense clinical annotations. OASIS-3 provides longitudinal scans (multiple time points per subject) but sparse disease labels. Supervised learning requires extensive labeled data; obtaining labels is expensive and subjective (inter-rater variability in diagnosis).

**Decision**

Use temporal contrastive learning: treat scans from the same subject at different time points as positive pairs. The model learns to embed same-subject scans close together in latent space without requiring diagnostic labels.

**Consequences**

- **Benefit**: No manual annotation required; exploits natural temporal structure of longitudinal data.
- **Benefit**: Embeddings learn subject-specific stable features (anatomical structure) rather than confounding factors (scanner variability, acquisition parameters).
- **Limitation**: Requires longitudinal datasets; not applicable to cross-sectional studies.
- **Limitation**: Does not directly optimize for downstream clinical tasks (e.g., dementia classification); requires separate linear probe or fine-tuning.

---

## ADR-002: 3D Convolutional Networks Over 2D Slice-Based Models

**Context**

Brain MRI volumes are inherently 3D. Slice-based 2D CNNs process axial/coronal/sagittal slices independently, losing inter-slice spatial relationships. 3D CNNs process entire volumes but require significantly more memory.

**Decision**

Implement 3D CNN encoder (SimpleCNN) with Conv3d layers operating on full 128×128×128 volumes.

**Consequences**

- **Benefit**: Captures volumetric spatial relationships critical for structural brain analysis (e.g., cortical thickness gradients, subcortical structure geometry).
- **Benefit**: More accurate representation of 3D anatomy compared to 2D approaches.
- **Limitation**: Higher memory cost per sample; requires A100-class GPUs for training.
- **Limitation**: Smaller batch sizes achievable compared to 2D models; may slow convergence.

---

## ADR-003: Subject-Level Train/Val/Test Splits

**Context**

Naive random splitting of scans can place multiple scans from the same subject in different splits (e.g., subject OAS30001's baseline scan in train, follow-up scan in test). This causes data leakage: the model can learn subject-specific features in training and exploit them in test evaluation, inflating performance metrics artificially.

**Decision**

Perform subject-level splitting: all scans from a given subject appear in exactly one split (train, val, or test). Shuffle subjects (not individual scans) before assignment.

**Consequences**

- **Benefit**: Prevents data leakage; ensures test set truly evaluates generalization to unseen subjects.
- **Benefit**: Aligns with real-world deployment: model must generalize to new patients, not just new time points of known patients.
- **Limitation**: Reduces effective training set size (compared to scan-level splitting); fewer unique subjects than total scans.
- **Limitation**: Requires careful tracking of subject IDs across preprocessing and splitting stages.

---

## ADR-004: MONAI Preprocessing Pipeline

**Context**

Medical imaging preprocessing involves domain-specific transforms: orientation standardization (RAS convention), resampling to isotropic spacing, intensity normalization (CT Hounsfield units, MRI arbitrary units). Implementing these transforms from scratch is error-prone.

**Decision**

Use MONAI's preprocessing transforms: `Orientation`, `Spacing`, `ScaleIntensityRange`, `CropForeground`, `NormalizeIntensity`.

**Consequences**

- **Benefit**: Standardized, well-tested transforms designed for medical imaging.
- **Benefit**: GPU-accelerated transforms (compatible with PyTorch dataloaders).
- **Benefit**: Handles edge cases (4D volumes with time dimension, non-isotropic spacing, orientation permutations).
- **Limitation**: Adds dependency on MONAI library (versioning, compatibility with PyTorch versions).
- **Limitation**: MONAI's abstractions may obscure low-level preprocessing details; debugging requires understanding MONAI internals.

---

## ADR-005: Google Cloud Storage for Data Storage

**Context**

OASIS-3 dataset is ~4,000 T1-weighted scans, each ~50-200MB raw NIfTI, totaling ~500GB-1TB. Preprocessed tensors are smaller (~10-20MB each) but still require ~50GB storage. Training requires high-throughput reads from multiple workers. Google Colab provides free A100 GPU access but limited local storage.

**Decision**

Store all data (raw, preprocessed, splits, embeddings) in Google Cloud Storage (GCS) buckets. Access via `gcsfs` library for filesystem-like operations.

**Consequences**

- **Benefit**: Decouples storage from compute; dataset accessible from any Colab session or GCP VM.
- **Benefit**: Parallel access from multiple training jobs without data duplication.
- **Benefit**: Automatic replication and durability (GCS multi-region storage).
- **Limitation**: Requires GCP account with billing enabled (egress costs, storage costs).
- **Limitation**: Network latency for GCS reads may bottleneck dataloaders if not parallelized.
- **Limitation**: Code is not portable; cannot run locally without GCS setup or refactoring to filesystem I/O.
- **Limitation**: Debugging requires GCS authentication; harder to develop/test offline.

---

## ADR-006: Notebook-Driven Training Over Script-Based Training

**Context**

Deep learning experimentation involves iterative hyperparameter tuning, model architecture changes, and training monitoring. Scripts require rerunning from scratch for each change. Notebooks allow incremental execution and inline visualization.

**Decision**

Implement training in Jupyter notebook (`model_train.ipynb`) rather than standalone Python script.

**Consequences**

- **Benefit**: Interactive development: modify hyperparameters, re-run training loop without reloading data.
- **Benefit**: Inline visualizations (loss curves, embedding projections) without separate plotting scripts.
- **Benefit**: Compatible with Google Colab's notebook interface.
- **Limitation**: Not reproducible by default: no command-line interface, no automated logging of hyperparameters.
- **Limitation**: Version control diff is noisy (notebooks include execution metadata, outputs).
- **Limitation**: Harder to integrate into CI/CD pipelines compared to scripts.
- **Limitation**: No automated experiment tracking (MLflow, Weights & Biases) without manual instrumentation.

---

## ADR-007: Fixed Random Seed for Split Creation

**Context**

Dataset splits must be reproducible: rerunning `Splits_Creation.py` should produce identical train/val/test assignments. NumPy's `shuffle()` depends on random state.

**Decision**

Set fixed random seed (`np.random.seed(42)`) before shuffling scan list in `Splits_Creation.py`.

**Consequences**

- **Benefit**: Reproducible splits: same seed → same subject assignments across runs.
- **Benefit**: Enables fair comparison of models trained at different times or by different users.
- **Limitation**: GCS file listing order (`fs.ls()`) is not guaranteed stable by GCS API; if file order varies, shuffle output differs despite fixed seed.
- **Limitation**: Seed value (42) is arbitrary; no sensitivity analysis to seed choice.

---

## ADR-008: Synthetic Labels for Linear Probe Demonstration

**Context**

Linear probe evaluation measures whether frozen embeddings contain clinically relevant information by training a linear classifier on diagnostic labels (e.g., CDR for dementia severity). OASIS-3 provides CDR labels, but they are not loaded in the current implementation.

**Decision**

Use synthetic labels (`hash(subj) % 2`) in `Linear_Probe.py` to demonstrate the linear probing workflow without implementing CDR label loading.

**Consequences**

- **Benefit**: Demonstrates linear probe code structure (train logistic regression, compute ROC-AUC, plot confusion matrix).
- **Benefit**: Does not require parsing OASIS-3 metadata CSV files (subject demographics, CDR scores).
- **Limitation**: Metrics (ROC-AUC, accuracy) are meaningless; synthetic labels have no clinical relevance.
- **Limitation**: Cannot evaluate whether embeddings capture dementia-related features without real labels.
- **Limitation**: Misleading to users who may interpret results as clinically valid.

**Note**: This is a placeholder for demonstration purposes. A production system must load actual CDR labels from OASIS-3 metadata.

---

## ADR-009: GCS Bucket Paths Hardcoded in Scripts

**Context**

All scripts read/write from specific GCS bucket paths (e.g., `gs://clinimcl-data/OASIS3/preprocessed_new/`). Configuration options: hardcode paths, environment variables, CLI arguments, config files.

**Decision**

Hardcode GCS bucket paths directly in scripts.

**Consequences**

- **Benefit**: Simple; no configuration file parsing or CLI argument parsing.
- **Benefit**: No risk of mismatched configuration across scripts.
- **Limitation**: Not portable; other users must edit source code to change bucket paths.
- **Limitation**: Difficult to test with different datasets or storage backends.
- **Limitation**: Code duplication: bucket path strings repeated across 6+ scripts.

**Improvement Path**: Refactor to centralized config file (`config.yaml`) with single source of truth for all paths.

---

## ADR-010: SimpleCNN Architecture for Embedding Extraction

**Context**

Model architecture design involves trade-offs between capacity (depth, width) and efficiency (memory, compute). State-of-the-art 3D medical image models (3D ResNets, Med3D, DenseNet3D) are large (millions of parameters). For contrastive learning, a lightweight encoder may suffice if the dataset is not extremely large.

**Decision**

Implement SimpleCNN: 2-layer 3D CNN with minimal channels (1→8→16), adaptive pooling, and small embedding dimension (128).

**Consequences**

- **Benefit**: Low memory footprint; fits in A100 GPU memory with reasonable batch size.
- **Benefit**: Fast training: fewer parameters → faster forward/backward passes.
- **Benefit**: Sufficient capacity for OASIS-3 dataset (~4,000 scans, relatively homogeneous: single-site, T1w only).
- **Limitation**: May underfit if scaled to larger, more diverse datasets (multi-site, multi-modal).
- **Limitation**: No skip connections, batch norm, or dropout; limited regularization.

**Note**: Architecture choice is empirical; no hyperparameter search performed. A production system should compare SimpleCNN against ResNet3D-18, DenseNet3D-121, or other baselines.
