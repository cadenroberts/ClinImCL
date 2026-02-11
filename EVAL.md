# Evaluation

## Correctness Definition

ClinImCL's correctness is defined across four dimensions:

### 1. Preprocessing Correctness

**Invariants**:
- All preprocessed volumes have shape `(1, 128, 128, 128)`
- All voxel values are finite (no NaN or Inf)
- Orientation is RAS (Right-Anterior-Superior)
- Spacing is 1mm isotropic
- Intensity is normalized: mean ≈ 0, std ≈ 1 (for non-zero voxels)

**Verification**:
```bash
python -c "
import torch, glob
files = glob.glob('gs://clinimcl-data/OASIS3/preprocessed_new/*.pt')[:10]
for f in files:
    t = torch.load(f)
    assert t.shape == (1,128,128,128), f'Shape: {t.shape}'
    assert torch.isfinite(t).all(), f'Non-finite in {f}'
print('PASS: 10 samples validated')
"
```

### 2. Split Integrity

**Invariants**:
- No subject appears in multiple splits (train/val/test)
- Split ratios: train=68%, test=20%, val=12% (±1%)
- All files in splits exist in preprocessed bucket

**Verification**:
```bash
python -c "
import gcsfs, re
fs = gcsfs.GCSFileSystem(token='google_default')
splits = {'train': [], 'test': [], 'val': []}
for split in splits:
    paths = fs.glob(f'clinimcl-data/OASIS3/splits_new/{split}_new/part_*/*.pt')
    subjects = {re.search(r'(OAS\d+)', p).group(1) for p in paths}
    splits[split] = subjects
assert len(splits['train'] & splits['test']) == 0, 'Train-test overlap'
assert len(splits['train'] & splits['val']) == 0, 'Train-val overlap'
assert len(splits['test'] & splits['val']) == 0, 'Test-val overlap'
print('PASS: No subject overlap across splits')
"
```

### 3. Embedding Quality

**Metrics**:
- **Embedding stability**: Mean cosine similarity for same-subject scan pairs > 0.7
- **Separation**: Mean cosine similarity for different-subject pairs < 0.3
- **Dimensionality**: Embedding vectors have shape `(128,)`

**Verification**:
```bash
python Testing.py
# Expected output: embedding stability plot, t-SNE visualization
# Manual inspection: same-subject points cluster together in t-SNE
```

### 4. Linear Probe Performance

**Note**: Current implementation uses synthetic labels (`hash(subj) % 2`). Metrics are not clinically meaningful. Ground truth evaluation requires loading actual CDR labels from OASIS-3 metadata.

**Expected behavior** (with real labels):
- ROC-AUC > 0.6 (better than random for binary classification)
- Confusion matrix diagonal elements > off-diagonal (more correct than incorrect predictions)

**Verification**:
```bash
python Linear_Probe.py
# Outputs ROC curve, confusion matrix
# Check ROC-AUC value in stdout
```

---

## Evaluation Protocol

### Full Evaluation (Requires Trained Model)

1. **Preprocess validation set**:
   ```bash
   # Already completed by Preprocessing.py
   # Validation: check file count
   gsutil ls gs://clinimcl-data/OASIS3/preprocessed_new/ | wc -l
   # Expected: ≥4116 files
   ```

2. **Verify splits**:
   ```bash
   python -c "
   import gcsfs
   fs = gcsfs.GCSFileSystem(token='google_default')
   train = len(fs.ls('clinimcl-data/OASIS3/splits_new/train_new/part_01'))
   test = len(fs.ls('clinimcl-data/OASIS3/splits_new/test_new/part_01'))
   val = len(fs.ls('clinimcl-data/OASIS3/splits_new/validation_new/part_01'))
   print(f'Counts per part: train={train}, test={test}, val={val}')
   print('PASS' if train > 0 and test > 0 and val > 0 else 'FAIL')
   "
   ```

3. **Train model**:
   ```bash
   # Open model_train.ipynb in Google Colab
   # Configure hyperparameters (learning rate, batch size, epochs)
   # Execute all cells
   # Monitor training loss: should decrease monotonically
   ```

4. **Evaluate embeddings**:
   ```bash
   python Testing.py
   python PCA_Visualization.py
   python TSNE_visualization.py
   ```

5. **Linear probe**:
   ```bash
   python Linear_Probe.py
   # Note: Synthetic labels; metrics not clinically valid
   ```

---

## Pass/Fail Criteria

| Check | Command | Pass Condition |
|-------|---------|----------------|
| Preprocessing shape | `python preprocess_validate.py` | All volumes `(1,128,128,128)` |
| Preprocessing finite | `python preprocess_validate.py` | No NaN/Inf values |
| Split integrity | `python split_validate.py` | No subject overlap |
| Split ratios | `python split_validate.py` | Train=2800±50, test=840±20, val=476±10 |
| Embedding stability | `python Testing.py` | Mean same-subject cosine similarity > 0.7 |
| Embedding separation | `python Testing.py` | Mean different-subject cosine similarity < 0.3 |
| Linear probe (real labels) | `python Linear_Probe.py` | ROC-AUC > 0.6 |

**Note**: `preprocess_validate.py` and `split_validate.py` do not exist; inline validation commands provided above.

---

## Known Limitations

- Dataset: OASIS-3 is a single-site longitudinal study; generalization to multi-site data is untested.
- Progress coordinate: CDR is a coarse 5-level ordinal scale; finer-grained progression metrics are not evaluated.
- Compute: Training on 3,000+ 3D volumes requires A100-class GPUs; lower-memory GPUs require reduced batch size or patch-based training.
- No automated hyperparameter search; training parameters are manually configured in the notebook.
