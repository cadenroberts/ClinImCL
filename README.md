# ClinImCL

Clinical Imaging Contrastive Learning
====================================

This repository contains code and scripts for a contrastive learning project on
longitudinal clinical MRI data (OASIS-3). It was developed as part of the
CSE 290D Neural Computation course and is structured around a data pipeline
that downloads, preprocesses, trains, and analyses embeddings stored in
Google Cloud Storage.



**Repository Structure**

```
/Users/cwr/ClinImCL
├── data_download.sh               # orchestrates bulk download via NITRC
├── download_oasis_scans.sh        # helper script (from oasis-scripts repo)
├── preprocess_t1w.py              # MONAI preprocessing of raw NIfTI files
├── model_train.ipynb              # interactive Colab notebook for training
├── Linear_Probe.py                # script to train/evaluate a linear probe
├── PCA_Visualization.py           # simple PCA plot of embeddings
├── README.md                      # this file
└── (others may be added as project evolves)
```

Workflow Overview
----------------


1. **Data acquisition**
   - `data_download.sh` uses NITRC credentials to pull OASIS‑3 MRI experiments.
   - Delegates per‑session downloads to `download_oasis_scans.sh` (external
     script, included for completeness).
   - After download, the data are synced to `gs://clinimcl-data/OASIS3/raw/`.

2. **Preprocessing**
   - `preprocess_t1w.py` traverses the raw directory, identifies T1‑weighted
     scans, and applies MONAI transforms (orientation, spacing, intensity
     scaling, foreground cropping, normalization).
   - Output tensors are saved as `.pt` files in a `preprocessed_t1w` folder and
     a CSV index is generated for longitudinal subjects.

3. **Training**
   - `model_train.ipynb` contains a full training pipeline.
     - A “baseline” ResNet‑50 encoder training block (first cells).
     - A “fast” variant later in the notebook that uses a lightweight
       encoder, streaming loader, rotating subject subsets, and automatic
       checkpoint upload to `gs://clinimcl-data/checkpoints/`.
   - Training is done on 3‑D volumes, contrastively pairing time‑points within
     the same subject against random negatives.
   - Hyper‑parameters (batch size, image size, epochs, etc.) are clearly
     demarcated by comments and environment variables.

4. **Representation analysis**
   - `Linear_Probe.py` downloads saved embeddings from the bucket,
     fits a logistic regression classifier, computes ROC/CM plots, and
     uploads the figures back to the bucket.
   - `PCA_Visualization.py` fetches a subset of embeddings and produces a 2‑D
     PCA scatter plot to inspect embedding structure.



**Dependencies**

The Python components rely on the following packages (installable via pip):

- `torch` (CUDA-enabled version recommended)
- `monai`
- `gcsfs`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`, `tqdm` (used by preprocessing script)

Bash scripts require `curl`, `gsutil`, and `gcloud` for GCP operations.

Install using:

```
pip install torch monai gcsfs numpy scikit-learn matplotlib seaborn pandas tqdm
```



**Usage Notes**

1. Credentials
   - Set `NITRC_USER`/`NITRC_PASS` before running download scripts.
   - Colab notebooks authenticate with `google.colab.auth` and use `gcsfs` for
     bucket access.

2. Bucket paths are hardcoded; adjust `bucket_path`/`GCS_BUCKET_PREFIX` if you
   migrate data or fork the project.

3. Training variations
   - The notebook is divided into sections; execute cells sequentially in
     Colab (GPU runtime recommended). The fast variant shown in earlier logs is
     contained in a later cell.

4. Extending the pipeline
   - Add classification heads or downstream tasks by adapting
     `Linear_Probe.py` or adding notebook cells after training.
   - Further visualizations (t-SNE, UMAP) follow the pattern in
     `PCA_Visualization.py`.


Cohesion Across Files
----------------------


Every script forms a stage in the end-to-end pipeline:

- Downloaded raw data → preprocessed by `preprocess_t1w.py` → used by the
  notebook loader functions.
- Embeddings produced by `model_train.ipynb` are consumed by the two analysis
  scripts (`Linear_Probe.py`, `PCA_Visualization.py`) which respectively train a
  linear classifier and visualize the latent space.
- Cloud storage (`gs://clinimcl-data`) serves as the central hub for data and
  checkpoints, ensuring the components remain loosely coupled and portable.

This modular structure allows replacing any stage (e.g. swapping the encoder
architecture or using a local disk instead of GCS) without touching the other
parts, which is why the workspace exhibits strong cohesion and clear
separation of concerns.



Going Forward
--------------


- Expand the training notebook with logging (TensorBoard or Weights & Biases).
- Add additional downstream tasks (clinical score regression, segmentation).
- Package preprocessing into a CLI tool or workflow manager (Snakemake, etc.).
- Document environment setup (Dockerfile/requirements.txt) for reproducibility.

