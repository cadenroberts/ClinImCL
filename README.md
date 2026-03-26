# ClinImCL

Contrastive learning pipeline for longitudinal 3D MRI representation learning on OASIS-3 data. Trains a 3D CNN encoder with InfoNCE loss over temporal scan pairs, producing embeddings evaluated via UMAP trajectories, PCA projections, and linear probe classification.

## Files

| File | Purpose |
|---|---|
| `model.py` | Shared 3D CNN encoder + projection head (`ClinImCL`, `Encoder`, `Block`, `IMG`) |
| `model.ipynb` | Training loop (Colab/GPU, GCS-backed data and checkpoints) |
| `preprocess.py` | MONAI pipeline: NIfTI to 96^3 `.pt` tensors |
| `visualize.py` | Embedding evaluation: PCA, UMAP, cosine stability, linear probe |
| `download.sh` | OASIS-3 MRI downloader (NITRC auth, parallel via tmux) |
| `Makefile` | `make all` validates figures; `make clean` removes caches, cookies, CSVs, and `visualizations/` |
| `requirements.txt` | Python dependencies (PyTorch, MONAI, sklearn, umap-learn, gcsfs) |
| `figures/` | 8 committed result PNGs (reference outputs from training runs) |

## Entry Points

| Command | Description |
|---|---|
| `model.ipynb` (Colab) | Train encoder on GCS-hosted preprocessed volumes |
| `python preprocess.py --data_dir /data/OASIS3 --out_dir /data/OASIS3/preprocessed` | Preprocess raw NIfTI scans |
| `python visualize.py --mode gcs --gcs_bucket <path> --epoch 20` | Evaluate embeddings from GCS |
| `python visualize.py --mode local --ckpt <path> --data_dir <path>` | Evaluate embeddings from local .pt files |
| `bash download.sh` | Download OASIS-3 scans (requires `NITRC_USER`, `NITRC_PASS`) |

## Verification

```bash
make all          # confirms all 8 expected figures exist
make clean        # removes caches, cookies, generated CSVs, visualizations/
python -c "from model import ClinImCL, IMG; import torch; m=ClinImCL(); z,h=m(torch.randn(1,1,IMG,IMG,IMG)); assert z.shape==(1,128) and h.shape==(1,256)"
```

## Architecture

```mermaid
graph TD
    A[OASIS-3 NIfTI scans] -->|download.sh| B[Raw MRI volumes]
    B -->|preprocess.py| C[96^3 .pt tensors]
    C -->|GCS upload| D[gs://clinimcl-data/OASIS3/preprocessed/]
    D -->|model.ipynb| E[ClinImCL Encoder]
    E -->|InfoNCE contrastive loss| F[Trained checkpoint .pth]
    F -->|visualize.py| G[PCA / UMAP / Linear Probe]
    G --> H[figures/]

    subgraph model.py
        E1[Block: Conv3d + BN + ReLU] --> E2[Encoder: 4x Block + MaxPool3d + AdaptiveAvgPool]
        E2 --> E3[Projection head: Linear + ReLU + Linear + L2 norm]
    end
```

## Data and Training

- Cloud references: project `clinimcl`, data `gs://clinimcl-data/OASIS3/preprocessed/`, checkpoints `gs://clinimcl-data/checkpoints/`
- OASIS-3 imaging must be obtained under [OASIS](https://www.oasis-brains.org/) terms. This repo does not redistribute scans or patient data.

## Requirements

Python 3.11+. For GPU training, install PyTorch for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) first, then:

```bash
pip install -r requirements.txt
```
