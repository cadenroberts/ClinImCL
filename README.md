# ClinImCL: Self-Supervised Contrastive Learning for Longitudinal Clinical Imaging Analysis

ClinImCL is a self-supervised learning framework designed to analyze longitudinal brain MRI scans. It leverages temporal consistency in longitudinal data to learn stable embeddings without requiring manual annotations. This repository contains the code and scripts necessary to preprocess data, train the model, and evaluate the results.

## Table of Contents
1. [Overview](#overview)
2. [Folder Structure](#folder-structure)
3. [Installation](#installation)
4. [Reproducing Results](#reproducing-results)
5. [File Descriptions](#file-descriptions)
6. [Citations](#citations)

---

## Overview

ClinImCL uses a lightweight 3D convolutional neural network (CNN) to process longitudinal MRI scans. The framework is designed to:
- Preprocess MRI scans to ensure consistency across datasets.
- Train a self-supervised model using temporal contrastive learning.
- Evaluate embeddings using dimensionality reduction techniques (PCA, UMAP, t-SNE) and linear probing.

The framework was tested on the OASIS-3 dataset, which contains longitudinal MRI scans of subjects with varying stages of cognitive decline.

---

## Folder Structure

The repository contains the following key files:

```
ClinImCL/
├── data_download.sh          # Script to download the OASIS-3 dataset
├── download_oasis_scans.sh   # Helper script for downloading specific scans
├── Example_MRI_Slice.py      # Script to visualize example MRI slices
├── Linear_Probe.py           # Script to evaluate embeddings using linear probing
├── model_train.ipynb         # Jupyter notebook for training the ClinImCL model
├── PCA_Visualization.py      # Script for PCA-based embedding visualization
├── preprocess_t1w.py         # Preprocessing pipeline for T1-weighted MRI scans
├── Preprocessing.py          # General preprocessing utilities
├── Scan_Distribution.py      # Script to analyze scan distribution in the dataset
├── Splits_Creation.py        # Script to create training/validation/test splits
├── Testing.py                # Script to evaluate the model on test data
├── TSNE_visualization.py     # Script for t-SNE-based embedding visualization
└── README.md                 # Project documentation
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cadenroberts/ClinImCL.git
   cd ClinImCL
   ```

2. (Optional) Install MONAI for medical imaging preprocessing:
   ```bash
   pip install monai
   ```

---

## Reproducing Results

To reproduce the results from the paper, follow these steps:

1. **Download the OASIS-3 Dataset**:
   - Run `data_download.sh` to download the dataset.
   - Use `download_oasis_scans.sh` to fetch specific scans.

2. **Preprocess the Data**:
   - Use `preprocess_t1w.py` to preprocess the MRI scans. This includes intensity normalization, reorientation, cropping, and resizing.

3. **Train the Model**:
   - Open `model_train.ipynb` and execute the cells to train the ClinImCL model. Training parameters (e.g., batch size, learning rate) can be adjusted in the notebook.

4. **Evaluate the Model**:
   - Use `Testing.py` to evaluate the trained model on the test set.
   - Visualize embeddings using `PCA_Visualization.py` and `TSNE_visualization.py`.

5. **Analyze Results**:
   - Use `Linear_Probe.py` to assess the quality of embeddings for downstream tasks.
   - Analyze scan distribution with `Scan_Distribution.py`.

---

## File Descriptions

### Data Handling
- **`data_download.sh`**: Automates the download of the OASIS-3 dataset.
- **`download_oasis_scans.sh`**: Helper script to download specific MRI scans.

### Preprocessing
- **`preprocess_t1w.py`**: Preprocesses T1-weighted MRI scans (normalization, cropping, resizing).
- **`Preprocessing.py`**: Contains utility functions for preprocessing.

### Training
- **`model_train.ipynb`**: Jupyter notebook for training the ClinImCL model.

### Evaluation
- **`Testing.py`**: Evaluates the trained model on a held-out test set.
- **`Linear_Probe.py`**: Evaluates embeddings using a linear classifier.
- **`PCA_Visualization.py`**: Visualizes embeddings using PCA.
- **`TSNE_visualization.py`**: Visualizes embeddings using t-SNE.

### Analysis
- **`Scan_Distribution.py`**: Analyzes the distribution of scans in the dataset.
- **`Example_MRI_Slice.py`**: Visualizes example MRI slices.

### Utilities
- **`Splits_Creation.py`**: Creates training, validation, and test splits.

---

## Citations

If you use this code, please cite us.
