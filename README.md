# ClinImCL

Contrastive representation learning for longitudinal MRI.

## Repository Layout

```
ClinImCL/
├── download.sh
├── preprocess.py
├── model.ipynb
├── visualize.py
├── report.tex
├── report.pdf
├── epoch1_projections.png
├── epoch20_projections.png
├── epoch40_projections.png
├── linearprobe_cm.png
├── linearprobe_roc.png
├── test_projections.png
├── test_confusion.png
├── oasisbrains.png
├── Makefile
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- TeX Live with `pdflatex`

## Standard Commands

Use `make` targets for repeatable runs:

```bash
make pdf
make all
make clean
```

## Data and Training Notes

- The notebook workflow uses the following cloud references:
  - token `google_default`
  - project `clinimcl`
  - preprocessed path `gs://clinimcl-data/OASIS3/preprocessed/`
  - checkpoint path `gs://clinimcl-data/checkpoints/`
- `preprocess.py` handles MRI preprocessing.
- `model.ipynb` and `visualize.py` are the main training and analysis paths.

## Project Notes

- Do not rename figure files without updating `report.tex`.
- Use `make clean` after compiling to remove LaTeX temporary files.
- Keep `report.pdf` aligned with `report.tex`.
