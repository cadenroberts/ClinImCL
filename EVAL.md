# Evaluation

## Metrics

| Metric | Definition | Script |
|--------|-----------|--------|
| Embedding stability | Cosine similarity between same-subject embeddings across longitudinal time points | `Testing.py` |
| Linear probe accuracy | Classification accuracy of a linear classifier trained on frozen embeddings (CDR prediction) | `Linear_Probe.py` |
| Cluster visualization | PCA and t-SNE projections of embedding space, colored by diagnostic group | `PCA_Visualization.py`, `TSNE_visualization.py` |
| Dataset coverage | Distribution of scans per subject, time points per subject, diagnostic groups | `Scan_Distribution.py` |

## Validation protocol

1. **Preprocessing validation**: Verify that all scans have consistent orientation, spacing, and intensity range after MONAI transforms.
2. **Split integrity**: Confirm no subject appears in more than one split (train/val/test). Subject-level splitting prevents data leakage.
3. **Training convergence**: Monitor contrastive loss over epochs; verify monotonic decrease on training set and stable validation loss.
4. **Embedding quality**: Compute mean cosine similarity for same-subject pairs vs. different-subject pairs. Stable embeddings show significantly higher within-subject similarity.
5. **Downstream task**: Linear probe on CDR classification measures whether embeddings capture clinically relevant structure without fine-tuning.

## Known limitations

- Dataset: OASIS-3 is a single-site longitudinal study; generalization to multi-site data is untested.
- Progress coordinate: CDR is a coarse 5-level ordinal scale; finer-grained progression metrics are not evaluated.
- Compute: Training on 3,000+ 3D volumes requires A100-class GPUs; lower-memory GPUs require reduced batch size or patch-based training.
- No automated hyperparameter search; training parameters are manually configured in the notebook.
