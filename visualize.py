#!/usr/bin/env python3
"""
Visualization and evaluation for ClinImCL:
(1) PCA 2D scatter  (2) UMAP trajectories  (3) Embedding stability  (4) Linear probe
"""

import os, glob, re, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics.pairwise import cosine_distances
import gcsfs
import umap

# ── Model (must match model.ipynb for checkpoint loading) ──────────

class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(ci, co, 3, padding=1),
                                 nn.BatchNorm3d(co), nn.ReLU(True))
    def forward(self, x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self, base=32, out=256):
        super().__init__()
        ch = [1, base, base*2, base*4, base*8]
        self.blocks = nn.ModuleList(Block(ch[i], ch[i+1]) for i in range(4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                  nn.Linear(ch[-1], out))
    def forward(self, x):
        for b in self.blocks:
            x = b(F.max_pool3d(x, 2))
        return self.head(x)

class ClinImCL(nn.Module):
    def __init__(self, proj=128):
        super().__init__()
        self.enc = Encoder()
        self.proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True),
                                  nn.Linear(256, proj))
    def forward(self, x):
        h = self.enc(x)
        return F.normalize(self.proj(h), dim=1), h

# ── Data loading ───────────────────────────────────────────────────

def load_embeddings_from_gcs(bucket_path, epoch, max_subjects=None):
    fs = gcsfs.GCSFileSystem(token='google_default')
    parts = sorted([p for p in fs.ls(bucket_path) if "part_" in p])

    X, subjects = [], []
    print(f"Loading embeddings from GCS: {bucket_path}")

    for part_idx, part in enumerate(parts, 1):
        subj_list = sorted(fs.ls(part))
        print(f"Part {part_idx}/{len(parts)}: {os.path.basename(part)}")

        for subj_idx, subj in enumerate(subj_list, 1):
            embed_path = f"{subj}/epoch_{epoch:03d}/embedding.npy"
            if fs.exists(embed_path):
                with fs.open(embed_path, 'rb') as f:
                    X.append(np.load(f).squeeze())
                    subjects.append(os.path.basename(subj))

            if subj_idx % 25 == 0 or subj_idx == len(subj_list):
                print(f"  Loaded {subj_idx}/{len(subj_list)} subjects")

            if max_subjects and len(X) >= max_subjects:
                break
        if max_subjects and len(X) >= max_subjects:
            break

    print(f"Total embeddings loaded: {len(X)}")
    return np.array(X), subjects

def load_labels(labels_csv, subjects):
    import pandas as pd
    df = pd.read_csv(labels_csv)
    subj_col = [c for c in df.columns if "subject" in c.lower() or "id" in c.lower()][0]
    label_col = [c for c in df.columns if "label" in c.lower() or "cdr" in c.lower() or "dx" in c.lower()][0]
    label_map = dict(zip(df[subj_col].astype(str), df[label_col]))
    y = []
    for s in subjects:
        m = re.search(r"(OAS3\d+|OAS\d+)", s)
        key = m.group(1) if m else s
        y.append(label_map.get(key, label_map.get(s, -1)))
    return np.array(y)

# ── Visualization ──────────────────────────────────────────────────

def plot_pca_scatter(embeddings, output_path):
    print("[viz] PCA 2D scatter...")
    reduced = PCA(n_components=2).fit_transform(embeddings.reshape(embeddings.shape[0], -1))
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.8)
    plt.title('PCA Visualization of MRI Embeddings')
    plt.xlabel('PC 1'); plt.ylabel('PC 2')
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[save] {output_path}")

def plot_umap_trajectories(feats, labels, subj_days, output_path):
    print("[viz] UMAP trajectories...")
    pca = PCA(n_components=min(feats.shape[0], feats.shape[1], 50)).fit_transform(feats)
    u = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(pca)

    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10", len(uniq))
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), height_ratios=[2, 1])

    for i, subj in enumerate(uniq):
        idx = [j for j, l in enumerate(labels) if l == subj]
        if not idx: continue
        idx = np.array(idx)[np.argsort([subj_days[j] for j in idx])]
        axes[0].plot(u[idx, 0], u[idx, 1], "-", color=cmap(i), alpha=0.8, lw=1)
        axes[0].scatter(u[idx[0], 0], u[idx[0], 1], color=cmap(i), edgecolors="k", s=35, alpha=0.6)
        axes[0].scatter(u[idx[-1], 0], u[idx[-1], 1], color=cmap(i), s=45, alpha=0.9, label=subj)

    axes[0].set_title("UMAP projection (chronological trajectories)")
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
    axes[0].grid(alpha=0.3)
    axes[0].legend(title="Subject", fontsize=7, ncol=2, bbox_to_anchor=(1.05, 1),
                   loc="upper left", frameon=False)

    mean_dists = []
    for subj in uniq:
        idx = [j for j, l in enumerate(labels) if l == subj]
        if len(idx) < 2: continue
        order = np.argsort(np.array(subj_days)[idx])
        sf = feats[idx][order]
        mean_dists.append(np.mean([cosine_distances([sf[k]], [sf[k+1]])[0, 0]
                                   for k in range(len(sf) - 1)]))

    axes[1].bar(range(len(mean_dists)), mean_dists,
                color=[cmap(i) for i in range(len(mean_dists))])
    axes[1].set_xticks(range(len(mean_dists)))
    axes[1].set_xticklabels(uniq, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Mean cosine dist\n(consecutive scans)")
    axes[1].set_title("Embedding stability per subject")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight"); plt.close()
    print(f"[save] {output_path}")

# ── Linear probe ───────────────────────────────────────────────────

def linear_probe_evaluation(X, y, output_dir):
    print("[eval] Linear probe...")
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_all = np.zeros(len(y), dtype=np.float64)
    y_pred_all = np.zeros(len(y), dtype=int)

    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        y_pred_all[va] = clf.predict(X[va])
        y_prob_all[va] = clf.predict_proba(X[va])[:, 1]

    auc = roc_auc_score(y, y_prob_all)
    cm = confusion_matrix(y, y_pred_all)
    fpr, tpr, _ = roc_curve(y, y_prob_all)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Linear Probe ROC"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "linear_probe_ROC.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Linear Probe CM')
    plt.savefig(os.path.join(output_dir, "linear_probe_CM.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Linear Probe AUC: {auc:.4f}")
    return auc

# ── Main ───────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "gcs":
        X, subjects = load_embeddings_from_gcs(args.gcs_bucket, args.epoch, args.max_subjects)
        plot_pca_scatter(X, os.path.join(args.output_dir, "pca_visualization.png"))

        if args.labels_csv:
            y = load_labels(args.labels_csv, subjects)
            mask = y >= 0
            if mask.sum() < 10:
                print(f"[warn] only {mask.sum()} matched labels; skipping probe")
            else:
                linear_probe_evaluation(X[mask], y[mask], args.output_dir)
        else:
            print("[warn] no --labels_csv; skipping linear probe")

    elif args.mode == "local":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[env] device={device}")

        model = ClinImCL().to(device)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "model" in ckpt: ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
        model.eval()
        print(f"[load] {args.ckpt}")

        from torch.serialization import add_safe_globals
        try:
            from monai.data.meta_tensor import MetaTensor
            from monai.utils.enums import TraceKeys
            add_safe_globals([MetaTensor, TraceKeys, np.ndarray])
        except Exception:
            add_safe_globals([np.ndarray])

        paths = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
        print(f"[data] {len(paths)} .pt files")
        feats, labels, subj_days = [], [], []

        with torch.no_grad():
            for p in paths:
                vol = torch.load(p, map_location="cpu", weights_only=False)
                vol = torch.as_tensor(vol, dtype=torch.float32)
                if vol.ndim == 3: vol = vol.unsqueeze(0)
                if vol.shape[0] != 1: vol = vol[:1]
                vol = F.interpolate(vol.unsqueeze(0), size=(96, 96, 96),
                                    mode="trilinear", align_corners=False).squeeze(0)
                _, h = model(vol.unsqueeze(0).to(device))
                feats.append(h.cpu().numpy().squeeze())
                m = re.search(r"(OAS3\d+)", os.path.basename(p))
                labels.append(m.group(1) if m else "unknown")
                subj_days.append(int(re.search(r"d(\d+)", os.path.basename(p)).group(1)))

        plot_umap_trajectories(np.stack(feats), labels, subj_days,
                               os.path.join(args.output_dir, "umap_trajectories.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gcs", "local"], required=True)
    parser.add_argument("--output_dir", default="./visualizations")
    parser.add_argument("--gcs_bucket", default="clinimcl-data/OASIS3/train_new_outputs_REAL")
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--labels_csv", help="CSV with subject IDs and clinical labels")
    parser.add_argument("--ckpt", help="Checkpoint path (local mode)")
    parser.add_argument("--data_dir", help="Directory of .pt files (local mode)")
    args = parser.parse_args()

    if args.mode == "local" and (not args.ckpt or not args.data_dir):
        parser.error("--ckpt and --data_dir required for local mode")

    main(args)
