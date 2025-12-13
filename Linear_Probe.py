import torch
import gcsfs
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

fs = gcsfs.GCSFileSystem(token='google_default')
bucket_path = "clinimcl-data/OASIS3/train_new_outputs_REAL"
parts = sorted([p for p in fs.ls(bucket_path) if "part_" in p])

X, y = [], []

print("Starting embedding loading for linear probe...")
for part_idx, part in enumerate(parts, 1):
    subjects = sorted(fs.ls(part))
    print(f"\n--- Part {part_idx}/{len(parts)}: {os.path.basename(part)} ---")
    for subj_idx, subj in enumerate(subjects, 1):
        embed_path = f"{subj}/epoch_040/embedding.npy"
        if fs.exists(embed_path):
            with fs.open(embed_path, 'rb') as f:
                emb = np.load(f).squeeze()
                X.append(emb)
                label = hash(subj) % 2
                y.append(label)
        if subj_idx % 25 == 0 or subj_idx == len(subjects):
            print(f"✅ Loaded {subj_idx}/{len(subjects)} subjects from {os.path.basename(part)}")
print(f"\n✅ All embeddings loaded: {len(X)} subjects.")

X = np.array(X).reshape(-1, 128)
y = np.repeat(y, 4)
print(f"Embeddings shape after reshaping: {X.shape}")
assert X.ndim == 2, f"Embeddings must be 2-dimensional, got {X.ndim}"

# Linear Classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
y_pred = clf.predict(X)
y_prob = clf.predict_proba(X)[:, 1]

# Evaluation
auc = roc_auc_score(y, y_prob)
cm = confusion_matrix(y, y_pred)
fpr, tpr, _ = roc_curve(y, y_prob)

# ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Linear Probe ROC")
plt.legend()
roc_path = f"{bucket_path}/linear_probe_ROC.png"
with fs.open(roc_path, 'wb') as f:
    plt.savefig(f)
plt.close()
print(f"✅ ROC Curve saved: {roc_path}")

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Linear Probe Confusion Matrix')
cm_path = f"{bucket_path}/linear_probe_CM.png"
with fs.open(cm_path, 'wb') as f:
    plt.savefig(f)
plt.close()
print(f"✅ Confusion Matrix saved: {cm_path}")
print(f"Linear Probe Completed | AUC: {auc:.4f}")