import torch
import gcsfs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import umap.umap_ as umap
import torch.nn as nn
import os
import random

torch.serialization.add_safe_globals({
    'monai.data.meta_tensor': ['MetaTensor'],
    'numpy._core.multiarray': ['_reconstruct']
})

fs = gcsfs.GCSFileSystem(token='google_default')
bucket_path = "clinimcl-data/OASIS3"
test_bucket = f"{bucket_path}/splits_new/test_new"
output_bucket = f"{bucket_path}/test_new_outputs"

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(16, 128)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

model = SimpleCNN().cuda()
model.eval()
parts = sorted([p for p in fs.ls(test_bucket) if "part_" in p])
all_embeddings, all_labels = [], []
print("Starting testing process")
for part_idx, part in enumerate(parts, 1):
    files = fs.ls(part)
    random.shuffle(files)
    selected_files = files[:15]
    embeddings, labels = [], []
    for file in selected_files:
        with fs.open(file, 'rb') as f:
            data = torch.load(f, weights_only=False).cuda().unsqueeze(0)
        with torch.no_grad():
            embed = model(data)
        embeddings.append(embed.cpu().numpy().flatten())
        subject_id = int(os.path.basename(file).split('_')[-2][1:])
        labels.append(subject_id % 2)
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    all_embeddings.append(embeddings)
    all_labels.append(labels)
    print(f"Part {part_idx} ({os.path.basename(part)}) completed with {len(selected_files)} scans.")
all_embeddings = np.vstack(all_embeddings)
all_labels = np.concatenate(all_labels)

# PCA, UMAP, and t-SNE
pca = PCA(n_components=2).fit_transform(all_embeddings)
umap_embed = umap.UMAP(random_state=42).fit_transform(all_embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1)).fit_transform(all_embeddings)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].scatter(pca[:, 0], pca[:, 1], c=all_labels, cmap='coolwarm')
ax[0].set_title('PCA')
ax[1].scatter(umap_embed[:, 0], umap_embed[:, 1], c=all_labels, cmap='coolwarm')
ax[1].set_title('UMAP')
ax[2].scatter(tsne[:, 0], tsne[:, 1], c=all_labels, cmap='coolwarm')
ax[2].set_title('t-SNE')
plt.tight_layout()

fs.makedirs(output_bucket, exist_ok=True)
with fs.open(f"{output_bucket}/testing_projections.png", 'wb') as f:
    plt.savefig(f)
plt.close()

# Linear Probe (Logistic Regression)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(all_embeddings, all_labels)
predictions = clf.predict(all_embeddings)

# Confusion Matrix
cm = confusion_matrix(all_labels, predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')

with fs.open(f"{output_bucket}/testing_confusion_matrix.png", 'wb') as f:
    plt.savefig(f)
plt.close()

print("\nTesting completed successfully.")
print(f"Visualizations saved at {output_bucket}")