import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gcsfs

fs = gcsfs.GCSFileSystem(token='google_default')
bucket = 'clinimcl-data/OASIS3/train_new_outputs/part_01_output'
embedding_files = [file for file in fs.glob(f"{bucket}/*/epoch_001/embedding.npy")[:50]]
embeddings = np.stack([np.load(fs.open(file, 'rb')) for file in embedding_files])
embeddings = embeddings.reshape(embeddings.shape[0], -1)

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.8)
plt.title('PCA Visualization of MRI Embeddings (Epoch 1)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig("pca_visualization.png")