import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from mpl_toolkits.mplot3d import Axes3D
import time

# Chargement des données
data_path = 'data.npy'
labels_path = 'labels.npy'
data = np.load(data_path)
labels = np.load(labels_path)

# Configuration de la figure pour les graphiques
fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # 3 méthodes (lignes) x 2 dimensions (colonnes)

# ------------------- PCA Section -------------------
start_time_pca = time.time()

# PCA pour 2 dimensions
pca_2d = PCA(n_components=2)
data_pca_2d = pca_2d.fit_transform(data)

# PCA pour 3 dimensions
pca_3d = PCA(n_components=3)
data_pca_3d = pca_3d.fit_transform(data)

end_time_pca = time.time()

# Affichage PCA 2D et 3D
axs[0, 0].scatter(data_pca_2d[:, 0], data_pca_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
axs[0, 0].set_title('PCA - 2 Dimensions')
axs[0, 1] = fig.add_subplot(3, 2, 2, projection='3d')
axs[0, 1].scatter(data_pca_3d[:, 0], data_pca_3d[:, 1], data_pca_3d[:, 2], c=labels, cmap='viridis', alpha=0.5)
axs[0, 1].set_title('PCA - 3 Dimensions')

# ------------------- t-SNE Section -------------------
start_time_tsne = time.time()

# t-SNE pour 2 dimensions
tsne_2d = TSNE(n_components=2, random_state=42)
data_tsne_2d = tsne_2d.fit_transform(data)

end_time_tsne = time.time()

# Affichage t-SNE 2D
axs[1, 0].scatter(data_tsne_2d[:, 0], data_tsne_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
axs[1, 0].set_title('t-SNE - 2 Dimensions')

# ------------------- Isomap Section -------------------
start_time_isomap = time.time()

# Isomap pour 2 dimensions
isomap_2d = Isomap(n_components=2)
data_isomap_2d = isomap_2d.fit_transform(data)

# Isomap pour 3 dimensions
isomap_3d = Isomap(n_components=3)
data_isomap_3d = isomap_3d.fit_transform(data)

end_time_isomap = time.time()

# Affichage Isomap 2D
axs[2, 0].scatter(data_isomap_2d[:, 0], data_isomap_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
axs[2, 0].set_title('Isomap - 2 Dimensions')
axs[2, 1] = fig.add_subplot(3, 2, 6, projection='3d')
axs[2, 1].scatter(data_isomap_3d[:, 0], data_isomap_3d[:, 1], data_isomap_3d[:, 2], c=labels, cmap='viridis', alpha=0.5)
axs[2, 1].set_title('Isomap - 3 Dimensions')

plt.tight_layout()
plt.show()

# ------------------- Performance Comparison -------------------
print("\nPerformance Comparison:")
pca_time = end_time_pca - start_time_pca
tsne_time = end_time_tsne - start_time_tsne
isomap_time = end_time_isomap - start_time_isomap
print("PCA Time: {:.2f} seconds".format(pca_time))
print("t-SNE Time: {:.2f} seconds".format(tsne_time))
print("Isomap Time: {:.2f} seconds".format(isomap_time))
