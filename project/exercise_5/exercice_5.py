from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns

# Chargement et analyse exploratoire du jeu de données Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Conversion en DataFrame pour une analyse plus facile
df = pd.DataFrame(X, columns=feature_names)
print(df.describe())  # Statistiques descriptives
sns.pairplot(df, diag_kind='kde')  # Visualisation des distributions
plt.show()

# Corrélation entre les caractéristiques
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PCA pour la réduction de dimensionnalité
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Visualisation des clusters en 2D et 3D
# Plot the 2D PCA-reduced data colored by cluster assignment
plt.figure(figsize=(10, 7))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters, cmap='viridis', label='Cluster')
plt.title('2D PCA of Iris Dataset (colored by cluster assignment)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Plot the 3D PCA-reduced data colored by cluster assignment
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=clusters, cmap='viridis', label='Cluster')
ax.set_title('3D PCA of Iris Dataset (colored by cluster assignment)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.legend()

plt.show()

# Calcul de plusieurs métriques d'évaluation
silhouette_avg = silhouette_score(X_scaled, clusters)
calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)
print(f"Score de silhouette: {silhouette_avg}")
print(f"Score de Calinski-Harabasz: {calinski_harabasz}")