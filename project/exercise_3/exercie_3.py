import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Charger le dataset
data_path = 'data.npy' 
data = np.load(data_path)
scaled_data = StandardScaler().fit_transform(data)

# Préparation pour les graphiques
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Initialisation des listes pour sauvegarder les métriques
silhouette_scores_kmeans = []
calinski_harabasz_scores_kmeans = []

silhouette_scores_agglo = []
calinski_harabasz_scores_agglo = []

# Calcul des métriques pour K-Means et Clustering Agglomératif
for i in range(2, 11):
    # K-Means
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    labels_kmeans = kmeans.fit_predict(data)
    silhouette_scores_kmeans.append(silhouette_score(data, labels_kmeans))
    calinski_harabasz_scores_kmeans.append(calinski_harabasz_score(data, labels_kmeans))
    
    # Clustering Agglomératif
    agglo = AgglomerativeClustering(n_clusters=i)
    labels_agglo = agglo.fit_predict(scaled_data)
    silhouette_scores_agglo.append(silhouette_score(scaled_data, labels_agglo))
    calinski_harabasz_scores_agglo.append(calinski_harabasz_score(scaled_data, labels_agglo))

# Tracer les résultats pour K-Means
axes[0, 0].plot(range(2, 11), silhouette_scores_kmeans, marker='o')
axes[0, 0].set_title('Score Silhouette pour K-Means')
axes[0, 0].set_xlabel('Nombre de clusters')
axes[0, 0].set_ylabel('Score Silhouette')

axes[0, 1].plot(range(2, 11), calinski_harabasz_scores_kmeans, marker='o')
axes[0, 1].set_title('Score Calinski-Harabasz pour K-Means')
axes[0, 1].set_xlabel('Nombre de clusters')
axes[0, 1].set_ylabel('Score Calinski-Harabasz')

# Tracer les résultats pour Clustering Agglomératif
axes[1, 0].plot(range(2, 11), silhouette_scores_agglo, marker='o')
axes[1, 0].set_title('Score Silhouette pour Clustering Agglomératif')
axes[1, 0].set_xlabel('Nombre de clusters')
axes[1, 0].set_ylabel('Score Silhouette')

axes[1, 1].plot(range(2, 11), calinski_harabasz_scores_agglo, marker='o')
axes[1, 1].set_title('Score Calinski-Harabasz pour Clustering Agglomératif')
axes[1, 1].set_xlabel('Nombre de clusters')
axes[1, 1].set_ylabel('Score Calinski-Harabasz')

# Afficher les graphiques
plt.tight_layout()
plt.show()