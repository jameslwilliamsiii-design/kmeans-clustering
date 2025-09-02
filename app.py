# Clustering with K-Means and PCA Visualization on Telescope Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv('telescope.csv')

# Drop duplicate rows
df = df.drop_duplicates()

# Show data summary
print(f"Dataset shape after removing duplicates: {df.shape}")
print(df.info())
print(df.describe())

# Correlation heatmap (excluding target column)
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=['class']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Prepare data for clustering
features = df.drop(columns=['class'])  # Unsupervised: drop labels
scaled_features = StandardScaler().fit_transform(features)

# Elbow Method to find optimal k
wcss = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.tight_layout()
plt.show()

# Apply KMeans with chosen k (e.g., k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Evaluate clustering with silhouette score
sil_score = silhouette_score(scaled_features, df['Cluster'])
print(f"Silhouette Score: {sil_score:.4f}")

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['Cluster'], palette='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering (PCA-Reduced Features)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
