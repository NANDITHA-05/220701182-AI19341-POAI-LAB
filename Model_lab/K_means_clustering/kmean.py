import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def kmeans(X, k, max_iterations=100):
    # Randomly initialize the cluster centers
    n_samples, n_features = X.shape
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iterations):
       
        clusters = [[] for _ in range(k)]
        for idx, sample in enumerate(X):
            distances = [euclidean_distance(sample, point) for point in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(idx)

       
        new_centroids = np.zeros((k, n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(X[cluster], axis=0)
            new_centroids[cluster_index] = cluster_mean

       
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

   
    labels = np.empty(n_samples)
    for cluster_index, cluster in enumerate(clusters):
        for sample_index in cluster:
            labels[sample_index] = cluster_index

    return centroids, labels


X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)


k = 4  # Number of clusters
centroids, labels = kmeans(X, k)


plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids')
plt.legend()
plt.show()
