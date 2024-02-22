import numpy as np
class cluster:

    def __init__(self, k=5, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X, balanced=False):
        X = np.array(X)
        n, d = X.shape

        # Choosing centroids
        centroids = X[np.random.choice(n, self.k, replace=False)]

        for _ in range(self.max_iterations):
            clusters = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

            if balanced:
                new_centroids = np.zeros_like(centroids)
                for i in range(self.k):
                    cluster_instances = X[clusters == i]
                    if len(cluster_instances) > 0:
                        new_centroids[i] = cluster_instances.mean(axis=0)
                    else:
                        new_centroids[i] = centroids[i]
            else:
                new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])

            if np.all(np.equal(centroids, new_centroids)):
                break

            centroids = new_centroids

        return clusters, centroids