import numpy as np
class cluster:

    def __init__(self, k=5, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        X = np.array(X)
        n, d = X.shape

        # Choosing centroids
        centroids = X[np.random.choice(n, self.k, replace=False)]

        for _ in range(self.max_iterations):
            clusters = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

            new_centroids = [np.mean(X[np.array(clusters) == i], axis=0) for i in range(self.k)]

            if np.all(np.equal(centroids, new_centroids)):
                break

            centroids = new_centroids

        return clusters, centroids