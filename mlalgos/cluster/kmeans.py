import numpy as np


class KMeans:
    def __init__(self, n_clusters: int = 2, random_state: int = 0) -> None:
        """
        Initialization method for k-means clustering algorithm.
        """
        self._n_clusters = n_clusters
        self._cluster_centers: np.ndarray = None
        self._labels: np.ndarray = None

        np.random.seed(random_state)

    @property
    def cluster_centers(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def _calculate_cluster_center(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate the cluster center of given points.

        Args:
            points (np.ndarray): Points to calculate the cluster center for.

        Returns:
            np.ndarray: The cluster center of the points.
        """
        return np.mean(points, axis=0)

    def fit(self, X: np.ndarray, tol: float = 1e-4, max_iter: int = 300) -> None:
        """
        Compute k-means clustering.

        Args:
            X (np.ndarray): Training examples to cluster.
            tol (float): Tolernace to declare convergence.
            max_iter (int): Maximal number of iterations of the k-means algorithm.
        """
        random_indices = np.random.choice(
            X.shape[0], size=self._n_clusters, replace=False
        )
        cluster_centers = X[random_indices]

        for _ in range(max_iter):
            clusters = [[] for _ in range(self._n_clusters)]
            labels = np.empty(X.shape[0], dtype=int)

            for idx, point in enumerate(X):
                distances = np.linalg.norm(point - cluster_centers, axis=1)
                cluster_assignment = np.argmin(distances)
                clusters[cluster_assignment].append(point)
                labels[idx] = cluster_assignment

            new_cluster_centers = np.array(
                [self._calculate_cluster_center(cluster) for cluster in clusters]
            )

            if np.all(
                np.linalg.norm(new_cluster_centers - cluster_centers, axis=1) < tol
            ):
                break

            cluster_centers = new_cluster_centers

        self._cluster_centers = new_cluster_centers
        self._labels = labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each example in X belongs to.

        Args:
            X (np.ndarray): The data to predict.

        Returns:
            np.ndarray: The indices of the cluster each sample belongs to.
        """
        response = []

        for point in X:
            distances = np.linalg.norm(point - self._cluster_centers, axis=1)
            cluster_assignment = np.argmin(distances)
            response.append(cluster_assignment)

        return np.array(response)
