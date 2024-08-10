from typing import Optional
from mlalgos import ArrayLike
from mlalgos.helpers import check_array
import numpy as np


class AffinityPropagation:
    def __init__(
        self,
        damping: float = 0.5,
        max_iter: int = 300,
        convergence_iter: int = 15,
        preferences: Optional[ArrayLike] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialization method for the Affinity Propagation clustering algorithm.

        Args:
            damping (float): Damping factor between 0.5 and 1.
            max_iter (int): Maximum number of iterations.
            convergence_iter (int): Number of iterations with no change in exemplars to declare convergence.
            preferences (Optional[ArrayLike]): Preferences for each data point. If None, set to median similarity.
            random_state (Optional[int]): Random seed for reproducibility.
        """
        self._damping = damping
        self._max_iter = max_iter
        self._convergence_iter = convergence_iter
        self._preferences = check_array(preferences) if preferences else None

        if random_state:
            np.random.seed(random_state)

        self._cluster_centers: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._sim_matrix: Optional[np.ndarray] = None

    @property
    def cluster_centers(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def _negative_squared_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the negative squared distance.

        Args:
            x (np.ndarray): A 1-dimensional array representing the first point.
            y (np.ndarray): A 1-dimensional array representing the second point.

        Returns:
            float: The negative squared distance between x and y.
        """
        return -np.power(np.linalg.norm(x - y), 2)

    def _calculate_similarity_matrix(self, X: np.ndarray) -> None:
        """
        Calculates the similarity matrix for the samples in X.

        Args:
            X (np.ndarray): The samples for which the similarity matrix will be calculated.
        """
        self._sim_matrix = np.array(
            [[self._negative_squared_distance(x, y) for x in X] for y in X]
        )

    def _calculate_responsibility_matrix(
        self, sim_matrix: np.ndarray, avail_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the new responsibility matrix from the similarity matrix and the availability matrix.

        Args:
            sim_matrix (np.ndarray): The similarity matrix.
            avail_matrix (np.ndarray): The previous availability matrix.

        Returns:
            np.ndarray: The new responsibility matrix.
        """
        l = len(sim_matrix)
        new_resp_matrix = np.zeros((l, l))

        for i in range(l):
            for k in range(l):
                max_val = max(
                    [avail_matrix[i][j] + sim_matrix[i][j] for j in range(l) if j != k]
                )
                new_resp_matrix[i][k] = sim_matrix[i][k] - max_val

        return new_resp_matrix

    def _calculate_availability_matrix(self, resp_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the new availability matrix from the previous responsibility matrix.

        Args:
            resp_matrix (np.ndarray): The previous responsibility matrix.

        Returns:
            np.ndarray: The new availability matrix.
        """
        l = len(resp_matrix)
        new_avail_matrix = np.zeros((l, l))

        for i in range(l):
            for k in range(l):
                if i != k:
                    sum_val = sum(
                        [
                            max(0, resp_matrix[j][k])
                            for j in range(l)
                            if j != i and j != k
                        ]
                    )
                    new_avail_matrix[i][k] = min(0, resp_matrix[k][k] + sum_val)
                else:
                    sum_val = sum(
                        [max(0, resp_matrix[j][k]) for j in range(l) if j != k]
                    )
                    new_avail_matrix[k][k] = sum_val

        return new_avail_matrix

    def _assign_labels(self, X: np.ndarray, exemplars: np.ndarray) -> np.ndarray:
        """
        Assign the labels the samples belong to.

        Args:
            X (np.ndarray): The samples to be labeled.
            exemplars (np.ndarray): The cluster centers.

        Returns:
            np.ndarray: An array containing the labels for the samples.
        """
        exemplar_labels = {exemplar: label for label, exemplar in enumerate(exemplars)}

        labels = np.full(len(X), -1)

        for i in range(len(X)):
            if i in exemplar_labels:
                labels[i] = exemplar_labels[i]
            else:
                nearest_exemplar = exemplars[np.argmax(self._sim_matrix[i, exemplars])]
                labels[i] = exemplar_labels[nearest_exemplar]

        return labels

    def fit(self, X: ArrayLike) -> None:
        """
        Compute the affinity propagation clustering.

        Args:
            X (ArrayLike): Training examples to cluster.

        """
        X = check_array(X)

        self._calculate_similarity_matrix(X)

        if self._preferences is None:
            self._preferences = np.median(self._sim_matrix)

        np.fill_diagonal(self._sim_matrix, self._preferences)

        resp_matrix = np.zeros((len(X), len(X)))
        avail_matrix = np.zeros((len(X), len(X)))

        exemplars = np.array([])
        convergence_iter_counter = 0

        for _ in range(self._max_iter):
            new_resp_matrix = self._calculate_responsibility_matrix(
                self._sim_matrix, avail_matrix
            )

            new_avail_matrix = self._calculate_availability_matrix(resp_matrix)

            resp_matrix = (
                self._damping * resp_matrix + (1 - self._damping) * new_resp_matrix
            )
            avail_matrix = (
                self._damping * avail_matrix + (1 - self._damping) * new_avail_matrix
            )

            s = resp_matrix + avail_matrix

            new_exemplars = np.where(np.diag(s) > 0)[0]

            if np.array_equal(exemplars, new_exemplars):
                convergence_iter_counter += 1
            else:
                convergence_iter_counter = 0
                exemplars = new_exemplars

            if convergence_iter_counter >= self._convergence_iter:
                break

        self._cluster_centers = X[exemplars]
        self._labels = self._assign_labels(X, exemplars)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict the closest exemplar each example in X belongs to.

        Args:
            X (ArrayLike): The data to predict.

        Returns:
            np.ndarray: The indices of the exemplar each sample belongs to.
        """
        X = check_array(X)

        return np.array(
            [
                np.argmax(
                    [
                        self._negative_squared_distance(x, cluster_center)
                        for cluster_center in self._cluster_centers
                    ]
                )
                for x in X
            ]
        )
