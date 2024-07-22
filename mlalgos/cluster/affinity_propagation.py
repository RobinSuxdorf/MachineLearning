from typing import Optional
import numpy as np


class AffinityPropagation:
    def __init__(
        self,
        damping: float = 0.5,
        max_iter: int = 300,
        convergence_iter: int = 15,
        preferences: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self._damping = damping
        self._max_iter = max_iter
        self._convergence_iter = convergence_iter
        self._preferences = preferences
        
        if random_state:
            np.random.seed(random_state)

        self._cluster_centers: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    @property
    def cluster_centers(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def fit(self, X: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
