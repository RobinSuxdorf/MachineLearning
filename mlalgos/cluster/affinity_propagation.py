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
        pass

    @property
    def cluster_centers(self) -> np.ndarray:
        pass

    @property
    def labels(self) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
