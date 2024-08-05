from typing import Any
import numpy as np


class OneHotEncoder():
    def __init__(self) -> None:
        self._value_to_idx: dict[Any, str] = {}
        self._dimension: int = 0

    def fit(self, X: np.ndarray) -> None:
        unique_values = np.unique(X)
        self._value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
        self._dimension = len(self._value_to_idx)

    def _row_to_one_hot(self, row: np.ndarray) -> np.ndarray:
        one_hot = np.zeros(self._dimension)
        ids = [self._value_to_idx[x] for x in row]
        one_hot[ids] = 1
        return one_hot

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return np.array([self._row_to_one_hot(row) for row in X])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass
