from typing import Any, Union
import numpy as np


class OneHotEncoder():
    def __init__(self, handle_unknown: str = "error") -> None:
        self._handle_unknown = handle_unknown
        self._categories: list[np.ndarray] = []
        self._value_to_idx: dict[Any, int] = {}
        self._idx_to_value: dict[int, Any] = {}
        self._dimension: int = 0

    def fit(self, X: Union[np.ndarray, list[Any]]) -> None:
        if isinstance(X, list):
            X = np.array(X, dtype=object)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._categories = [np.unique(col) for col in zip(*X)]

        idx = 0

        for cat in self._categories:
            for value in cat:
                self._value_to_idx[value] = idx
                self._idx_to_value[idx] = value

                idx += 1
        
        self._dimension = idx

    def _row_to_one_hot(self, row: np.ndarray) -> np.ndarray:
        one_hot = np.zeros(self._dimension)

        ids: list[int] = []

        for x in row:
            try:
                ids.append(self._value_to_idx[x])
            except KeyError:
                if self._handle_unknown == "error":
                    raise KeyError(f"The value '{x}' is unknown.")

        one_hot[ids] = 1
        return one_hot

    def transform(self, X: Union[np.ndarray, list[Any]]) -> np.ndarray:
        if isinstance(X, list):
            X = np.array(X, dtype=object)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return np.array([self._row_to_one_hot(row) for row in X])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def _inverse_single_vector(self, one_hot: np.ndarray) -> np.ndarray:
        if self._handle_unknown == "error" and np.sum(one_hot) != len(self._categories):
            raise ValueError(f"The one hot encoded vector {one_hot} does not match the fitting data.")

        data: list[Any] = []

        for cat in self._categories:
            found = False
            for idx, x in enumerate(one_hot):
                if x == 1 and self._idx_to_value[idx] in cat:
                    data.append(self._idx_to_value[idx])
                    found = True
                    break
            if not found:
                data.append(None)

        return np.array(data, dtype=object)

    def inverse_transform(self, X: Union[np.ndarray, list[Any]]) -> np.ndarray:
        if isinstance(X, list):
            X = np.array(X, dtype=object)

        if X.ndim == 1:
            return self._inverse_single_vector(X)

        return np.array([self._inverse_single_vector(x) for x in X], dtype=object)
