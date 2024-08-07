import numpy as np
from typing import Any, Optional, Union


class StandardScaler:
    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self._with_mean = with_mean
        self._with_std = with_std

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, list[Any]]) -> None:
        if isinstance(X, list):
            X = np.array(X)

        if self._with_mean:
            self._mean = np.mean(X, axis=0)

        if self._with_std:
            self._std = np.std(X, axis=0)

    def transform(self, X: Union[np.ndarray, list[Any]]) -> np.ndarray:
        if isinstance(X, list):
            X = np.array(X)

        if self._with_mean and self._mean is not None:
            X = X - self._mean

        if self._with_std and self._std is not None:
            X = X / self._std

        return X

    def fit_transform(self, X: Union[np.ndarray, list[Any]]) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, list[Any]]) -> np.ndarray:
        if isinstance(X, list):
            X = np.array(X)

        if self._with_std and self._std is not None:
            X = X * self._std

        if self._with_mean and self._mean is not None:
            X = X + self._mean

        return X
