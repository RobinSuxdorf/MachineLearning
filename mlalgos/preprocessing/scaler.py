import numpy as np
from typing import Any, Optional, Union


class StandardScaler:
    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self._with_mean = with_mean
        self._with_std = with_std

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def mean(self) -> Optional[np.ndarray]:
        return self._mean

    @property
    def std(self) -> Optional[np.ndarray]:
        return self._std

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

class MinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (0, 1)) -> None:
        if feature_range[0] >= feature_range[1]:
            raise ValueError("The first value of feature_range must be smaller than the second value.")

        self._min = feature_range[0]
        self._max = feature_range[1]
        self._data_min: Optional[np.ndarray] = None
        self._data_max: Optional[np.ndarray] = None
        self._scale: Optional[float] = None

    @property
    def min_(self) -> float:
        return self._min

    @property
    def max_(self) -> float:
        return self._max 

    @property
    def data_min_(self) -> float:
        if self._data_min is None:
            raise AttributeError("data_min_ is not set. You need to call 'fit' first.")
        return self._data_min

    @property
    def data_max_(self) -> float:
        if self._data_max is None:
            raise AttributeError("data_max_ is not set. You need to call 'fit' first.")
        return self._data_max

    @property
    def scale_(self) -> float:
        if self._scale is None:
            raise AttributeError("scale_ is not set. You need to call 'fit' first.")
        return self._scale

    def fit(self, X: Union[np.ndarray, list[Any]]) -> None:
        if isinstance(X, list):
            X = np.array(X)

        self._data_min = X.min(axis=0)
        self._data_max = X.max(axis=0)

        self._scale = (self._max - self._min) / (self.data_max_ - self.data_min_)
        return self

    def transform(self, X: Union[np.ndarray, list[Any]]) -> np.ndarray:
        if isinstance(X, list):
            X = np.array(X)

        return (X - self._data_min) * self._scale + self._min

    def fit_transform(self, X: Union[np.ndarray, list[Any]]) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, list[Any]]) -> np.ndarray:
        if isinstance(X, list):
            X = np.array(X)

        return (X - self._min) / self._scale + self._data_min
