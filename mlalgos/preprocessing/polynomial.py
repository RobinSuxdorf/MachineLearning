import numpy as np
from mlalgos import ArrayLike
from mlalgos.helpers import check_array

class PolynomialFeatures:
    def __init__(self, degree: int = 2, include_bias: bool = True):
        self._degree = degree
        self._include_bias = include_bias

    def fit(self, X: ArrayLike) -> None:
        pass

    def transform(self, X: ArrayLike) -> np.ndarray:
        pass

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
