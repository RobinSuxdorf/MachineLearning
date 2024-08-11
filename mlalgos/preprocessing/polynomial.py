import numpy as np
from mlalgos import ArrayLike
from mlalgos.helpers import check_array
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree: int = 2, include_bias: bool = True):
        self._degree = degree
        self._include_bias = include_bias
        self._combinations: list[tuple] = []

    def fit(self, X: ArrayLike) -> None:
        X = check_array(X)

        _, n_features = X.shape

        start_degree = 0 if self._include_bias else 1

        for d in range(start_degree, self._degree + 1):
            self._combinations.extend(list(combinations_with_replacement(range(n_features), d)))


    def transform(self, X: ArrayLike) -> np.ndarray:
        X = check_array(X)

        n_samples, _ = X.shape
        n_output_features = len(self._combinations)

        X_out = np.zeros((n_samples, n_output_features))

        for i, comb in enumerate(self._combinations):
            X_out[:, i] = np.prod(X[:, comb], axis=1)

        return X_out

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
