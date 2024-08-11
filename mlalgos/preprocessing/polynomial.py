import numpy as np
from typing import Optional
from mlalgos import ArrayLike
from mlalgos.helpers import check_array
from itertools import combinations_with_replacement


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.

    This class transforms input data by generating polynomial features up to a specified degree.
    These features consist of all polynomial combinations of the input features, including
    interaction terms. For example, for a 2-feature input [a, b], the 2nd-degree polynomial
    features would be [1, a, b, a^2, ab, b^2].
    """

    def __init__(self, degree: int = 2, include_bias: bool = True) -> None:
        self._degree = degree
        self._include_bias = include_bias
        self._combinations: Optional[list[tuple]] = None

    def fit(self, X: ArrayLike) -> None:
        """
        Compute the polynomial feature combinations for the input data X.

        Args:
            X (ArrayLike): The input data, which should be a 2D array-like structure with shape (n_samples, n_features).

        """
        X = check_array(X)

        n_features = X.shape[1]

        start_degree = 0 if self._include_bias else 1

        combinations: list[tuple] = []
        for d in range(start_degree, self._degree + 1):
            combinations.extend(
                list(combinations_with_replacement(range(n_features), d))
            )

        self._combinations = combinations

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform the input data X to include polynomial features.

        Args:
            X (ArrayLike): The input data, which should be a 2D array-like structure with shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed data, containing the polynomial features, with shape (n_samples, n_output_features).
        """
        X = check_array(X)

        if self._combinations is None:
            raise ValueError("The model has not been fitted yet.")

        n_samples = X.shape[0]
        n_output_features = len(self._combinations)

        X_out = np.zeros((n_samples, n_output_features))

        for i, comb in enumerate(self._combinations):
            X_out[:, i] = np.prod(X[:, comb], axis=1)

        return X_out

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Fit the model and transform the input data X in a single step.

        Args:
            X (ArrayLike): The input data, which should be a 2D array-like structure with shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed data, containing the polynomial features, with shape (n_samples, n_output_features).
        """
        self.fit(X)
        return self.transform(X)
