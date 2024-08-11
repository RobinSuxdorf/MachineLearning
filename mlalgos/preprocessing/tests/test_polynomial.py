import pytest
import numpy as np
from mlalgos import ArrayLike
from mlalgos.preprocessing import PolynomialFeatures

@pytest.mark.parametrize(
    "degree, include_bias, X, expected", [
        (1, False, np.arange(6).reshape(3, 2), np.arange(6).reshape(3, 2)),
        (1, True, np.arange(6).reshape(3, 2), np.array([[1., 0., 1.], [1., 2., 3.], [1., 4., 5.]])),
        (2, True, np.arange(6).reshape(3, 2), np.array([[1, 0, 1, 0, 0, 1], [1, 2, 3, 4, 6, 9], [1, 4, 5, 16, 20, 25]])),
        (3, True, np.arange(6).reshape(2, 3), np.array([[0, 1, 2, 0, 0, 0, 1, 2, 4] ,[3, 4, 5, 9, 12, 15, 16, 20, 25]]))
    ]
)
def test_polynomial_features(degree: int, include_bias: bool, X: ArrayLike, expected: np.ndarray) -> None:
    poly = PolynomialFeatures(degree, include_bias)
    transformed = poly.fit_transform(X)
    assert np.array_equal(transformed, expected)
