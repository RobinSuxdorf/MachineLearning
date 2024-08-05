import pytest
import numpy as np

from mlalgos.preprocessing import OneHotEncoder

@pytest.mark.parametrize(
    "X, expected", [
        (np.array([0, 1, 2, 1, 0]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])),
        (np.array(["bird", "cat", "dog", "cat", "bird"]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])),
        (np.array([['Male', 1], ['Female', 3], ['Female', 2]]), np.array([[1, 0, 0, 0, 1], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0]]))
    ]
)
def test_one_hot_encoder_fit_transform(X: np.ndarray, expected: np.ndarray) -> None:
    encoder = OneHotEncoder()

    transformed = encoder.fit_transform(X)

    assert np.array_equal(transformed, expected)

# test inverse transform

# test error handling

# test non np input