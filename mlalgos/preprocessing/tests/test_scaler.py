import pytest
import numpy as np
from typing import Any, Union
from mlalgos.preprocessing import StandardScaler

@pytest.mark.parametrize(
    "X, expected_transformed, with_mean, with_std", [
        ([[0, 0], [0, 0], [1, 1], [1, 1]], np.array([[-1, -1], [-1, -1], [1,  1], [1,  1]]), True, True),
        ([[0, 0], [0, 0], [1, 1], [1, 1]], np.array([[0, 0], [0, 0], [2,  2], [2,  2]]), False, True),
        ([[0, 0], [0, 0], [1, 1], [1, 1]], np.array([[-0.5, -0.5], [-0.5, -0.5], [0.5,  0.5], [0.5,  0.5]]), True, False),
        ([[0, 0], [0, 0], [1, 1], [1, 1]], np.array([[0, 0], [0, 0], [1, 1], [1, 1]]), False, False)
    ]
)
def test_standard_scaler_fit_transform(
    X: Union[np.ndarray, list[Any]], 
    expected_transformed: np.ndarray, 
    with_mean: bool, 
    with_std: bool
) -> None:
    standard_scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

    transformed = standard_scaler.fit_transform(X)

    assert np.array_equal(transformed, expected_transformed)

@pytest.mark.parametrize(
    "X, input, expected_transformed, with_mean, with_std", [
        ([[0, 0], [0, 0], [1, 1], [1, 1]], [2, 2], [3, 3], True, True),
        ([[0, 0], [0, 0], [1, 1], [1, 1]], [2, 2], [4, 4], False, True),
        ([[0, 0], [0, 0], [1, 1], [1, 1]], [2, 2], [1.5, 1.5], True, False),
        ([[0, 0], [0, 0], [1, 1], [1, 1]], [2, 2], [2, 2], False, False)
    ]
)
def test_standard_scaler_transform_inverse_transform(
    X: Union[np.ndarray, list[Any]], 
    input: Union[np.ndarray, list[Any]], 
    expected_transformed: np.ndarray, 
    with_mean: bool, 
    with_std: bool
) -> None:
    standard_scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

    standard_scaler.fit(X)

    transformed = standard_scaler.transform(input)

    assert np.array_equal(transformed, expected_transformed)

    inverse_transformed = standard_scaler.inverse_transform(transformed)

    assert np.array_equal(inverse_transformed, input)
