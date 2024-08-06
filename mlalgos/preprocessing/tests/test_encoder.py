import pytest
from typing import Any
import numpy as np

from mlalgos.preprocessing import OneHotEncoder


@pytest.mark.parametrize(
    "X, expected",
    [
        (
            [0, 1, 2, 1, 0],
            np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=object
            ),
        ),
        (
            np.array([0, 1, 2, 1, 0], dtype=object),
            np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=object
            ),
        ),
        (
            ["bird", "cat", "dog", "cat", "bird"],
            np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=object
            ),
        ),
        (
            np.array([["Male", 1], ["Female", 3], ["Female", 2]], dtype=object),
            np.array([[0, 1, 1, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]], dtype=object),
        ),
    ],
)
def test_one_hot_encoder_fit_transform(X: np.ndarray, expected: np.ndarray) -> None:
    encoder = OneHotEncoder()

    transformed = encoder.fit_transform(X)

    assert np.array_equal(transformed, expected)


def test_one_hot_encoder_handle_unknown_error() -> None:
    with pytest.raises(KeyError, match=r"The value '(\S+)' is unknown."):
        encoder = OneHotEncoder("error")

        X = np.array([0, 1, 2, 1, 0], dtype=object)

        encoder.fit(X)

        encoder.transform(np.array([0, 1, 2, 4], dtype=object))


def test_one_hot_encoder_handle_unknown_ignore() -> None:
    encoder = OneHotEncoder("ignore")

    X = np.array([0, 1, 2, 1, 0], dtype=object)

    encoder.fit(X)

    transformed = encoder.transform(np.array([0, 1, 2, 3], dtype=object))

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=object)

    assert np.array_equal(transformed, expected)


@pytest.mark.parametrize(
    "X, encoded_vector, expected",
    [
        ([0, 1, 2, 1, 0], [0, 1, 0], np.array([1], dtype=object)),
        (
            [["Male", 1], ["Female", 3], ["Female", 2]],
            [[1, 0, 0, 0, 1], [0, 1, 0, 1, 0]],
            np.array([["Female", 3], ["Male", 2]], dtype=object),
        ),
    ],
)
def test_one_hot_encoder_inverse_transform(
    X: np.ndarray, encoded_vector: np.ndarray, expected: np.ndarray
) -> None:
    encoder = OneHotEncoder()

    encoder.fit(X)

    inverse_transformed = encoder.inverse_transform(encoded_vector)

    assert np.array_equal(inverse_transformed, expected)


def test_one_hot_encoder_inverse_transform_handle_unknown_error() -> None:
    with pytest.raises(
        ValueError,
        match=r"The one hot encoded vector \[0 0 0 1 0\] does not match the fitting data.",
    ):
        encoder = OneHotEncoder("error")

        X = np.array([["Male", 1], ["Female", 3], ["Female", 2]], dtype=object)

        encoder.fit(X)

        encoder.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])


def test_one_hot_encoder_inverse_transform_handle_unknown_ignore() -> None:
    encoder = OneHotEncoder("ignore")

    X = np.array([["Male", 1], ["Female", 3], ["Female", 2]], dtype=object)

    encoder.fit(X)

    inverse_transformed = encoder.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])

    expected = np.array([["Male", 1], [None, 2]], dtype=object)

    assert np.array_equal(inverse_transformed, expected)
