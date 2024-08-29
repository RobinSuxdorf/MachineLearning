import pytest
from contextlib import nullcontext
from typing import Any, ContextManager
from mlalgos import ArrayLike
import numpy as np
from mlalgos.helpers import check_array, check_length, check_type


@pytest.mark.parametrize(
    "X, expectation",
    [([1, 2, 3], np.array([1, 2, 3])), (np.array([1, 2, 3]), np.array([1, 2, 3]))],
)
def test_check_array_single_list(X: ArrayLike, expectation: np.ndarray) -> None:
    assert np.array_equal(check_array(X), expectation)


def test_check_array_multiple_lists() -> None:
    X = check_array([1, 2], np.array([3, 4]))
    expected = [np.array([1, 2]), np.array([3, 4])]
    assert np.array_equal(X, expected)


@pytest.mark.parametrize("X", [1, 3.14, "Hello World", True])
def test_check_array_wrong_type(X: Any) -> None:
    with pytest.raises(ValueError):
        check_array(X)


@pytest.mark.parametrize(
    "list1,list2,expectation",
    [
        (
            [],
            [0],
            pytest.raises(ValueError, match=r"The inputs do not have the same length."),
        ),
        ([1, 2, 3], [4, 5, 6], nullcontext()),
    ],
)
def test_check_length(
    list1: ArrayLike, list2: ArrayLike, expectation: ContextManager
) -> None:
    with expectation:
        assert check_length(list1, list2) is None


@pytest.mark.parametrize(
    "list1,list2,expectation",
    [
        ([], [0], pytest.raises(ValueError, match=r"The first array is empty.")),
        (
            [0, ""],
            [0, ""],
            pytest.raises(
                ValueError,
                match=r"Not all elements in the first array are of the same type.",
            ),
        ),
        (
            [0, 1, 2],
            ["bird", "cat", "dog"],
            pytest.raises(
                ValueError,
                match=r"Not all elements in the second array are of the same type as the first element of the first array.",
            ),
        ),
        ([0.5, 1.3, 2.7], [2.72, 3.14], nullcontext()),
    ],
)
def test_check_type(
    list1: ArrayLike, list2: ArrayLike, expectation: ContextManager
) -> None:
    with expectation:
        assert check_type(list1, list2) is None
