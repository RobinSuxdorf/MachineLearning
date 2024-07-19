import pytest
import re
import numpy as np
from mlalgos.preprocessing import train_test_split

def test_train_test_split() -> None:
    X, y = np.arange(10).reshape((5, 2)), range(5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    expected_X_train = np.array([[0, 1], [4, 5], [6, 7]])
    expected_y_train = np.array([0, 2, 3])
    expected_X_test = np.array([[2, 3], [8, 9]])
    expected_y_test = np.array([1, 4])

    assert np.array_equal(X_train, expected_X_train)
    assert np.array_equal(y_train, expected_y_train)
    assert np.array_equal(X_test, expected_X_test)
    assert np.array_equal(y_test, expected_y_test)

def test_train_test_split_with_different_length() -> None:
    with pytest.raises(AssertionError, match=r"Length of X and y should match."):
        train_test_split(np.array([]), np.array([0]))

def test_train_test_split_with_wrong_test_size() -> None:
    with pytest.raises(AssertionError, match=re.escape("test_size should be in the interval [0,1].")):
        train_test_split(np.array([]), np.array([]), test_size=-1)