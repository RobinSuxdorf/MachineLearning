import numpy as np
from mlalgos.preprocessing import train_test_split

def test_train_test_split() -> None:
    X, y = np.arange(10).reshape((5, 2)), range(5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    expected_X_train = np.array([[4, 5], [0, 1], [6, 7]])
    expected_y_train = np.array([2, 0, 3])
    expected_X_test = np.array([[2, 3], [8, 9]])
    expected_y_test = np.array([1, 4])

    assert np.array_equal(X_train, expected_X_train)
    assert np.array_equal(y_train, expected_y_train)
    assert np.array_equal(X_test, expected_X_test)
    assert np.array_equal(y_test, expected_y_test)
