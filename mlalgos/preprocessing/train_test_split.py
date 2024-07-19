import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays random train and test subset.

    Args:
        X (np.ndarray): Features dataset.
        y (np.ndarray): Target dataset.
        test_size (float): Proposition of the dataset to include in the test split. Must be an element of [0,1].
        random_state (int): Seed for random number generator.

    Returns:
        tuple: Split datasets as (X_train, X_test, y_train, y_test).
    """
    assert len(X) == len(y), "Length of X and y should match."
    assert (
        test_size >= 0 and test_size <= 1
    ), "test_size should be in the interval [0,1]."

    X = np.asarray(X)
    y = np.asarray(y)

    np.random.seed(random_state)

    size = round(len(X) * test_size)
    test_indices = np.random.choice(len(X), size=size, replace=False)
    train_indices = np.setdiff1d(np.arange(len(X)), test_indices)

    X_test = X[test_indices]
    y_test = y[test_indices]

    X_train = X[train_indices]
    y_train = y[train_indices]

    return X_train, X_test, y_train, y_test
