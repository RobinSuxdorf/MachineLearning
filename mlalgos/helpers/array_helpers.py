from typing import Any
from mlalgos import ArrayLike
import numpy as np


def check_array(X: ArrayLike) -> np.ndarray:
    """
    Validates and converts input data to a numpy array.

    Args:
        X (ArrayLike): The input data to be checked and converted. This can be a list or a numpy array.

    Returns:
        np.ndarray: The validated and possibly converted input as a numpy array.
    """
    if isinstance(X, list):
        X = np.array(X)
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array or a list.")
    return X

def check_length(list1: list[Any], list2: list[Any]) -> None:
    """
    Checks if the two lists have the same length.

    Args:
        list1 (list[Any]): The first list to check.
        list2 (list[Any]): The second list to check.

    Raises:
        ValueError: If the lists do not have the same length.
    """
    if len(list1) != len(list2):
        raise ValueError("The lists do not have the same length.")


def check_type(list1: list[Any], list2: list[Any]) -> None:
    """
    Checks if all elements in both lists are of the same type.

    Args:
        list1 (list[Any]): The first list to check.
        lst2 (list[Any]): The second list to check.

    Raises:
        ValueError: If not all elements of list1 and list2 have the same type.
    """
    if len(list1) == 0:
        raise ValueError("The first list is empty.")

    list_type = type(list1[0])

    if not all(isinstance(entry, list_type) for entry in list1):
        raise ValueError("Not all elements in the first list are of the same type.")

    if not all(isinstance(entry, list_type) for entry in list2):
        raise ValueError(
            "Not all elements in the second list are of the same type as the first element of the first list."
        )
