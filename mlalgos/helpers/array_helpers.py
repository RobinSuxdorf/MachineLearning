from typing import Any
from mlalgos import ArrayLike
import numpy as np


def check_array(*args: ArrayLike) -> list[np.ndarray]:
    """
    Convert input lists to numpy arrays and validate all inputs.

    This function accepts one or more inputs, each of which can be either a numpy array or a list. 
    If an input is a list, it is converted to a numpy array. The function then checks that all 
    inputs are numpy arrays (either originally or after conversion). 

    Parameters:
        *args (ArrayLike): One or more inputs, each of which can be either a numpy array or a list.

    Returns:
        np.ndarray | list[np.ndarray]: 
            If a single argument is provided, returns the corresponding numpy array. 
            If multiple arguments are provided, returns a list of numpy arrays.
    """
    result = [np.array(X) if isinstance(X, list) else X for X in args]

    if any(not isinstance(X, np.ndarray) for X in result):
        raise ValueError("All inputs must be numpy arrays or lists.")

    return result[0] if len(result) == 1 else result


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
