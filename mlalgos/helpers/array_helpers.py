from mlalgos import ArrayLike
import numpy as np


def check_array(*args: ArrayLike) -> list[np.ndarray]:
    """
    Convert input lists to numpy arrays and validate all inputs.

    This function accepts one or more inputs, each of which can be either a numpy array or a list.
    If an input is a list, it is converted to a numpy array. The function then checks that all
    inputs are numpy arrays (either originally or after conversion).

    Args:
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


def check_length(array1: ArrayLike, array2: ArrayLike) -> None:
    """
    Checks if the two arrays (or lists) have the same length.

    Args:
        array1 (ArrayLike): The first array or list to check.
        array2 (ArrayLike): The second array or list to check.

    Raises:
        ValueError: If the arrays or lists do not have the same length.
    """
    array1_len = len(array1)
    array2_len = len(array2)

    if array1_len != array2_len:
        raise ValueError("The inputs do not have the same length.")


def check_type(array1: ArrayLike, array2: ArrayLike) -> None:
    """
    Checks if all elements in both arrays (or lists) are of the same type.

    Args:
        array1 (ArrayLike): The first array or list to check.
        array2 (ArrayLike): The second array or list to check.

    Raises:
        ValueError: If not all elements of array1 and array2 have the same type.
    """
    if isinstance(array1, list):
        array1 = np.array(array1, dtype=object)
    if isinstance(array2, list):
        array2 = np.array(array2, dtype=object)

    if array1.size == 0:
        raise ValueError("The first array is empty.")

    first_elem_type = type(array1.flat[0])

    if not all(isinstance(entry, first_elem_type) for entry in array1.flat):
        raise ValueError("Not all elements in the first array are of the same type.")

    if not all(isinstance(entry, first_elem_type) for entry in array2.flat):
        raise ValueError(
            "Not all elements in the second array are of the same type as the first element of the first array."
        )
