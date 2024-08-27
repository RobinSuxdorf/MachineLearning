import math
from mlalgos import ArrayLike
from mlalgos.neural_network import Value


def tanh(x: Value | ArrayLike) -> Value | ArrayLike:
    """
    Applies the hyperbolic tangent (tanh) activation function to the input.

    The tanh function is defined as:
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    It maps the input to the range [-1, 1].

    Args:
        x (Value | ArrayLike): The input to the tanh function, which can be a single `Value` object or a list of `Value` objects.

    Returns:
        Value | ArrayLike: The result of applying the tanh function to the input. Returns a `Value` object if the input is a single `Value`, or a list of `Value` objects if the input is a list.
    """

    def _single_tanh(v: Value) -> Value:
        t = (math.exp(2 * v.data) - 1) / (math.exp(2 * v.data) + 1)
        out = Value(t, (v,), "tanh")

        def _backward() -> None:
            v.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    if isinstance(x, list):
        return [_single_tanh(v) for v in x]
    else:
        return _single_tanh(x)


def relu(x: Value | ArrayLike) -> Value | ArrayLike:
    """
    Applies the rectified linear unit (ReLU) activation function to the input.

    The ReLU function is defined as:
    ReLU(x) = max(0, x)

    It sets all negative values to zero.

    Args:
        x (Value | ArrayLike): The input to the ReLU function, which can be a single `Value` object or a list of `Value` objects.

    Returns:
        Value | ArrayLike: The result of applying the ReLU function to the input. Returns a `Value` object if the input is a single `Value`, or a list of `Value` objects if the input is a list.
    """

    def _single_relu(v: Value) -> Value:
        out = Value(max(v.data, 0), (v,), "relu")

        def _backward() -> None:
            v.grad += (v.data > 0) * out.grad

        out._backward = _backward
        return out

    if isinstance(x, list):
        return [_single_relu(v) for v in x]
    else:
        return _single_relu(x)
