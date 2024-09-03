from typing import Callable
import math


class Value:
    def __init__(
        self, data: float, children: tuple["Value", ...] = (), op: str = ""
    ) -> None:
        """
        Initialize a Value object to represent a node in a computational graph.

        Args:
            data (float): The scalar value of the node.
            children (tuple[Value, ...]): A tuple of child nodes that are the inputs to this node.
            op (str): The operation that created this node (e.g., '+', '*', '**').
        """
        self._data = data
        self._prev: set["Value"] = set(children)
        self._op = op
        self._grad: float = 0.0
        self._backward: Callable[[], None] = self._default_backward

    def _default_backward(self) -> None:
        """
        Default backward pass method used for initialization.
        """
        pass

    @property
    def data(self) -> float:
        """
        Get the scalar value of this node.

        Returns:
            float: The scalar value of this node.
        """
        return self._data

    @data.setter
    def data(self, value: float) -> None:
        """
        Set the data value of this node.

        Args:
            value (float): The new data value to set.
        """
        self._data = value

    @property
    def grad(self) -> float:
        """
        Get the gradient of this node.

        Returns:
            float: The gradient of this node with respect to some scalar value.
        """
        return self._grad

    @grad.setter
    def grad(self, value: float) -> None:
        """
        Set the gradient of this node.

        Args:
            value (float): The new gradient value to set.
        """
        self._grad = value

    def __repr__(self) -> str:
        """
        Return a string representation of the Value object.

        Returns:
            str: A string representation of the object.
        """

        return f"Value(data={self.data})"

    def __add__(self, other: "Value | float") -> "Value":
        """
        Add this Value to another Value or float.

        Args:
            other (Value | float): The other value to add.

        Returns:
            Value: A new Value object representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        """
        Multiply this Value with another Value or float.

        Args:
            other (Value | float): The other value to multiply.

        Returns:
            Value: A new Value object representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> "Value":
        """
        Raise this Value to the power of exponent.

        Args:
            exponent (float): The exponent to raise the value to.

        Returns:
            Value: A new Value object representing the power operation.
        """
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: "Value | float") -> "Value":
        """
        Define the reverse addition operation.

        Args:
            other (Value or float): The value to add.

        Returns:
            Value: A new Value object representing the sum.
        """
        return self + other

    def __rmul__(self, other: "Value | float") -> "Value":
        """
        Define the reverse multiplication operation.

        Args:
            other (Value or float): The value to multiply.

        Returns:
            Value: A new Value object representing the product.
        """
        return self * other

    def __neg__(self) -> "Value":
        """
        Define the negation operation.

        Returns:
            Value: A new Value object representing the negation of this Value.
        """
        return self * -1

    def __sub__(self, other: "Value | float") -> "Value":
        """
        Define the subtraction operation.

        Args:
            other (Value or float): The value to subtract.

        Returns:
            Value: A new Value object representing the difference.
        """
        return self + (-other)

    def __truediv__(self, other: "Value | float") -> "Value":
        """
        Define the true division operation.

        Args:
            other (Value or float): The value to divide by.

        Returns:
            Value: A new Value object representing the division result.
        """
        return self * other**-1

    def backward(self) -> None:
        """
        Perform backpropagation to compute the gradients of all nodes in the computational graph.
        """

        def _topological_sort(root: Value) -> list[Value]:
            topo: list[Value] = []
            visited: set[Value] = set()

            def _build_topo(v: Value) -> None:
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        _build_topo(child)
                    topo.append(v)

            _build_topo(root)

            return topo

        topo = _topological_sort(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def log(self) -> "Value":
        """
        Applies the natural logarithm to the value.
        """
        out = Value(math.log(self.data), (self, ), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward

        return out
