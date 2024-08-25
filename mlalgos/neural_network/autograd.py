from typing import Callable

class Value:
    def __init__(self, data: float, children: tuple['Value', ...] = (), op: str = "") -> None:
        self._data = data
        self._prev: set['Value'] = set(children)
        self._op = op
        self._grad: float = 0
        self._backward: Callable = lambda: None

    @property
    def grad(self) -> float:
        return self._grad

    @grad.setter
    def grad(self, value: float) -> None:
        self._grad = value

    def __repr__(self) -> str:
        return f"Value(data={self._data})"

    def __add__(self, other: 'Value | float') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self._data + other._data, (self, other), "+")

        def _backward() -> None:
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: 'Value | float') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self._data * other._data, (self, other), "*")

        def _backward() -> None:
            self.grad = other.grad * out.grad
            other.grad = self.grad * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: 'Value | float') -> 'Value':
        return self + other

    def __rmul__(self, other: 'Value | float') -> 'Value':
        return self * other
