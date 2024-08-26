from typing import Callable

class Value:
    def __init__(self, data: float, children: tuple['Value', ...] = (), op: str = "") -> None:
        self._data = data
        self._prev: set['Value'] = set(children)
        self._op = op
        self._grad: float = 0
        self._backward: Callable = lambda: None

    @property
    def data(self) -> float:
        return self._data

    @property
    def grad(self) -> float:
        return self._grad

    @grad.setter
    def grad(self, value: float) -> None:
        self._grad = value

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: 'Value | float') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: 'Value | float') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> 'Value':
        out = Value(self.data ** other, (self, ), f"**{other}")

        def _backward() -> None:
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out.backward = _backward
        return out

    def backward(self) -> None:
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

        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other: 'Value | float') -> 'Value':
        return self + other

    def __rmul__(self, other: 'Value | float') -> 'Value':
        return self * other

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: 'Value | float') -> 'Value':
        return self + (-other)

    def __truediv__(self, other: 'Value | float') -> 'Value':
        return self * other**-1
