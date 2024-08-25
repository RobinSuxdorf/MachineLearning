class Value:
    def __init__(self, data: float) -> None:
        self._data = data

    def __repr__(self) -> str:
        return f"Value(data={self._data})"

    def __add__(self, other: 'Value | float') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self._data + other._data)
        return out

    def __mul__(self, other: 'Value | float') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self._data * other._data)
        return out

    def __radd__(self, other: 'Value | float') -> 'Value':
        return self + other

    def __rmul__(self, other: 'Value | float') -> 'Value':
        return self * other
