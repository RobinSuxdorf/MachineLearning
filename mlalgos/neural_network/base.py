from abc import ABC, abstractmethod
from typing import Generator
import random
from mlalgos import ArrayLike
from mlalgos.neural_network import Value


class Module(ABC):
    def parameters(self) -> Generator[Value, None, None]:
        """
        Recursively yields all the `Value` objects representing parameters
        of this module and its submodules.

        Yields:
            Value: A generator yielding `Value` objects which are the parameters
            of this module or its submodules.
        """
        for attr in self.__dict__.values():
            if isinstance(attr, Value):
                yield attr
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Value):
                        yield item
                    elif isinstance(item, Module):
                        yield from item.parameters()
            elif isinstance(attr, Module):
                yield from attr.parameters()

    def __call__(self, x: ArrayLike) -> Value | ArrayLike:
        """
        Makes an instance callable and delegates the call to the `forward` method.

        Args:
            x (ArrayLike): The input data to process.

        Returns:
            ArrayLike: The output after processing the input through the forward method.
        """
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ArrayLike) -> Value | ArrayLike:
        """
        Defines the computation performed at every call.

        Args:
            x (ArrayLike): The input data to process.

        Returns:
            ArrayLike: The output data after applying the module's computation.
        """
        pass

class Neuron(Module):
    def __init__(self, in_features: int) -> None:
        """
        Initialize a Neuron.

        Args:
            in_features (int): Number of input features.
        """
        self._w = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
        self._b = Value(random.uniform(-1, 1))

    def forward(self, x: ArrayLike) -> Value:
        if len(x) != len(self._w):
            raise ValueError(f"Expected input of length {len(self._w)}, but got {len(x)}.")

        out = sum((wi * xi for wi, xi in zip(self._w, x)), self._b)
        return out
