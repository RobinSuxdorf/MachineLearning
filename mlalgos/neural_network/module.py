from abc import ABC, abstractmethod
from typing import Generator
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

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """
        Makes an instance callable and delegates the call to the `forward` method.

        Args:
            x (ArrayLike): The input data to process.

        Returns:
            ArrayLike: The output after processing the input through the forward method.
        """
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ArrayLike) -> ArrayLike:
        """
        Defines the computation performed at every call.

        This method should be overridden by all subclasses to define the
        computation that the module should perform when applied to input data.

        Args:
            x (ArrayLike): The input data to process.

        Returns:
            ArrayLike: The output data after applying the module's computation.
        """
        pass
