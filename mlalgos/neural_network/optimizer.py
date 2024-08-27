from abc import ABC, abstractmethod
from typing import Generator
from mlalgos.neural_network import Value

class Optimizer(ABC):
    """
    Base class for all optimizers. Optimizers adjust the parameters of a model 
    to minimize the loss function by computing the gradients of the parameters.
    """

    def __init__(self, parameters: Generator[Value, None, None], lr: float) -> None:
        """
        Initialize the optimizer.

        Args:
            parameters (Generator[Value, None, None]): A generator yielding `Value` objects which are the parameters to optimize.
            lr (float): Learning rate to scale gradients during optimization.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self._parameters = list(parameters)
        self._lr = lr

    def zero_grad(self) -> None:
        """
        Resets the gradients of all optimized parameters to zero. This method is 
        typically called before the backpropagation step in a training loop.
        """
        for p in self._parameters:
            p.grad = 0.0

    @abstractmethod
    def step(self) -> None:
        """
        Method to update the parameters of the model based on their gradients.
        """
        pass


class SGD(Optimizer):
    """
    Implementation of Stochastic Gradient Descent.
    """
    def step(self) -> None:
        for p in self._parameters:
            p.data -= self._lr * p.grad
