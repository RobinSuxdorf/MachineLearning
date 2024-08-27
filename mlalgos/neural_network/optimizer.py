from abc import ABC, abstractmethod
from collections.abc import Iterable
import math
from mlalgos.neural_network import Value


class Optimizer(ABC):
    """
    Base class for all optimizers. Optimizers adjust the parameters of a model
    to minimize the loss function by computing the gradients of the parameters.
    """

    def __init__(self, parameters: Iterable[Value], lr: float) -> None:
        """
        Initialize the optimizer.

        Args:
            parameters (Iterable[Value]): An iterable yielding `Value` objects which are the parameters to optimize.
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
        """
        Updates the parameters by performing a Stochastic Gradient Descent optimization step.
        """
        for p in self._parameters:
            p.data -= self._lr * p.grad


class Adam(Optimizer):
    def __init__(self, parameters: Iterable[Value], lr: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        """
        Initializes the Adam optimizer.

        Args:
            parameters (Generator[Value, None, None]): An iterable yielding `Value` objects which are the parameters to optimize.
            lr (float): Learning rate to scale gradients during optimization.
            beta1 (float): Exponential decay rate for the first moment estimates. Default is 0.9.
            beta2 (float): Exponential decay rate for the second moment estimates. Default is 0.999.
            epsilon (float): Small constant to prevent division by zero. Default is 1e-8.
        """
        super().__init__(parameters, lr)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._m = {p: 0.0 for p in self._parameters}
        self._v = {p: 0.0 for p in self._parameters}
        self._t = 0

    def step(self) -> None:
        """
        Updates the parameters by performing an Adam optimization step.
        """
        self._t += 1
        for p in self._parameters:
            self._m[p] = self._beta1 * self._m[p] + (1 - self._beta1) * p.grad
            self._v[p] = self._beta2 * self._v[p] + (1 - self._beta2) * (p.grad ** 2)

            m_hat = self._m[p] / (1 - self._beta1 ** self._t)
            v_hat = self._v[p] / (1 - self._beta2 ** self._t)

            p.data -= self._lr * m_hat / (math.sqrt(v_hat) + self._epsilon)
