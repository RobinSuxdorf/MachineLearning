from mlalgos import ArrayLike
from mlalgos.neural_network import Module, Neuron


class Linear(Module):
    """
    Initialize a Linear layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        self._neurons = [Neuron(in_features) for _ in range(out_features)]

    def forward(self, x: ArrayLike) -> ArrayLike:
        outs = [n(x) for n in self._neurons]
        return outs
