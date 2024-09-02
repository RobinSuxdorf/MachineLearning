# import torch
# from mlalgos.neural_network import Value, relu


# def test_autograd() -> None:
#     x = Value(-5.0)
#     z = 0.5 * x**2 - 2 * x + 5
#     q = relu(z) + z * x
#     h = relu(z * z + 1)
#     y = z * h + q * x + z**2
#     y.backward()

#     x_ag, y_ag = x, y

#     x = torch.Tensor([-5.0]).double()
#     x.requires_grad = True
#     z = 0.5 * x**2 - 2 * x + 5
#     q = z.relu() + z * x
#     h = (z * z + 1).relu()
#     y = z * h + q * x + z**2
#     y.backward()
#     x_pt, y_pt = x, y

#     assert y_ag.data == y_pt.data.item()
#     assert x_ag.grad == x_pt.grad.item()
