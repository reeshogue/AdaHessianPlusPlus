import torch
from math import pi
from optim import AdaHessian

def rosenbrock(tensor):
    x, y = tensor
    return (1-x) ** 2 + 1 * (y - x ** 2) ** 2

def rastrigin(tensor):
    x, y = tensor
    A = 10
    f = (
        A*2 
    + (x ** 2 - A * torch.cos(x * pi * 2))
    + (y ** 2 - A * torch.cos(y * pi * 2))
    )

    return f

def test_rosenbrock(optim):
    x = torch.Tensor([1.5, 1.5]).requires_grad_(True)
    optim = optim([x], 0.3)

    for i in range(10):
        optim.zero_grad()
        f = rosenbrock(x)
        f.backward(retain_graph=True, create_graph=True)
        optim.step()
        print(f.item())

test_rosenbrock(AdaHessian)
