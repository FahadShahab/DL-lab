import torch
import math

x = torch.tensor(2.0, requires_grad=True)

f = torch.exp(-x ** 2 - 2 * x - torch.sin(x))

f.backward()

print(f'Computed gradient : {x.grad}')


def analytical():
    analytical_ans = torch.exp(-x ** 2 - 2 * x - torch.sin(x)) * (-2 * x - 2 - torch.cos(x))
    print(f'Analytical gradient : {analytical_ans}')


analytical()
