import torch

x = torch.tensor(2.0, requires_grad=True)

analytical_grad = 32*x**3 + 9*x**2 + 14*x + 6

print(f'Analytical gradient : {analytical_grad}')

y=8*x**4+3*x**3+7*x**2+6*x+3
y.backward()
computed_grad = x.grad

print(f'Computed graient : {computed_grad}')