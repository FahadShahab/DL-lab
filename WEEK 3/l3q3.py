import torch
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self):
        self.w = torch.tensor([1.0], requires_grad=True)
        self.b = torch.tensor([1.0], requires_grad=True)

    def forward(self, x):
        return self.w * x + self.b

    def update(self, learning_rate):
        self.w -= learning_rate * self.w.grad
        self.b -= learning_rate * self.b.grad

    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

    def criterion(self, y, yp):
        return (yp - y) ** 2

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
learning_rate = torch.tensor(0.001)

model = RegressionModel()

epochs = 100
loss_list = []

for epoch in range(epochs):
    loss = 0.0
    for j in range(len(x)):
        y_p = model.forward(x[j])
        loss += model.criterion(y[j], y_p)

    loss = loss/len(x)

    loss_list.append(loss.item())

    loss.backward()

    with torch.no_grad():
        model.update(learning_rate)

    model.reset_grad()

    print(f'The parameters are w : {model.w.item()},  b : {model.b.item()},  loss : {loss.item()}')

plt.plot(range(epochs), loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
