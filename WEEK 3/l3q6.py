import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

X = np.array([[3, 8], [4, 5], [5, 7], [6, 3], [2, 1]], dtype=np.float32)
Y = np.array([-3.7, 3.5, 2.5, 11.5, 5.7], dtype=np.float32)

X_train = torch.tensor(X)
Y_train = torch.tensor(Y).view(-1, 1)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    model.train()
    Y_pred = model(X_train)

    loss = criterion(Y_pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
test_input = torch.tensor([[3, 2]], dtype=torch.float32)
predicted_output = model(test_input).item()

print(f"Predicted Y for X1=3, X2=2: {predicted_output:.4f}")