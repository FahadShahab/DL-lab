import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.w * x + self.b

class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

model = RegressionModel()

optimizer = optim.SGD(model.parameters(), lr=0.001)

dataset = RegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

criterion = nn.MSELoss()

epochs = 100
loss_list = []

for epoch in range(epochs):
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        y_pred = model(inputs)

        loss = criterion(y_pred, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    loss_list.append(total_loss / len(dataloader))

    print(f'Epoch [{epoch + 1}/{epochs}], w: {model.w.item()}, b: {model.b.item()}, Loss: {total_loss / len(dataloader)}')

plt.plot(range(epochs), loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()

print(f'Final model parameters: w = {model.w.item()}, b = {model.b.item()}')
