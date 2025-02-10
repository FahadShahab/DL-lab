import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Model Definitions
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(64, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 10, bias=True)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))


class ReducedFiltersCNN(nn.Module):
    def __init__(self):
        super(ReducedFiltersCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(32, 64, kernel_size=3),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 32, kernel_size=3),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(32, 10, bias=True),  # Reduced fully connected layer
            nn.ReLU(),
            nn.Linear(10, 10, bias=True)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Function to train and evaluate a model
def train_and_evaluate(model, train_loader, test_loader, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_labels, all_preds


# Train and evaluate all models
models = {
    "OriginalCNN": CNNClassifier(),
    "ReducedFiltersCNN": ReducedFiltersCNN()
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    accuracy, all_labels, all_preds = train_and_evaluate(model, train_loader, test_loader)
    num_params = count_parameters(model)
    results[name] = {"accuracy": accuracy, "num_params": num_params}

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"{name} Confusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

# Print results for all models
for name, result in results.items():
    print(f"{name}: Accuracy = {result['accuracy']:.4f}, Parameters = {result['num_params']}")


# Calculate percentage drop in parameters
original_params = results["OriginalCNN"]["num_params"]
for name, result in results.items():
    result["param_drop"] = 100 * (original_params - result["num_params"]) / original_params

# Extract data for plotting
names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in names]
param_drops = [results[name]["param_drop"] for name in names]

# Plot Percentage Drop in Parameters vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(param_drops, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Percentage Drop in Parameters')
plt.ylabel('Accuracy')
plt.title('Percentage Drop in Parameters vs Accuracy')
plt.grid(True)
plt.show()
