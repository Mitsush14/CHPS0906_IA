import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import time

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups = 1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def init_data_CIFAR10():
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    return transform, train_loader, test_loader


def accuracy(output, target):
    _, predicted = torch.max(output, 1)  # Prend la classe avec la probabilité la plus élevée
    correct = (predicted == target).sum().item()  # Compte les bonnes prédictions
    total = target.size(0)  # Nombre total d'exemples dans le batch
    return correct / total  # Calcul de la précision

def train_and_test(model, train_loader, criterion, optimizer, epochs, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    start_time = time.time()
    train_model(model, train_loader, criterion, optimizer, epochs, device)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Entraînement : {execution_time:.2f} secondes")
    test_model(model, test_loader, device)

def train_model(model, train_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        model.train()
        train_acc, train_loss = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_acc += accuracy(outputs, labels)
            train_loss += loss.item()
        print(f"Précision du batch: {train_acc / len(train_loader):.4f}")
        print(f"Valeur de la perte: {train_loss / len(train_loader):.4f}")

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test d'accuracy : {accuracy:.4f}")

