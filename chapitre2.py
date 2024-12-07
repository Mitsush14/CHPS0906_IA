import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Définir le modèle
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MyClass(nn.Module):
    def __init__(self, hidden_dim=32, out=10):
        super(MyClass, self).__init__()
        # Utiliser ConvBNReLU pour chaque couche de convolution
        self.conv1 = ConvBNReLU(3, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.fc = nn.Linear(hidden_dim * 32 * 32, out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        #applatissement des données
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
def accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total
# Définir les transformations et charger les données
transform = transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

# Initialiser le modèle et l'optimiseur
model = MyClass(hidden_dim=32, out=10)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Entraînement

print("Début de l'entrainement")
for epoch in range(5):
    print('Epoch {}/{}'.format(epoch+1, 5))
    model.train()
    train_acc, train_loss = 0, 0 #reset du pourcentage d'accuracy et de perte à chaque epoch pour vérifier si elle a augmenté ou baissé
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_acc += accuracy(outputs, labels)
        train_loss += loss.item()

    print(f"Précision du batch: {train_acc/len(train_loader):.4f}")
    print(f"Valeur de la perte: {train_loss/len(train_loader):.4f}")
