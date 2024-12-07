import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms


class Network(nn.Module):
    def __init__(self, size_out):
        super().__init__()
        # définition des couches linéaires
        self.a = nn.Linear(3 * 32 * 32, size_out)
        self.b = nn.Linear(size_out, 24)
        self.c = nn.Linear(24, 10)

    def forward(self, x):
        return self.c(self.b(self.a(x.view(x.size(0), -1))))


# Fonction accuracy pour calculer le pourcentage de prédictions correctes
def accuracy(output, target):
    # Récupération des prédictions (classe avec la probabilité maximale)
    _, predicted = torch.max(output, 1)
    # Calcul de la précision
    correct = (predicted == target).sum().item()
    return correct / len(target)


# Fonction d'entraînement et de validation sur une époque complète
def fit_one_cycle(model, train_loader, val_loader, optimizer, criterion):
    # Mode entraînement
    model.train()
    train_loss, train_accuracy = 0, 0

    # Boucle sur les batches de données d'entraînement
    for images, labels in train_loader:
        # Forward pass
        preds = model(images)
        loss = criterion(preds, labels)

        # Backward pass et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcul de la loss et de l'accuracy
        train_loss += loss.item()
        train_accuracy += accuracy(preds, labels)

    # Moyenne de la loss et de l'accuracy pour l'ensemble d'entraînement
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    # Mode évaluation
    model.eval()
    val_loss, val_accuracy = 0, 0

    # Boucle sur les batches de données de validation
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images)
            loss = criterion(preds, labels)

            # Calcul de la loss et de l'accuracy
            val_loss += loss.item()
            val_accuracy += accuracy(preds, labels)

    # Moyenne de la loss et de l'accuracy pour l'ensemble de validation
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)

    return train_loss, train_accuracy, val_loss, val_accuracy


# Préparation des données CIFAR-10 avec transformation en tenseur
trans = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32)

# Création et passage des données dans le modèle
model = Network(10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Boucle d'entraînement pour plusieurs époques
epochs = 5  # Nombre d'époques pour l'entraînement

for epoch in range(epochs):
    train_loss, train_accuracy, val_loss, val_accuracy = fit_one_cycle(model, train_loader, val_loader, optimizer,
                                                                       criterion)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
