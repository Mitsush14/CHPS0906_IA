from library import *
class Bottleneck(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, kernel_size):
        super(Bottleneck, self).__init__()
        # Utiliser ConvBNReLU pour chaque couche de convolution
        self.reduire = ConvBNReLU(in_dim, hidden_dim, kernel_size=kernel_size)
        self.maintenir = ConvBNReLU(hidden_dim, hidden_dim, kernel_size=kernel_size)
        self.revenir = ConvBNReLU(hidden_dim, out_dim, kernel_size=kernel_size)
        self.residual = in_dim == out_dim

        if not self.residual:
            self.residual_connections = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        x = self.reduire(x)
        x = self.maintenir(x)
        x = self.revenir(x)
        if self.residual:
            x = x + identity
        else:
            x = x + self.residual_connections(identity)

        return torch.relu(x)

class StackedBottleneck(nn.Module):
    def __init__(self):
        super(StackedBottleneck, self).__init__()
        self.bottleneck1 = Bottleneck(3,32,64, 3)
        self.bottleneck2 = Bottleneck(64, 64, 128, 5)
        self.bottleneck3 = Bottleneck(128, 128, 256, 1)
        self.fc = nn.Linear(256 * 32 * 32, 10)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    # initialisation des données pour l'IA
    transform, train_loader, test_loader = init_data_CIFAR10()
    model = StackedBottleneck()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #entrainement et test de l'ia
    train_and_test(model, train_loader, criterion, optimizer, 5, test_loader)

if __name__ == '__main__':
    main()