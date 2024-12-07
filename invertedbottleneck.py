from library import *
import time

class InvertedBottleneck(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, kernel_size, stride = 1):
        super(InvertedBottleneck, self).__init__()
        self.expand = ConvBNReLU(in_dim, hidden_dim, kernel_size=1)
        self.depthwise = ConvBNReLU(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim)
        self.reduce = ConvBNReLU(hidden_dim, out_dim, kernel_size=1)
        self.residual = in_dim == out_dim
        if not self.residual:
            self.residual_connections = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.reduce(x)
        if self.residual:
            x = x + identity
        else:
            x = x + self.residual_connections(identity)
        return x

class StackedInvertedBottleneck(nn.Module):
    def __init__(self):
        super(StackedInvertedBottleneck, self).__init__()
        self.bottleneck1 = InvertedBottleneck(3, 32, 16, kernel_size=3, stride=1)
        self.bottleneck2 = InvertedBottleneck(16, 64, 32, kernel_size=5, stride=1)
        self.bottleneck3 = InvertedBottleneck(32, 128, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        return self.fc(x)

def main():
    #initialisation des données pour l'IA
    transform, train_loader, test_loader = init_data_CIFAR10()
    model = StackedInvertedBottleneck()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # entrainement et test de l'ia
    train_and_test(model, train_loader, criterion, optimizer, 10, test_loader)


if __name__ == '__main__':
    main()