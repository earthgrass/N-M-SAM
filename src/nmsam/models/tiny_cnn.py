from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Tanh(),
                                 nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Tanh(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))

    def forward(self, x):
        return self.net(x)
