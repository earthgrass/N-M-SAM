"""Self-contained CIFAR ResNet18 and an explicitly smoke-only tiny model."""
import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.shortcut = nn.Sequential()
        if stride != 1 or incoming != outgoing:
            self.shortcut = nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        width = 64
        for index, target in enumerate((64, 128, 256, 512), 1):
            stride = 1 if index == 1 else 2
            setattr(self, f"layer{index}", nn.Sequential(BasicBlock(width, target, stride), BasicBlock(target, target)))
            width = target
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for index in range(1, 5):
            x = getattr(self, f"layer{index}")(x)
        return self.linear(F.adaptive_avg_pool2d(x, 1).flatten(1))


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(self.conv(x))), 1).flatten(1))


def build_model(name):
    if name == "resnet18":
        return ResNet18()
    if name == "tiny":
        return TinyNet()
    raise ValueError(name)
