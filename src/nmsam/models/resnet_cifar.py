"""CIFAR ResNet18: 3x3 stride-one stem, no initial maxpool."""
from torch import nn
from torchvision.models import resnet18


def resnet18_cifar():
    model=resnet18(weights=None,num_classes=10)
    model.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity()
    return model
