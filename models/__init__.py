"""Models package."""
import torch
import torch.nn as nn

from .resnet_cifar import ResNetCIFAR, resnet18_cifar, BasicBlock
from .simple_cnn import SimpleCNN

__all__ = ["ResNetCIFAR", "resnet18_cifar", "BasicBlock", "SimpleCNN"]


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    name = name.lower()
    if name in ("resnet18_cifar", "resnet18", "resnet18-cifar"):
        return resnet18_cifar(num_classes=num_classes)
    if name in ("simplecnn", "simple_cnn", "simple"):
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'")
