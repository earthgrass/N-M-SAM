"""Frozen CIFAR-100 entry point for the existing SSAM2022-style ResNet18."""
from __future__ import annotations

from benchmark.model import BasicBlock, ResNet

CIFAR100_MODEL_ID = "ssam2022_cifar_resnet18_c100_v1"
CIFAR100_NUM_CLASSES = 100
CIFAR100_BLOCKS = (2, 2, 2, 2)


def build_cifar100_model() -> ResNet:
    """Build the 100-class variant without changing the CIFAR-10 factory."""
    return ResNet(BasicBlock, list(CIFAR100_BLOCKS), num_classes=CIFAR100_NUM_CLASSES)
