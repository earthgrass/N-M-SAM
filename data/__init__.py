"""Data package."""
from .cifar10 import (
    CIFAR10Custom,
    build_cifar10,
    compute_normalization_stats,
    CIFAR10_MEAN,
    CIFAR10_STD,
)

__all__ = [
    "CIFAR10Custom",
    "build_cifar10",
    "compute_normalization_stats",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
]
