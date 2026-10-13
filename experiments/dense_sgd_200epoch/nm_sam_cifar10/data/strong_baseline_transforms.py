"""Frozen augmentation protocol for the Phase-2A strong dense baseline."""
from __future__ import annotations

import torch

from .cifar10 import CIFAR10_MEAN, CIFAR10_STD


class Cutout:
    """Mask square regions in a CHW tensor.

    Phase-2A uses the standard CIFAR choice of one 16x16 hole.  The transform
    is intentionally applied after ``ToTensor`` and before ``Normalize`` so a
    fill value of zero corresponds to a black pixel in the original [0, 1]
    tensor domain.
    """

    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        if n_holes <= 0 or length <= 0:
            raise ValueError("n_holes and length must be positive")
        self.n_holes = int(n_holes)
        self.length = int(length)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise TypeError("Cutout expects a CHW torch.Tensor")
        height, width = image.shape[-2:]
        output = image.clone()
        for _ in range(self.n_holes):
            center_y = int(torch.randint(0, height, (1,)).item())
            center_x = int(torch.randint(0, width, (1,)).item())
            half = self.length // 2
            y1 = max(0, center_y - half)
            y2 = min(height, center_y + self.length - half)
            x1 = max(0, center_x - half)
            x2 = min(width, center_x + self.length - half)
            output[:, y1:y2, x1:x2] = 0.0
        return output

    def __repr__(self) -> str:
        return f"Cutout(n_holes={self.n_holes}, length={self.length})"


def build_strong_baseline_transforms():
    """Return ``train_aug_transform, deterministic_eval_transform``."""
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


__all__ = ["Cutout", "build_strong_baseline_transforms"]
