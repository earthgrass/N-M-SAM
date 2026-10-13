"""Frozen augmentation protocol for the Phase-2A strong dense baseline."""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageOps

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


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class RandomCrop:
    def __init__(self, size: int, padding: int = 0):
        self.size = int(size)
        self.padding = int(padding)

    def __call__(self, image: Image.Image) -> Image.Image:
        padded = ImageOps.expand(image, border=self.padding, fill=0)
        extent = 2 * self.padding
        left = random.randint(0, extent)
        top = random.randint(0, extent)
        return padded.crop((left, top, left + self.size, top + self.size))


class RandomHorizontalFlip:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if random.random() < 0.5 else image


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1))


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


def build_strong_baseline_transforms():
    """Return ``train_aug_transform, deterministic_eval_transform``."""
    try:
        from torchvision import transforms
        compose = transforms.Compose
        random_crop = transforms.RandomCrop(32, padding=4)
        random_flip = transforms.RandomHorizontalFlip()
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    except Exception:
        compose = Compose
        random_crop = RandomCrop(32, padding=4)
        random_flip = RandomHorizontalFlip()
        to_tensor = ToTensor()
        normalize = Normalize(CIFAR10_MEAN, CIFAR10_STD)

    train_transform = compose([
        random_crop,
        random_flip,
        to_tensor,
        Cutout(n_holes=1, length=16),
        normalize,
    ])
    eval_transform = compose([
        ToTensor() if compose is Compose else transforms.ToTensor(),
        Normalize(CIFAR10_MEAN, CIFAR10_STD) if compose is Compose else transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


__all__ = ["Cutout", "build_strong_baseline_transforms"]
