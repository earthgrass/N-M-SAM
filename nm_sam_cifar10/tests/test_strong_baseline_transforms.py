"""Frozen Phase-2A augmentation protocol tests."""
from __future__ import annotations

import torch

from nm_sam_cifar10.data.strong_baseline_transforms import (
    Cutout,
    build_strong_baseline_transforms,
)


def test_cutout_masks_tensor_without_mutating_input():
    torch.manual_seed(0)
    image = torch.ones(3, 32, 32)
    output = Cutout(n_holes=1, length=16)(image)
    assert torch.all(image == 1)
    assert output.shape == image.shape
    assert int((output == 0).sum().item()) > 0
    assert int((output == 0).sum().item()) <= 3 * 16 * 16


def test_frozen_transform_order_and_eval_determinism():
    train_transform, eval_transform = build_strong_baseline_transforms()
    train_names = [type(transform).__name__ for transform in train_transform.transforms]
    eval_names = [type(transform).__name__ for transform in eval_transform.transforms]
    assert train_names == [
        "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Cutout", "Normalize"
    ]
    assert eval_names == ["ToTensor", "Normalize"]
