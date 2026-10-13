from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from .cifar_archive import prepare_archive
from .transforms import cifar_transform


def fixed_split(size: int = 50000, validation_size: int = 5000, seed: int = 0):
    if not 0 < validation_size < size:
        raise ValueError("invalid split sizes")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).numpy().astype("<i8")
    train, validation = order[validation_size:], order[:validation_size]
    return train, validation, hashlib.sha256(order.tobytes()).hexdigest()


def cifar_loaders(archive_path: str | Path, cache_dir: str | Path, *, batch_size=4, seed=0):
    root, archive_hash = prepare_archive(archive_path, cache_dir)
    train = CIFAR10(str(root), train=True, transform=cifar_transform(True), download=False)
    eval_train = CIFAR10(str(root), train=True, transform=cifar_transform(False), download=False)
    test = CIFAR10(str(root), train=False, transform=cifar_transform(False), download=False)
    if len(train) != 50000 or len(test) != 10000 or set(train.targets) != set(range(10)):
        raise ValueError("unexpected CIFAR counts/classes")
    core, val, split_hash = fixed_split(seed=seed)
    def loader(dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
    return {"train": loader(Subset(train, core)), "validation": loader(Subset(eval_train, val)),
            "test": loader(test), "calibration": loader(Subset(eval_train, core)),
            "provenance": {"archive_sha256": archive_hash, "split_sha256": split_hash,
                           "train_count": len(train), "test_count": len(test),
                           "train_core": len(core), "validation": len(val), "classes": 10}}
