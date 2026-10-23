"""Fail-closed real data; epoch/sample keyed augmentation and permutations."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..repro import project_path, seed_for

MEAN = torch.tensor((.4914, .4822, .4465)).view(3, 1, 1)
STD = torch.tensor((.2470, .2435, .2616)).view(3, 1, 1)


class Images(Dataset):
    def __init__(self, images, labels, indices, seed, augment=False, cutout=0):
        self.images, self.labels = images, labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seed, self.augment, self.cutout = seed, augment, cutout
        self.epoch = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        x = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255
        if self.augment:
            gen = torch.Generator().manual_seed(seed_for(self.seed, "augment", self.epoch, index))
            top, left = torch.randint(0, 9, (2,), generator=gen).tolist()
            x = F.pad(x, (4, 4, 4, 4))[:, top:top+32, left:left+32]
            if torch.rand((), generator=gen).item() < .5:
                x = x.flip(2)
            if self.cutout:
                y, z = torch.randint(0, 32, (2,), generator=gen).tolist()
                half = self.cutout // 2
                x[:, max(0, y-half):min(32, y+self.cutout-half),
                  max(0, z-half):min(32, z+self.cutout-half)] = 0
        return (x - MEAN) / STD, int(self.labels[index]), index


def build_data(config, root: Path):
    d, seed = config["data"], config["seed"]
    if d["kind"] == "cifar10":
        from torchvision.datasets import CIFAR10
        location = project_path(root, d["root"])
        # torchvision verifies official MD5s; failure is fatal, never FakeData.
        train = CIFAR10(str(location), train=True, download=d["download"])
        test = CIFAR10(str(location), train=False, download=d["download"])
        images, labels = train.data, np.asarray(train.targets)
        test_images, test_labels = test.data, np.asarray(test.targets)
        identity = "official-cifar10-md5-verified"
    else:
        gen = np.random.default_rng(seed_for(seed, "synthetic"))
        count, test_count = d["synthetic_train"], d["synthetic_test"]
        if not isinstance(count, int) or not isinstance(test_count, int) or min(count, test_count) < 2:
            raise ValueError("Explicit synthetic sizes required")
        images = gen.integers(0, 256, (count, 32, 32, 3), dtype=np.uint8)
        labels = gen.integers(0, 10, count)
        test_images = gen.integers(0, 256, (test_count, 32, 32, 3), dtype=np.uint8)
        test_labels = gen.integers(0, 10, test_count)
        identity = "SYNTHETIC-SMOKE-ONLY"
    size = d["validation_size"]
    if not isinstance(size, int) or not 0 < size < len(images)-1:
        raise ValueError("Explicit nonempty held-out validation split required")
    order = np.random.default_rng(seed_for(seed, "split")).permutation(len(images))
    val_indices, train_indices = order[:size], order[size:]
    kwargs = dict(images=images, labels=labels, seed=seed)
    datasets = {"train": Images(**kwargs, indices=train_indices, augment=d["augment"], cutout=d["cutout"]),
                "validation": Images(**kwargs, indices=val_indices),
                "test": Images(test_images, test_labels, np.arange(len(test_images)), seed)}
    manifest = {"identity": identity, "train_indices": train_indices.tolist(),
                "validation_indices": val_indices.tolist(), "test_size": len(test_images),
                "seed": seed, "augmentation": {"enabled": d["augment"], "cutout": d["cutout"]}}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return datasets, manifest


def loader(dataset, batch_size, seed, epoch=0, start_batch=0, workers=0, shuffle=False):
    dataset.epoch = epoch
    if shuffle:
        gen = torch.Generator().manual_seed(seed_for(seed, "order", epoch))
        indices = torch.randperm(len(dataset), generator=gen).tolist()
    else:
        indices = list(range(len(dataset)))
    batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    # Dedicated loader stream prevents worker creation consuming model RNG.
    gen = torch.Generator().manual_seed(seed_for(seed, "loader", epoch))
    return DataLoader(dataset, batch_sampler=batches[start_batch:], num_workers=workers,
                      generator=gen, persistent_workers=False)
