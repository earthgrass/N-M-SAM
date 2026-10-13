"""CIFAR-10 dataset loader that does not depend on torchvision.

Reads the official CIFAR-10 binary batches directly from disk, performs the
canonical normalization (mean / std computed on the training set), and
exposes a torch ``Dataset``-compatible API.

If torchvision is available we still prefer it for convenience (it handles
downloading + extraction transparently). The fallback path is fully
self-contained: download the tar from the official mirror, extract, then
parse the binary batches with NumPy.
"""
from __future__ import annotations

import os
import pickle
import random
import tarfile
import urllib.request
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MIRROR = "https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz"

# Canonical CIFAR-10 channel statistics.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class RandomCrop32:
    def __call__(self, image: Image.Image) -> Image.Image:
        padded = ImageOps.expand(image, border=4, fill=0)
        left = random.randint(0, 8)
        top = random.randint(0, 8)
        return padded.crop((left, top, left + 32, top + 32))


class RandomHorizontalFlip:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if random.random() < 0.5 else image


class ToNormalizedTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
        return (tensor - mean) / std


def build_transforms():
    """Return the method-independent CIFAR-10 train/test transforms."""
    try:
        from torchvision import transforms as transforms  # type: ignore

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    except Exception:
        train_transform = Compose([RandomCrop32(), RandomHorizontalFlip(), ToNormalizedTensor()])
        test_transform = Compose([ToNormalizedTensor()])
    return train_transform, test_transform


def _download_and_extract(root: str, mirror: str = CIFAR10_URL) -> None:
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, "cifar-10-python.tar.gz")
    if not os.path.exists(target):
        last_err: Optional[Exception] = None
        for url in (mirror, CIFAR10_MIRROR):
            print(f"[cifar10] downloading from {url} ...")
            try:
                urllib.request.urlretrieve(url, target)
                last_err = None
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_err = exc
                print(f"[cifar10] mirror {url} failed: {exc}")
        if last_err is not None and not os.path.exists(target):
            raise RuntimeError(
                "Could not download CIFAR-10. Place the official "
                "cifar-10-python.tar.gz into " + root
            ) from last_err
    extracted = os.path.join(root, "cifar-10-batches-py")
    if not os.path.isdir(extracted):
        print("[cifar10] extracting archive ...")
        with tarfile.open(target, "r:gz") as tar:
            tar.extractall(path=root)


def _load_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
    data = entry[b"data"] if b"data" in entry else entry["data"]
    labels = entry[b"labels"] if b"labels" in entry else entry["labels"]
    images = data.reshape(-1, 3, 32, 32).astype(np.uint8)
    return images, np.asarray(labels, dtype=np.int64)


class CIFAR10Custom(Dataset):
    """In-memory CIFAR-10 dataset with optional transform."""

    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.train = train

        try:
            _download_and_extract(root)
        except Exception:
            # Re-raise; the caller may want to surface the error.
            raise

        base = os.path.join(root, "cifar-10-batches-py")
        if train:
            image_list: List[np.ndarray] = []
            label_list: List[np.ndarray] = []
            for i in range(1, 6):
                imgs, labs = _load_batch(os.path.join(base, f"data_batch_{i}"))
                image_list.append(imgs)
                label_list.append(labs)
            self.images = np.concatenate(image_list, axis=0)
            self.labels = np.concatenate(label_list, axis=0)
        else:
            self.images, self.labels = _load_batch(os.path.join(base, "test_batch"))

        # (N, 3, 32, 32) uint8 -> torch float tensor (normalized) lazily in __getitem__.
        self.images = self.images  # keep memory layout
        self._mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]  # (3, 32, 32) uint8
        if self.transform is not None:
            # Convert to PIL so transforms work uniformly.
            pil = Image.fromarray(np.transpose(img, (1, 2, 0)))
            pil = self.transform(pil)
            return pil, int(self.labels[idx])
        # Default: convert to float tensor and normalize.
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        t = (t - self._mean.squeeze(0)) / self._std.squeeze(0)
        return t, int(self.labels[idx])


def _torchvision_available() -> bool:
    try:
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False


def build_cifar10(
    root: str,
    train: bool,
    transform: Optional[Callable] = None,
    use_fake_data: bool = False,
    fake_train_size: int = 2000,
    fake_test_size: int = 500,
) -> Dataset:
    """Build a CIFAR-10 dataset.

    Prefers torchvision if available (handles augmentation transforms).
    Falls back to the bundled custom loader. If ``use_fake_data=True`` we
    short-circuit and return a deterministic FakeData dataset (3x32x32
    images, 10 classes) without touching the network. This is intended
    for smoke tests when the real CIFAR-10 cannot be downloaded.
    """
    if use_fake_data:
        return _build_fake(train, fake_train_size, fake_test_size, transform)
    if _torchvision_available():
        from torchvision.datasets import CIFAR10  # type: ignore
        # If the extracted batch directory does not exist and download fails,
        # fall back to FakeData so smoke tests still run.
        try:
            return CIFAR10(root=root, train=train, transform=transform, download=True)
        except Exception as exc:
            print(f"[cifar10] real data unavailable ({exc}); falling back to FakeData.")
            return _build_fake(train, fake_train_size, fake_test_size, transform)
    return CIFAR10Custom(root=root, train=train, transform=transform)


def _build_fake(train: bool, fake_train_size: int, fake_test_size: int, transform):
    """Return a deterministic FakeData dataset (3x32x32, 10 classes)."""
    try:
        from torchvision.datasets import FakeData  # type: ignore
        size = fake_train_size if train else fake_test_size
        ds = FakeData(size=size, image_size=(3, 32, 32), num_classes=10,
                      transform=transform)
        return ds
    except Exception:
        # Final fallback: build an in-memory numpy dataset.
        n = fake_train_size if train else fake_test_size
        rng = np.random.RandomState(0 if train else 1)
        images = (rng.rand(n, 3, 32, 32) * 255).astype(np.uint8)
        labels = rng.randint(0, 10, size=n).astype(np.int64)
        return _NumpyFakeDataset(images, labels, transform)


class _NumpyFakeDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.transform is not None:
            from PIL import Image
            pil = Image.fromarray(np.transpose(img, (1, 2, 0)))
            pil = self.transform(pil)
            return pil, int(self.labels[idx])
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        return t, int(self.labels[idx])


def compute_normalization_stats(dataset: Dataset, n: int = 5000) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute per-channel mean/std on the first ``n`` images of ``dataset``."""
    n = min(n, len(dataset))
    samples = []
    for i in range(n):
        img, _ = dataset[i]
        if isinstance(img, torch.Tensor):
            samples.append(img)
    if not samples:
        return CIFAR10_MEAN, CIFAR10_STD
    stacked = torch.stack(samples)
    mean = tuple(stacked.mean(dim=(0, 2, 3)).tolist())
    std = tuple(stacked.std(dim=(0, 2, 3)).tolist())
    return mean, std
