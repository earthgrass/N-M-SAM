"""Verified official CIFAR-100 Train loading and frozen baseline transforms.

Training reads only the Train tar member. Test loading has a separate explicit
freeze-evidence guard and is called solely by the final evaluator.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ARCHIVE_BYTES = 169001437
ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
ARCHIVE_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
MEAN = (0.507075159237132, 0.486548873314951, 0.440917843367034)
STD = (0.267334285879241, 0.256438462917088, 0.276150471325684)


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def archive_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "cifar-100-python.tar.gz"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def verify_archive(path: Path | str) -> tuple[Path, dict]:
    path = archive_path(path)
    size = path.stat().st_size
    if size != ARCHIVE_BYTES:
        raise ValueError(f"Official archive size mismatch: {size}")
    md5 = digest_file(path, "md5")
    if md5 != ARCHIVE_MD5:
        raise ValueError("Official archive MD5 mismatch")
    sha256 = digest_file(path)
    if sha256 != ARCHIVE_SHA256:
        raise ValueError("Official archive SHA256 mismatch")
    return path, {"archive_bytes": size, "archive_md5": md5, "archive_sha256": sha256}


def _load_member(path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            if not (member.isfile() or member.isdir()):
                raise ValueError("Unsupported archive member")
        wanted = f"cifar-100-python/{split}"
        matches = [member for member in members if member.name == wanted]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError(f"Missing or duplicate {split} member")
        with archive.extractfile(matches[0]) as stream:
            raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"])
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    size = 50000 if split == "train" else 10000
    if data.dtype != np.uint8 or data.shape != (size, 3072) or labels.shape != (size,):
        raise ValueError(f"Invalid {split} data shape/dtype")
    if labels.min() != 0 or labels.max() != 99 or not np.all(np.bincount(labels) == size // 100):
        raise ValueError(f"Invalid {split} fine labels")
    return data, labels


def load_train(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict]:
    path, identity = verify_archive(path)
    data, labels = _load_member(path, "train")
    identity.update(train_samples=len(labels), test_accessed=False)
    return data, labels, identity


def load_test(path: Path | str, *, frozen_manifest: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    frozen = json.loads(Path(frozen_manifest).read_text(encoding="utf-8"))
    if frozen.get("status") != "ALL_ENDPOINTS_FROZEN" or frozen.get("test_accessed") is not False:
        raise ValueError("Final Test requires all four endpoints frozen first")
    if set(frozen.get("endpoints", {})) != {"dense", "final_q_mask_only", "final_q_fixed_obs", "sparsegpt_native"}:
        raise ValueError("Incomplete frozen evaluation matrix")
    for item in frozen["endpoints"].values():
        if digest_file(Path(item["checkpoint_path"])) != item["checkpoint_sha256"]:
            raise ValueError("Frozen evaluation checkpoint changed")
    path, identity = verify_archive(path)
    data, labels = _load_member(path, "test")
    identity.update(test_samples=len(labels), test_accessed=True)
    return data, labels, identity


def transform(augment: bool):
    from torchvision import transforms
    steps = [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(.5)] if augment else []
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ImageDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, *, augment: bool = False, ids=None):
        self.data, self.labels = data, labels
        self.ids = np.arange(len(labels), dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.transform = transform(augment)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = int(self.ids[index])
        image = self.data[sample_id].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(image), int(self.labels[sample_id]), sample_id


def make_train_dataset(data, labels):
    return ImageDataset(data, labels, augment=True)


def worker_seed(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def training_loader(dataset, batch_size=128, seed=0, epoch=0, workers=2):
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch * 97 + 1)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False, worker_init_fn=worker_seed, generator=generator)


def calibration_ids():
    return np.random.default_rng(20261027).permutation(50000)[:1024].tolist()


def calibration_loader(data, labels, batch_size=128, workers=2, ids=None):
    dataset = ImageDataset(data, labels, augment=False, ids=calibration_ids() if ids is None else ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False, worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(20261027))


def evaluation_loader(data, labels, batch_size=128, workers=2):
    return DataLoader(ImageDataset(data, labels), batch_size=batch_size, shuffle=False,
                      drop_last=False, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(20261027))
