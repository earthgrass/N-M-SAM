from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
import torch


def fake_loader(*, batch_size: int, steps: int, seed: int) -> DataLoader:
    """No download, deterministic data and sampler RNG isolated from model RNG."""
    data = FakeData(size=batch_size * steps, image_size=(1, 4, 4), num_classes=3,
                    transform=ToTensor(), random_offset=seed)
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
