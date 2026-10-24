"""Frozen split, SSAM transforms, and isolated deterministic sample RNG."""
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def seed_for(seed, *parts):
    value = json.dumps([seed, *parts], ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') % (2**63-1)


class Cutout:
    """Same p gate, x/y sampling, boundary clipping, and placement as SSAM2022."""
    def __init__(self, size=16, p=0.5):
        self.size, self.p = size, p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image
        h, w = image.shape[-2:]
        x, y = np.random.randint(w), np.random.randint(h)
        mask = torch.ones_like(image)
        half = self.size//2
        mask[:,max(0,y-half):min(h,y+half),max(0,x-half):min(w,x+half)] = 0
        return image*mask


@contextmanager
def sample_rng(seed):
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


class Images(Dataset):
    def __init__(self, dataset, ids, config, training=False, epoch=0):
        self.dataset, self.ids, self.config = dataset, list(ids), config
        self.training, self.epoch = training, epoch
        parts = []
        if training:
            parts = [transforms.RandomCrop(32, padding=config['crop_padding']),
                     transforms.RandomHorizontalFlip(config['horizontal_flip_probability'])]
        parts += [transforms.ToTensor(),transforms.Normalize(config['normalize_mean'], config['normalize_std'])]
        if training:
            parts += [Cutout(config['cutout_size'], config['cutout_probability'])]
        self.transform = transforms.Compose(parts)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, position):
        index = self.ids[position]
        image, label = self.dataset[index]
        with sample_rng(seed_for(self.config['seed'], 'augment', self.epoch, index)):
            tensor = self.transform(image)
        return tensor, label, index


def worker_seed(_):
    seed = torch.initial_seed()
    random.seed(seed); np.random.seed(seed % 2**32)


def make_loader(dataset, config, epoch=0, shuffle=False, workers=None):
    if hasattr(dataset, 'epoch'):
        dataset.epoch = epoch
    ids = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(
        seed_for(config['seed'], 'order', epoch))).tolist() if shuffle else list(range(len(dataset)))
    batches = [ids[i:i+config['batch_size']] for i in range(0,len(ids),config['batch_size'])]
    return DataLoader(dataset, batch_sampler=batches, num_workers=config['workers'] if workers is None else workers,
                      pin_memory=config['pin_memory'], worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(seed_for(config['seed'], 'loader', epoch)))


def load_split(root, config):
    path = Path(root)/config['split_file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=config['split_sha256']:
        raise ValueError('Historical split SHA256 mismatch')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    train, val = manifest['train_indices'], manifest['validation_indices']
    if (len(train),len(val)) != (45000,5000) or set(train)&set(val) or set(train+val)!=set(range(50000)):
        raise ValueError('Frozen split integrity')
    if manifest['seed'] != config['seed']:
        raise ValueError('Split seed mismatch')
    return manifest


def real_data(root, config):
    manifest = load_split(root, config)
    dataset = CIFAR10(str(Path(root)/config['data_root']), train=True, download=False)
    return (Images(dataset,manifest['train_indices'],config,True),
            Images(dataset,manifest['validation_indices'],config),manifest)
