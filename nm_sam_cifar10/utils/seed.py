"""Reproducibility utilities: seed setting for Python, NumPy, and PyTorch (CPU + CUDA)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Args:
        seed: integer seed value.
        deterministic: if True, also enable deterministic cuDNN. May slow training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Auto-detect device. CUDA if available and prefer_cuda, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
