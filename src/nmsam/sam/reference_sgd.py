"""Checked torch SGD matching the zero-buffer Flax momentum recurrence."""
import math
from collections.abc import Iterable
import torch


def create_optimizer(parameters: Iterable[torch.nn.Parameter], *, lr: float = 0.1,
                     beta: float = 0.9, nesterov: bool = True) -> torch.optim.SGD:
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("lr > 0 and 0 <= beta < 1 must be finite")
    if nesterov and beta == 0:
        raise ValueError("Nesterov needs positive momentum")
    return torch.optim.SGD(parameters, lr=lr, momentum=beta, dampening=0,
                           nesterov=nesterov, weight_decay=0)

