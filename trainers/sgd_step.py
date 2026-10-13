"""One shared SGD training step."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SGDStepResult:
    loss: float
    accuracy: float
    batch_size: int


def sgd_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> SGDStepResult:
    model.train()
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return SGDStepResult(float(loss.item()), float(accuracy), int(inputs.shape[0]))


__all__ = ["SGDStepResult", "sgd_step"]
