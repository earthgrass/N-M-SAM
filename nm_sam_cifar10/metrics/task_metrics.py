"""Task-performance evaluation shared by all methods."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TaskMetrics:
    loss: float
    accuracy: float
    samples: int


def evaluate_task(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> TaskMetrics:
    was_training = model.training
    model.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss_sum += criterion(logits, targets).item() * inputs.shape[0]
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(inputs.shape[0])
    if was_training:
        model.train()
    return TaskMetrics(loss_sum / max(total, 1), correct / max(total, 1), total)


__all__ = ["TaskMetrics", "evaluate_task"]
