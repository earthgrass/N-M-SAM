"""Task CE and explicit optimizer L2 are separate observable quantities."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .batch import StepBatch


@dataclass(frozen=True)
class LossReport:
    task_loss: torch.Tensor
    l2_penalty: torch.Tensor
    optimizer_loss: torch.Tensor
    logits: torch.Tensor | None = None


def classification_loss(model: nn.Module, batch: StepBatch, *, l2_reg: float = 0.0,
                        no_weight_decay_on_bn: bool = False) -> LossReport:
    """Official exclusion uses ndim>1 (also excludes all 1D biases)."""
    if not math.isfinite(l2_reg) or l2_reg < 0:
        raise ValueError("l2_reg must be finite and nonnegative")
    logits = model(batch.inputs)
    task = F.cross_entropy(logits, batch.targets)
    penalty = task.new_zeros(())
    for p in model.parameters():
        if p.requires_grad and (not no_weight_decay_on_bn or p.ndim > 1):
            penalty = penalty + (l2_reg * 0.5) * p.square().sum()
    return LossReport(task, penalty, task + penalty, logits)

