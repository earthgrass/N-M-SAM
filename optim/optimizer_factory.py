"""Factory for the shared SGD and SAM optimizer components."""
from __future__ import annotations

from typing import Any

import torch

from .sam import SAM, make_sam_optimizer


def build_optimizer(model: torch.nn.Module, kind: str, args: Any):
    if kind == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    if kind == "sam":
        return make_sam_optimizer(
            model,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rho=args.rho,
            nesterov=args.nesterov,
            adaptive=args.adaptive,
        )
    raise ValueError(f"Unknown optimizer kind '{kind}'")


def base_optimizer(optimizer):
    return optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer


__all__ = ["build_optimizer", "base_optimizer"]
