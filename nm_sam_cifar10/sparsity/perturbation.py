"""Canonical SAM perturbation geometry shared by training and evaluation."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

MaskFn = Callable[[torch.Tensor], Optional[torch.Tensor]]


def build_constrained_directions(
    parameters,
    mask_fn: Optional[MaskFn] = None,
    adaptive: bool = False,
) -> Dict[torch.Tensor, torch.Tensor]:
    """Return ``parameter -> allowed gradient`` in parameter iteration order.

    Sparse master weights are masked *before* the global norm is computed.
    Dense parameters, biases and normalization parameters remain unrestricted.
    """
    if isinstance(parameters, torch.nn.Module):
        parameters = parameters.parameters()
    directions: Dict[torch.Tensor, torch.Tensor] = {}
    for parameter in parameters:
        if parameter.grad is None:
            continue
        direction = parameter.grad
        if adaptive:
            direction = direction * (parameter.abs() + 1e-12)
        mask = mask_fn(parameter) if mask_fn is not None else None
        if mask is not None:
            direction = direction * mask
        directions[parameter] = direction
    return directions


def global_direction_norm(directions: Dict[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Global L2 norm over a set of allowed directions."""
    if not directions:
        return torch.tensor(0.0)
    first = next(iter(directions.values()))
    total = torch.zeros((), device=first.device, dtype=first.dtype)
    for direction in directions.values():
        total = total + direction.pow(2).sum()
    return total.sqrt()


def scaled_perturbations(
    directions: Dict[torch.Tensor, torch.Tensor], rho: float
) -> tuple[Dict[torch.Tensor, torch.Tensor], torch.Tensor]:
    norm = global_direction_norm(directions)
    if float(norm.item()) < 1e-12:
        return {}, norm
    scale = float(rho) / norm
    return {parameter: direction * scale for parameter, direction in directions.items()}, norm


__all__ = [
    "MaskFn",
    "build_constrained_directions",
    "global_direction_norm",
    "scaled_perturbations",
]
