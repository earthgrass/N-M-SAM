"""N:M structural and dynamic-mask metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SparsitySnapshot:
    configured_n: int | None
    configured_m: int | None
    configured_sparsity: float
    actual_sparsity: float
    total_blocks: int
    bad_block_count: int
    layer_sparsity: Dict[str, float]


def collect_sparsity_metrics(model: nn.Module, n: int = 2, m: int = 4) -> SparsitySnapshot:
    wrappers = list(getattr(model, "_nm_wrappers", []))
    if not wrappers:
        return SparsitySnapshot(None, None, 0.0, 0.0, 0, 0, {})

    total_supported_weights = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
            total_supported_weights += module.weight.numel()
    sparse_weight_count = sum(wrapper.mask.numel() for wrapper in wrappers)
    total_supported_weights += sparse_weight_count

    zeros = 0
    total_blocks = 0
    bad_blocks = 0
    layer_sparsity: Dict[str, float] = {}
    for wrapper in wrappers:
        mask = wrapper.mask
        zeros += int((mask == 0).sum().item())
        layer_sparsity[wrapper.name] = float((mask == 0).float().mean().item())
        flat = mask.reshape(mask.shape[0], -1)
        full = (flat.shape[1] // m) * m
        if full:
            sums = flat[:, :full].reshape(-1, m).sum(dim=1)
            total_blocks += int(sums.numel())
            bad_blocks += int((sums != n).sum().item())
    return SparsitySnapshot(
        n,
        m,
        1.0 - n / m,
        zeros / max(total_supported_weights, 1),
        total_blocks,
        bad_blocks,
        layer_sparsity,
    )


__all__ = ["SparsitySnapshot", "collect_sparsity_metrics"]
