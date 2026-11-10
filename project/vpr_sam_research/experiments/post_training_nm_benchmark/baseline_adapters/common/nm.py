"""Canonical output-row-contiguous N:M utilities.

``keep_n`` is the number retained. Official Wanda/SparseGPT arguments named
``prune_n`` instead count removed entries; that distinction is explicit here.
"""
from __future__ import annotations

import torch
from torch import Tensor


def keep_mask_from_prune_metric(metric: Tensor, keep_n: int = 2, m: int = 4) -> Tensor:
    if metric.ndim != 2 or not 0 < keep_n < m:
        raise ValueError("Expected a 2-D metric and 0 < keep_n < m")
    rows, cols = metric.shape
    end = cols // m * m
    keep = torch.ones_like(metric, dtype=torch.bool)
    if end:
        blocks = metric[:, :end].reshape(rows, -1, m)
        # A smaller pruning metric is removed. stable=True freezes lower-index
        # removal on exact ties and matches the project convention explicitly.
        removed = blocks.argsort(dim=-1, stable=True)[..., : m - keep_n]
        block_keep = torch.ones_like(blocks, dtype=torch.bool)
        block_keep.scatter_(-1, removed, False)
        keep[:, :end] = block_keep.reshape(rows, end)
    return keep


def validate_keep_mask(mask: Tensor, keep_n: int = 2, m: int = 4) -> None:
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("Mask must be a 2-D bool tensor")
    end = mask.shape[1] // m * m
    if end and not bool((mask[:, :end].reshape(-1, m).sum(1) == keep_n).all()):
        raise ValueError("N:M violation")
    if not bool(mask[:, end:].all()):
        raise ValueError("Incomplete tail must remain dense")
