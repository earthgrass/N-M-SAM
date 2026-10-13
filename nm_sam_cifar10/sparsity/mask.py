"""N:M structured mask construction (block-wise top-K).

This module exposes a **single canonical entry point** :func:`make_nm_mask`
which automatically dispatches between Linear (2D) and Conv2d (4D)
layouts. Conv2d weights are flattened along ``[in, kh, kw]`` before
blocking; the mask is then reshaped back to the original 4D shape.
Trailing partial blocks (length < M) are kept dense.

All mask builders in the project (magnitude, Taylor, SAM-aware) must go
through :func:`make_nm_mask` so that the per-block layout is impossible
to get wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .mask_scores import compute_mask_score


def _blockwise_topk_mask_along_last(flat: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Apply block-wise top-K along the LAST axis. Accepts arbitrary leading
    shape and treats the last axis as the one to block. Returns a 0/1
    mask with the same shape where the top-N entries per consecutive-M
    window are 1."""
    if flat.size(-1) % m != 0:
        raise ValueError(f"last dim {flat.size(-1)} not divisible by m={m}")
    leading = flat.shape[:-1]
    num_full = flat.size(-1) // m
    blocks = flat.reshape(*leading, num_full, m)
    _, idx = torch.topk(blocks, k=n, dim=-1, largest=True, sorted=False)
    out = torch.zeros_like(blocks)
    out.scatter_(-1, idx, 1.0)
    return out.reshape(*leading, num_full * m)


def make_nm_mask(
    weight: torch.Tensor,
    n: int = 2,
    m: int = 4,
    score: str = "magnitude",
    grad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Canonical 2:4 / N:M mask builder.

    Dispatches on ``weight.dim()``:

    * 2D (Linear): block on the last axis (``in_features``).
    * 4D (Conv2d): flatten ``[out, in, kh, kw]`` → ``[out, in*kh*kw]``,
      block on the second axis, reshape back.

    ``score`` chooses the per-position importance:

    * ``"magnitude"``: ``|W|``.
    * ``"taylor_g1"`` / ``"taylor_g2"``: ``|W * g|``.  The caller is
      responsible for supplying the gradient from the selected SAM backward.

    Tail positions (where ``inner % m != 0``) are kept dense.
    """
    if n <= 0 or m <= 0 or n > m:
        raise ValueError("Require 0 < n <= m")
    if weight.dim() not in (2, 4):
        raise ValueError(f"weight must be 2D or 4D, got {tuple(weight.shape)}")

    # ---- importance ----
    importance = compute_mask_score(weight, score, grad)

    # ---- flatten for blocking (Conv2d case) ----
    if weight.dim() == 4:
        out, in_c, kh, kw = weight.shape
        flat = importance.reshape(out, in_c * kh * kw)
    else:
        flat = importance

    # ---- block-wise top-K along the LAST axis ----
    inner = flat.size(-1)
    num_full = inner // m
    tail = inner - num_full * m
    if num_full == 0:
        return torch.ones_like(weight)

    head = flat[..., : num_full * m]
    blk = _blockwise_topk_mask_along_last(head, n=n, m=m)
    if tail > 0:
        tail_mask = torch.ones(*flat.shape[:-1], tail, dtype=weight.dtype, device=weight.device)
        full = torch.cat([blk, tail_mask], dim=-1)
    else:
        full = blk

    if weight.dim() == 4:
        return full.reshape(out, in_c, kh, kw)
    return full


# Back-compat thin wrappers used elsewhere.
def linear_mask(weight: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    return make_nm_mask(weight, n=n, m=m, score="magnitude")


def conv2d_mask(weight: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    return make_nm_mask(weight, n=n, m=m, score="magnitude")


def blockwise_topk_mask(weight: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    return make_nm_mask(weight, n=n, m=m, score="magnitude")


def sam_aware_mask(
    weight: torch.Tensor,
    grad: torch.Tensor,
    n: int = 2,
    m: int = 4,
    score: str = "sam_taylor",
) -> torch.Tensor:
    return make_nm_mask(weight, n=n, m=m, score=score, grad=grad)


@dataclass
class SparsityConfig:
    """Per-layer sparsity configuration."""

    n: int = 2
    m: int = 4
    enabled: bool = True
    score: str = "magnitude"  # for SAM-aware methods
    kind: str = "auto"  # 'linear' | 'conv2d' | 'auto'


def mask_for_weight(weight: torch.Tensor, cfg: SparsityConfig) -> torch.Tensor:
    if not cfg.enabled:
        return torch.ones_like(weight)
    return make_nm_mask(weight, n=cfg.n, m=cfg.m, score=cfg.score)
