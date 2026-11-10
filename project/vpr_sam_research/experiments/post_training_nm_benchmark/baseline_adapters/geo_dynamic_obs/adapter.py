"""Dynamic four-column group OBS with optional APBR geometric scoring.

After a group is processed, all four of its coordinates are frozen. The
conditional inverse for the next group is therefore the Schur complement of
Q with respect to *all* preceding columns. This sequential approximation is
not the global optimum with previously retained coordinates still free.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from baseline_adapters.common.nm import validate_keep_mask

PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
MODES = ("std_group_obs", "geo_select_std_comp", "geo_full")


@dataclass(frozen=True)
class DynamicGroupOBSResult:
    weight: Tensor
    keep_mask: Tensor
    chosen_pairs: Tensor
    candidate_costs: Tensor
    damp: float
    decomposition_seconds: float
    selection_compensation_seconds: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def dynamic_group_obs_prune(
    weight: Tensor,
    hessian: Tensor,
    *,
    log_scale: Tensor | None = None,
    mode: str = "std_group_obs",
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> DynamicGroupOBSResult:
    if mode not in MODES or blocksize <= 0 or blocksize % 4 or percdamp < 0:
        raise ValueError("Invalid mode, blocksize, or damping")
    if weight.ndim != 2 or weight.shape[1] % 4:
        raise ValueError("Weight must be [rows,K] with K divisible by 4")
    rows, width = weight.shape
    if hessian.shape != (width, width) or weight.device != hessian.device:
        raise ValueError("Weight/Hessian mismatch")
    if mode != "std_group_obs" and (log_scale is None or log_scale.shape != weight.shape):
        raise ValueError("Matched APBR log-scale required")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(hessian).all()):
        raise ValueError("Nonfinite weight or Hessian")
    if log_scale is not None and not bool(torch.isfinite(log_scale).all()):
        raise ValueError("Nonfinite APBR log-scale")
    h = hessian.double().clone()
    w = weight.double().clone()
    if log_scale is not None:
        inv_a = torch.exp(-log_scale.double())
        a = torch.exp(log_scale.double())
        if not bool(torch.isfinite(inv_a).all() and torch.isfinite(a).all()):
            raise ValueError("Invalid APBR geometry scale")
    else:
        inv_a = a = None
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    damp = percdamp * torch.diag(h).mean()
    indices = torch.arange(width, device=w.device)
    h[indices, indices] += damp
    _sync(w.device)
    decomposition_started = time.perf_counter()
    lower = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(inverse, upper=True)
    _sync(w.device)
    decomposition_seconds = time.perf_counter() - decomposition_started
    if mode == "geo_full":
        w *= inv_a
    pairs = torch.tensor(PAIRS, dtype=torch.long, device=w.device)
    p0, p1 = pairs[:, 0], pairs[:, 1]
    row_index = torch.arange(rows, device=w.device)
    keep = torch.ones_like(w, dtype=torch.bool)
    chosen = torch.empty((rows, width // 4), dtype=torch.uint8, device=w.device)
    costs_all = torch.empty((rows, width // 4, 6), dtype=torch.float64, device=w.device)
    _sync(w.device)
    started = time.perf_counter()
    for block_start in range(0, width, blocksize):
        block_end = min(block_start + blocksize, width)
        count = block_end - block_start
        block = w[:, block_start:block_end].clone()
        errors = torch.zeros_like(block)
        factor = upper[block_start:block_end, block_start:block_end]
        for offset in range(0, count, 4):
            columns = slice(offset, offset + 4)
            global_start = block_start + offset
            current = block[:, columns].clone()
            local_factor = factor[columns, columns]
            conditional = local_factor.T @ local_factor
            q11, q22, q12 = (conditional[p0, p0], conditional[p1, p1], conditional[p0, p1])
            determinant = q11 * q22 - q12.square()
            if not bool((determinant > 0).all()):
                raise RuntimeError(f"Nonpositive pair determinant at group {global_start // 4}")
            scored = (current * inv_a[:, global_start:global_start + 4]
                      if mode == "geo_select_std_comp" else current)
            v0, v1 = scored[:, p0], scored[:, p1]
            costs = .5 * (q22 * v0.square() - 2 * q12 * v0 * v1 + q11 * v1.square()) / determinant
            if not bool(torch.isfinite(costs).all()):
                raise FloatingPointError("Nonfinite pair cost")
            best = costs.argmin(dim=1)
            pair = pairs[best]
            chosen[:, global_start // 4] = best.to(torch.uint8)
            costs_all[:, global_start // 4] = costs
            d0, d1 = pair[:, 0], pair[:, 1]
            wd0, wd1 = current[row_index, d0], current[row_index, d1]
            c11, c22, c12 = conditional[d0, d0], conditional[d1, d1], conditional[d0, d1]
            det_selected = c11 * c22 - c12.square()
            lam0 = (c22 * wd0 - c12 * wd1) / det_selected
            lam1 = (c11 * wd1 - c12 * wd0) / det_selected
            multipliers = torch.zeros((rows, 4), dtype=torch.float64, device=w.device)
            multipliers[row_index, d0] = lam0
            multipliers[row_index, d1] = lam1
            block[:, columns] -= multipliers @ conditional
            block[row_index[:, None], offset + pair] = 0
            keep[row_index[:, None], global_start + pair] = False
            group_error = multipliers @ local_factor.T
            errors[:, columns] = group_error
            if offset + 4 < count:
                block[:, offset + 4:count] -= group_error @ factor[columns, offset + 4:count]
        w[:, block_start:block_end] = block
        if block_end < width:
            w[:, block_end:] -= errors @ upper[block_start:block_end, block_end:]
    if mode == "geo_full":
        w *= a
    w[~keep] = 0
    _sync(w.device)
    selection_compensation_seconds = time.perf_counter() - started
    validate_keep_mask(keep, 2, 4)
    if not bool(torch.isfinite(w).all()) or int(torch.count_nonzero(w[~keep])):
        raise FloatingPointError("Invalid final OBS output")
    return DynamicGroupOBSResult(w.to(weight.dtype), keep, chosen, costs_all,
                                 float(damp), decomposition_seconds, selection_compensation_seconds)
