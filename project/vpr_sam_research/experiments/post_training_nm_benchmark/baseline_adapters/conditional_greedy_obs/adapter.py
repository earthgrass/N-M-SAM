"""Two-stage conditional greedy 2:4 selection with unchanged joint OBS update.

The first deletion is chosen by singleton OBS cost. The second is chosen after
a *virtual* rank-one OBS update of the current quartet. The virtual update is
used only for selection; the selected pair is compensated once, jointly, from
the original current weights. All previously processed quartet coordinates are
frozen, as in the existing dynamic Group-OBS adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from baseline_adapters.common.nm import validate_keep_mask


@dataclass(frozen=True)
class ConditionalGreedyOBSResult:
    weight: Tensor
    keep_mask: Tensor
    first_choices: Tensor
    second_choices: Tensor
    chosen_pair_cost: Tensor
    damp: float
    decomposition_seconds: float
    selection_compensation_seconds: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def conditional_greedy_obs_prune(
    weight: Tensor,
    hessian: Tensor,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> ConditionalGreedyOBSResult:
    """Select two deletions greedily per quartet, then jointly compensate.

    The local 4x4 ``R`` is the conditional inverse ``U_gg.T @ U_gg`` from the
    shared damped Hessian. The first score is ``w_i^2/(2 R_ii)``. After
    virtual deletion of ``i``, candidate ``j`` has
    ``w'_j = w_j - R_ji w_i/R_ii`` and
    ``R'_jj = R_jj - R_ji^2/R_ii``. Only the final pair is applied to weights.
    """
    if blocksize <= 0 or blocksize % 4 or percdamp < 0:
        raise ValueError("Require a positive multiple-of-four blocksize and nonnegative damping")
    if weight.ndim != 2 or weight.shape[1] % 4:
        raise ValueError("Weight must be [rows,K] with K divisible by four")
    rows, width = weight.shape
    if hessian.shape != (width, width) or weight.device != hessian.device:
        raise ValueError("Weight/Hessian mismatch")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(hessian).all()):
        raise ValueError("Nonfinite weight or Hessian")

    h = hessian.double().clone()
    w = weight.double().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    damp_tensor = percdamp * torch.diag(h).mean()
    indices = torch.arange(width, device=w.device)
    h[indices, indices] += damp_tensor

    _sync(w.device)
    decomposition_started = time.perf_counter()
    lower = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(inverse, upper=True)
    _sync(w.device)
    decomposition_seconds = time.perf_counter() - decomposition_started

    row_index = torch.arange(rows, device=w.device)
    keep = torch.ones_like(w, dtype=torch.bool)
    first_choices = torch.empty((rows, width // 4), dtype=torch.uint8, device=w.device)
    second_choices = torch.empty_like(first_choices)
    pair_costs = torch.empty((rows, width // 4), dtype=torch.float64, device=w.device)
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
            group_index = global_start // 4
            current = block[:, columns].clone()
            local_factor = factor[columns, columns]
            conditional = local_factor.T @ local_factor
            diagonal = torch.diag(conditional)
            if not bool((diagonal > 0).all()):
                raise RuntimeError(f"Nonpositive conditional diagonal in group {group_index}")

            first_score = .5 * current.square() / diagonal[None, :]
            first = first_score.argmin(dim=1)
            wi = current[row_index, first]
            rii = diagonal[first]
            cross = conditional[first, :]  # [rows, 4]
            virtual_weight = current - (wi / rii)[:, None] * cross
            virtual_diagonal = diagonal[None, :] - cross.square() / rii[:, None]
            available = torch.ones_like(current, dtype=torch.bool)
            available[row_index, first] = False
            if not bool((virtual_diagonal[available] > 0).all()):
                raise RuntimeError(f"Nonpositive conditional second score in group {group_index}")
            safe_diagonal = torch.where(available, virtual_diagonal, torch.ones_like(virtual_diagonal))
            second_score = .5 * virtual_weight.square() / safe_diagonal
            second_score[~available] = torch.inf
            second = second_score.argmin(dim=1)
            if not bool(torch.isfinite(first_score).all() and torch.isfinite(second_score[row_index, second]).all()):
                raise FloatingPointError(f"Nonfinite greedy cost in group {group_index}")

            first_choices[:, group_index] = first.to(torch.uint8)
            second_choices[:, group_index] = second.to(torch.uint8)
            pair = torch.stack((first, second), dim=1)
            d0, d1 = first, second
            wd0, wd1 = current[row_index, d0], current[row_index, d1]
            c11, c22, c12 = conditional[d0, d0], conditional[d1, d1], conditional[d0, d1]
            determinant = c11 * c22 - c12.square()
            if not bool((determinant > 0).all()):
                raise RuntimeError(f"Nonpositive pair determinant in group {group_index}")
            lam0 = (c22 * wd0 - c12 * wd1) / determinant
            lam1 = (c11 * wd1 - c12 * wd0) / determinant
            direct_pair_cost = .5 * (wd0 * lam0 + wd1 * lam1)
            pair_costs[:, group_index] = direct_pair_cost

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
    w[~keep] = 0
    _sync(w.device)
    selection_compensation_seconds = time.perf_counter() - started
    validate_keep_mask(keep, 2, 4)
    if not bool(torch.isfinite(w).all() and torch.isfinite(pair_costs).all()) or int(torch.count_nonzero(w[~keep])):
        raise FloatingPointError("Invalid final greedy Group-OBS output")
    return ConditionalGreedyOBSResult(
        w.to(weight.dtype), keep, first_choices, second_choices, pair_costs,
        float(damp_tensor), decomposition_seconds, selection_compensation_seconds,
    )
