"""APBR near-optimal tie-break for physical conditional Group-OBS.

At each deletion, form the set whose *physical* conditional OBS singleton cost
is at most (1 + tau) times its minimum. APBR geometry only breaks the choice
inside that set using |w_j| exp(-s_j), with the virtual w' for the second
deletion. The selected pair receives the unmodified physical joint OBS update.
The baseline path at tau=0 explicitly uses the historical argmin, including
exact cost ties. This adapter never changes H, damping, or compensation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import time

import torch
from torch import Tensor

from baseline_adapters.common.nm import validate_keep_mask


GeometryMode = Literal["physical", "unit", "true", "shuffled"]


@dataclass(frozen=True)
class APBRNearOptimalConditionalOBSResult:
    weight: Tensor
    keep_mask: Tensor
    first_choices: Tensor
    second_choices: Tensor
    chosen_pair_cost: Tensor
    first_candidate_counts: Tensor
    second_candidate_counts: Tensor
    first_min_cost: Tensor
    first_selected_cost: Tensor
    second_min_cost: Tensor
    second_selected_cost: Tensor
    damp: float
    decomposition_seconds: float
    selection_compensation_seconds: float
    geometry_mode: GeometryMode
    tau: float
    shuffle_seed: int | None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def shuffle_log_scale_within_quartets(log_scale: Tensor, seed: int) -> Tensor:
    """Destroy coordinate identity while preserving each row/quartet multiset."""
    if log_scale.ndim != 2 or log_scale.shape[1] % 4:
        raise ValueError("log_scale must be [rows,K] with K divisible by four")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("shuffle seed must be a nonnegative integer")
    if not bool(torch.isfinite(log_scale).all()):
        raise ValueError("Nonfinite log_scale")
    rows, width = log_scale.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_keys = torch.rand((rows, width // 4, 4), generator=generator, dtype=torch.float64)
    permutation = random_keys.argsort(dim=-1).to(log_scale.device)
    return log_scale.reshape(rows, -1, 4).gather(-1, permutation).reshape_as(log_scale)


def _choose(
    physical_cost: Tensor,
    current_weight: Tensor,
    log_scale: Tensor | None,
    tau: float,
    mode: GeometryMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return choice, near-optimal candidate count, min and selected cost."""
    min_cost = physical_cost.min(dim=1).values
    window = torch.isfinite(physical_cost) & (
        physical_cost <= (1.0 + tau) * min_cost[:, None]
    )
    counts = window.sum(dim=1).to(torch.uint8)
    # An exact replay path is essential: geometry never resolves physical ties
    # when tau=0, and the physical-only arm ignores the optional tie-break.
    if tau == 0.0 or mode == "physical":
        choice = physical_cost.argmin(dim=1)
    else:
        assert log_scale is not None
        # log(|w/a|^2) has the same ordering as |w|^2 exp(-2s), without
        # overflow/underflow from exponentiating the learned FP64 log scales.
        geometric_cost = 2.0 * torch.log(current_weight.abs()) - 2.0 * log_scale
        choice = geometric_cost.masked_fill(~window, torch.inf).argmin(dim=1)
    selected_cost = physical_cost.gather(1, choice[:, None]).squeeze(1)
    return choice, counts, min_cost, selected_cost


@torch.no_grad()
def apbr_nearoptimal_conditional_obs_prune(
    weight: Tensor,
    hessian: Tensor,
    log_scale: Tensor | None = None,
    *,
    tau: float = 0.0,
    geometry_mode: GeometryMode = "physical",
    shuffle_seed: int | None = None,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> APBRNearOptimalConditionalOBSResult:
    """Select two deletions per quartet and apply physical joint OBS once.

    ``physical`` is the historical conditional greedy selector. ``unit`` sets
    s=0; unlike physical, it can alter decisions inside a nonzero tau window.
    ``true`` uses the supplied log geometry; ``shuffled`` permutes it within
    each output row/quartet using the explicit seed. No option modifies OBS.
    """
    if geometry_mode not in ("physical", "unit", "true", "shuffled"):
        raise ValueError("Unknown geometry_mode")
    if not isinstance(tau, (int, float)) or not (0 <= float(tau) < float("inf")):
        raise ValueError("tau must be finite and nonnegative")
    tau = float(tau)
    if blocksize <= 0 or blocksize % 4 or not math.isfinite(percdamp) or percdamp < 0:
        raise ValueError("Require positive multiple-of-four blocksize and nonnegative damping")
    if weight.ndim != 2 or weight.shape[1] % 4:
        raise ValueError("Weight must be [rows,K] with K divisible by four")
    if not weight.is_floating_point() or not hessian.is_floating_point():
        raise ValueError("Weight and Hessian must be floating-point tensors")
    rows, width = weight.shape
    if hessian.shape != (width, width) or weight.device != hessian.device:
        raise ValueError("Weight/Hessian mismatch")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(hessian).all()):
        raise ValueError("Nonfinite weight or Hessian")
    if geometry_mode in ("true", "shuffled"):
        if log_scale is None or log_scale.shape != weight.shape or log_scale.device != weight.device:
            raise ValueError("Matching log_scale is required")
        if not log_scale.is_floating_point():
            raise ValueError("log_scale must be floating-point")
        if not bool(torch.isfinite(log_scale).all()):
            raise ValueError("Nonfinite log_scale")
        s = log_scale.double().clone()
        if geometry_mode == "shuffled":
            if shuffle_seed is None:
                raise ValueError("shuffled mode requires shuffle_seed")
            s = shuffle_log_scale_within_quartets(s, shuffle_seed)
    elif geometry_mode == "unit":
        s = torch.zeros_like(weight, dtype=torch.float64)
    else:
        s = None

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
    candidate_counts = torch.empty((2, rows, width // 4), dtype=torch.uint8, device=w.device)
    min_costs = torch.empty((2, rows, width // 4), dtype=torch.float64, device=w.device)
    selected_costs = torch.empty_like(min_costs)

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

            scale = s[:, global_start:global_start + 4] if s is not None else None
            first_score = .5 * current.square() / diagonal[None, :]
            first, n1, c1min, c1selected = _choose(
                first_score, current, scale, tau, geometry_mode,
            )
            wi = current[row_index, first]
            rii = diagonal[first]
            cross = conditional[first, :]
            virtual_weight = current - (wi / rii)[:, None] * cross
            virtual_diagonal = diagonal[None, :] - cross.square() / rii[:, None]
            available = torch.ones_like(current, dtype=torch.bool)
            available[row_index, first] = False
            if not bool((virtual_diagonal[available] > 0).all()):
                raise RuntimeError(f"Nonpositive conditional second score in group {group_index}")
            safe_diagonal = torch.where(available, virtual_diagonal, torch.ones_like(virtual_diagonal))
            second_score = .5 * virtual_weight.square() / safe_diagonal
            second_score[~available] = torch.inf
            second, n2, c2min, c2selected = _choose(
                second_score, virtual_weight, scale, tau, geometry_mode,
            )
            if not bool(torch.isfinite(first_score).all() and torch.isfinite(second_score[row_index, second]).all()):
                raise FloatingPointError(f"Nonfinite greedy cost in group {group_index}")

            first_choices[:, group_index] = first.to(torch.uint8)
            second_choices[:, group_index] = second.to(torch.uint8)
            candidate_counts[0, :, group_index] = n1
            candidate_counts[1, :, group_index] = n2
            min_costs[0, :, group_index] = c1min
            min_costs[1, :, group_index] = c2min
            selected_costs[0, :, group_index] = c1selected
            selected_costs[1, :, group_index] = c2selected

            # The following physical OBS path is intentionally identical to
            # the historical conditional_greedy_obs adapter.
            pair = torch.stack((first, second), dim=1)
            d0, d1 = first, second
            wd0, wd1 = current[row_index, d0], current[row_index, d1]
            c11, c22, c12 = conditional[d0, d0], conditional[d1, d1], conditional[d0, d1]
            determinant = c11 * c22 - c12.square()
            if not bool((determinant > 0).all()):
                raise RuntimeError(f"Nonpositive pair determinant in group {group_index}")
            lam0 = (c22 * wd0 - c12 * wd1) / determinant
            lam1 = (c11 * wd1 - c12 * wd0) / determinant
            pair_costs[:, group_index] = .5 * (wd0 * lam0 + wd1 * lam1)

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
        raise FloatingPointError("Invalid near-optimal conditional OBS output")
    return APBRNearOptimalConditionalOBSResult(
        w.to(weight.dtype), keep, first_choices, second_choices, pair_costs,
        candidate_counts[0], candidate_counts[1],
        min_costs[0], selected_costs[0], min_costs[1], selected_costs[1],
        float(damp_tensor), decomposition_seconds, selection_compensation_seconds,
        geometry_mode, tau, shuffle_seed if geometry_mode == "shuffled" else None,
    )
