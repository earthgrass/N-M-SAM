"""Static local pairwise OBS selection for canonical output-row 2:4 groups.

This module changes mask selection only.  It does not implement a new OBS
compensator.  Selected masks must be passed to the existing fixed-mask OBS
adapter, which preserves the frozen SparseGPT update order.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from baseline_adapters.common.nm import validate_keep_mask


DELETION_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
)
APBR_SHUFFLE_SEEDS: tuple[int, int, int] = (20261031, 20261032, 20261033)
MAX_QDD_CONDITION = 1.0e8
TOLERANCE_ULPS = 64.0


@dataclass(frozen=True)
class OBSSystem:
    """One frozen damped OBS system.

    ``inverse_factor`` is upper triangular and satisfies
    ``inverse_hessian = inverse_factor.T @ inverse_factor``.
    """

    damped_hessian: Tensor
    inverse_hessian: Tensor
    inverse_factor: Tensor
    dead_columns: Tensor
    damp: float


@dataclass(frozen=True)
class GroupSelectionResult:
    keep_mask: Tensor
    selected_pair_index: Tensor
    gpt_costs: Tensor
    apbr_costs: Tensor | None
    selected_gpt_cost: Tensor
    minimum_gpt_cost: Tensor
    selected_apbr_cost: Tensor | None
    candidate_count: Tensor
    relative_gpt_increase: Tensor
    numerical_tolerance: Tensor
    shuffled_seed: int | None


def _require_float_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be torch.float32 or torch.float64")


def _pair_indices(device: torch.device) -> Tensor:
    return torch.tensor(DELETION_PAIRS, dtype=torch.long, device=device)


def prepare_obs_system(
    hessian: Tensor,
    *,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float64,
) -> OBSSystem:
    """Build H-damped, its exact inverse Q, and SparseGPT's upper factor."""
    _require_float_dtype(compute_dtype)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    if percdamp < 0:
        raise ValueError("percdamp must be non-negative")
    h = hessian.detach().to(compute_dtype).clone()
    if not bool(torch.isfinite(h).all()):
        raise ValueError("Non-finite Hessian")
    # Do not silently repair an asymmetric input.  The tolerance only admits
    # ordinary floating-point accumulation noise.
    eps = torch.finfo(compute_dtype).eps
    scale = max(1.0, float(h.abs().max()))
    if not bool(torch.allclose(h, h.T, rtol=64 * eps, atol=64 * eps * scale)):
        raise ValueError("Hessian is not symmetric")
    h = 0.5 * (h + h.T)
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    damp_tensor = percdamp * torch.diag(h).mean()
    diagonal = torch.arange(h.shape[0], device=h.device)
    h[diagonal, diagonal] += damp_tensor
    try:
        lower = torch.linalg.cholesky(h)
    except RuntimeError as exc:
        raise RuntimeError("Damped Hessian is not positive definite") from exc
    inverse = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(inverse, upper=True)
    if not bool(torch.isfinite(inverse).all() and torch.isfinite(upper).all()):
        raise FloatingPointError("OBS system contains NaN/Inf")
    return OBSSystem(h, inverse, upper, dead, float(damp_tensor))


def static_group_obs_costs(
    weight: Tensor,
    inverse_hessian: Tensor,
    *,
    m: int = 4,
    condition_limit: float = MAX_QDD_CONDITION,
) -> tuple[Tensor, Tensor]:
    """Return exact static local pair costs and Q_DD condition numbers.

    Output shapes are ``[output_rows, complete_groups, 6]`` and
    ``[complete_groups, 6]``.  This is an isolated-group diagnostic, not the
    exact cost of simultaneously deleting every selected group.
    """
    if m != 4:
        raise ValueError("This implementation is deliberately restricted to 2:4")
    if weight.ndim != 2:
        raise ValueError("Weight must be a 2-D output-row matrix")
    if inverse_hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Weight/Q coordinate mismatch")
    if condition_limit <= 1:
        raise ValueError("condition_limit must exceed one")
    q = inverse_hessian.detach()
    w = weight.detach().to(device=q.device, dtype=q.dtype)
    if not bool(torch.isfinite(w).all() and torch.isfinite(q).all()):
        raise ValueError("Non-finite weight or inverse Hessian")
    eps = torch.finfo(q.dtype).eps
    scale = max(1.0, float(q.abs().max()))
    if not bool(torch.allclose(q, q.T, rtol=64 * eps, atol=64 * eps * scale)):
        raise ValueError("Inverse Hessian is not symmetric")
    q = 0.5 * (q + q.T)
    end = weight.shape[1] // 4 * 4
    groups = end // 4
    if groups == 0:
        empty_cost = torch.empty(weight.shape[0], 0, 6, dtype=q.dtype, device=q.device)
        empty_condition = torch.empty(0, 6, dtype=q.dtype, device=q.device)
        return empty_cost, empty_condition

    pairs = _pair_indices(q.device)
    base = 4 * torch.arange(groups, device=q.device)[:, None]
    first = base + pairs[:, 0][None, :]
    second = base + pairs[:, 1][None, :]
    qii = q[first, first]
    qjj = q[second, second]
    qij = q[first, second]
    determinant = qii * qjj - qij.square()
    trace = qii + qjj
    discriminant = torch.sqrt(torch.clamp((qii - qjj).square() + 4 * qij.square(), min=0))
    eig_min = 0.5 * (trace - discriminant)
    eig_max = 0.5 * (trace + discriminant)
    condition = eig_max / eig_min
    if not bool(torch.isfinite(condition).all()):
        raise FloatingPointError("Non-finite Q_DD condition number")
    if not bool((determinant > 0).all() and (eig_min > 0).all()):
        raise RuntimeError("A Q_DD principal submatrix is not positive definite")
    if bool((condition > condition_limit).any()):
        worst = float(condition.max())
        raise RuntimeError(f"Q_DD condition number {worst:.9g} exceeds {condition_limit:.9g}")

    blocks = w[:, :end].reshape(w.shape[0], groups, 4)
    wi = blocks[..., pairs[:, 0]]
    wj = blocks[..., pairs[:, 1]]
    numerator = qjj[None] * wi.square() - 2 * qij[None] * wi * wj + qii[None] * wj.square()
    costs = 0.5 * numerator / determinant[None]
    cost_scale = costs.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1)
    tolerance = TOLERANCE_ULPS * eps * cost_scale
    if bool((costs < -tolerance).any()):
        raise FloatingPointError("Static Group-OBS produced a negative quadratic cost")
    costs = costs.clamp_min(0)
    if not bool(torch.isfinite(costs).all()):
        raise FloatingPointError("Static Group-OBS cost contains NaN/Inf")
    return costs, condition


def _shuffle_log_radius(log_radius: Tensor, *, seed: int, end: int) -> Tensor:
    """Shuffle the four geometry values independently inside every block."""
    rows = log_radius.shape[0]
    groups = end // 4
    blocks = log_radius[:, :end].reshape(rows, groups, 4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_keys = torch.rand((rows, groups, 4), generator=generator, dtype=torch.float64)
    permutation = random_keys.argsort(dim=-1, stable=True).to(blocks.device)
    shuffled = blocks.gather(-1, permutation)
    result = log_radius.clone()
    result[:, :end] = shuffled.reshape(rows, end)
    return result


def apbr_pair_costs(
    weight: Tensor,
    log_radius: Tensor,
    *,
    shuffled_seed: int | None = None,
) -> Tensor:
    """Compute sum(w_i^2 exp(-2s_i)) for all six deletion pairs."""
    if weight.ndim != 2 or log_radius.shape != weight.shape:
        raise ValueError("Weight/APBR geometry coordinate mismatch")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(log_radius).all()):
        raise ValueError("Non-finite weight or APBR geometry")
    end = weight.shape[1] // 4 * 4
    geometry = log_radius.detach().double().to(weight.device)
    if shuffled_seed is not None:
        geometry = _shuffle_log_radius(geometry, seed=shuffled_seed, end=end)
    coordinate_cost = weight.detach().double().square() * torch.exp(-2 * geometry)
    if not bool(torch.isfinite(coordinate_cost).all()):
        raise FloatingPointError("APBR pair cost overflowed")
    blocks = coordinate_cost[:, :end].reshape(weight.shape[0], end // 4, 4)
    pairs = _pair_indices(weight.device)
    return blocks[..., pairs[:, 0]] + blocks[..., pairs[:, 1]]


def select_group_mask(
    weight: Tensor,
    system: OBSSystem,
    *,
    log_radius: Tensor | None = None,
    tau: float = 0.0,
    shuffled_seed: int | None = None,
    condition_limit: float = MAX_QDD_CONDITION,
) -> GroupSelectionResult:
    """Select a strict 2:4 fixed mask from static local pair costs.

    Without ``log_radius``, this is Group-OBS Only.  With geometry, APBR cost
    chooses among candidates within ``tau`` of the minimum GPT pair cost.
    """
    if weight.ndim != 2 or system.inverse_hessian.shape[0] != weight.shape[1]:
        raise ValueError("Weight/OBS system coordinate mismatch")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if shuffled_seed is not None and log_radius is None:
        raise ValueError("A shuffled control requires APBR geometry")
    work = weight.detach().to(device=system.inverse_hessian.device, dtype=system.inverse_hessian.dtype).clone()
    work[:, system.dead_columns] = 0
    gpt_costs, _conditions = static_group_obs_costs(
        work, system.inverse_hessian, condition_limit=condition_limit
    )
    eps = torch.finfo(gpt_costs.dtype).eps
    # Purely relative block scale.  In particular, an all-zero block gets
    # zero tolerance instead of an arbitrary absolute window.
    block_scale = gpt_costs.abs().amax(dim=-1)
    numerical_tolerance = TOLERANCE_ULPS * eps * block_scale
    minimum = gpt_costs.min(dim=-1).values
    near = gpt_costs <= minimum[..., None] * (1 + float(tau)) + numerical_tolerance[..., None]
    candidate_count = near.sum(dim=-1)
    if not bool((candidate_count >= 1).all()):
        raise RuntimeError("Near-optimal candidate set is empty")

    apbr_cost = None
    selected_apbr = None
    if log_radius is None:
        # Group-OBS Only is the first minimum under the frozen pair ordering.
        selected = gpt_costs.argmin(dim=-1)
    else:
        geometry = log_radius.detach().to(device=work.device)
        apbr_cost = apbr_pair_costs(work, geometry, shuffled_seed=shuffled_seed)
        infinity = torch.full_like(apbr_cost, torch.inf)
        selected = torch.where(near, apbr_cost, infinity).argmin(dim=-1)
        selected_apbr = apbr_cost.gather(-1, selected[..., None]).squeeze(-1)

    selected_gpt = gpt_costs.gather(-1, selected[..., None]).squeeze(-1)
    relative = torch.zeros_like(selected_gpt)
    safe = minimum > numerical_tolerance
    relative[safe] = (selected_gpt[safe] - minimum[safe]) / minimum[safe]
    # For a numerically zero minimum, membership already limits the absolute
    # increase to tolerance; report zero rather than an unstable division.
    pairs = _pair_indices(work.device)
    deleted = pairs[selected]
    end = work.shape[1] // 4 * 4
    groups = end // 4
    keep = torch.ones_like(work, dtype=torch.bool)
    block_keep = torch.ones(work.shape[0], groups, 4, dtype=torch.bool, device=work.device)
    block_keep.scatter_(-1, deleted, False)
    keep[:, :end] = block_keep.reshape(work.shape[0], end)
    validate_keep_mask(keep, keep_n=2, m=4)
    return GroupSelectionResult(
        keep,
        selected,
        gpt_costs,
        apbr_cost,
        selected_gpt,
        minimum,
        selected_apbr,
        candidate_count,
        relative,
        numerical_tolerance,
        shuffled_seed,
    )
