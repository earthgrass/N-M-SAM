"""Joint group-level OBS costs and dynamic SparseGPT-ordered compensation.

The activation matrix is the SparseGPT reconstruction Gram, not a CE Hessian.
This module never changes damping or the OBS weight update.  APBR enters only
through an additive deletion cost and through synchronized linear propagation
of externally supplied global-APBR perturbation probes.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from baseline_adapters.common.nm import validate_keep_mask


DELETION_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
)


@dataclass(frozen=True)
class CandidateFactors:
    """Frozen training-time 2x2 factors for one layer.

    ``inverse_qdd`` has shape ``[complete_groups, 6, 2, 2]`` and stores
    ``(Q_DD)^-1`` for the six lexicographically ordered deletion pairs.
    """

    inverse_qdd: Tensor
    damped_hessian: Tensor
    inverse_hessian: Tensor
    inverse_factor: Tensor
    damp: float
    condition_max: float


@dataclass(frozen=True)
class DynamicJointOBSResult:
    weight: Tensor
    keep_mask: Tensor
    propagated_probes: Tensor
    selected_pair_index: Tensor
    selected_gpt_cost: Tensor
    selected_apbr_cost: Tensor
    candidate_gpt_costs: Tensor
    candidate_apbr_costs: Tensor
    damp: float
    decomposition_seconds: float
    selection_compensation_seconds: float


def _pairs(device: torch.device) -> Tensor:
    return torch.tensor(DELETION_PAIRS, dtype=torch.long, device=device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _quadratic(values: Tensor, matrices: Tensor) -> Tensor:
    """Evaluate x^T A x with broadcast dimensions before the final two."""
    return torch.einsum("...i,...ij,...j->...", values, matrices, values)


def build_candidate_factors(
    hessian: Tensor,
    *,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float32,
) -> CandidateFactors:
    """Build the frozen GPT system and static per-group training factors."""
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be float32 or float64")
    if percdamp < 0:
        raise ValueError("percdamp must be non-negative")
    h = hessian.detach().to(compute_dtype).clone()
    if not bool(torch.isfinite(h).all()):
        raise ValueError("Non-finite activation Gram")
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    damp = percdamp * torch.diag(h).mean()
    h.diagonal().add_(damp)
    lower, info = torch.linalg.cholesky_ex(h)
    if int(info.max()) != 0:
        raise RuntimeError("Damped activation Gram is not positive definite")
    q = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(q, upper=True)
    end = h.shape[0] // 4 * 4
    groups = end // 4
    pair = _pairs(h.device)
    if groups:
        base = 4 * torch.arange(groups, device=h.device)[:, None]
        index = base[:, :, None] + pair[None, :, :]
        flat = index.reshape(-1, 2)
        qdd = q[flat[:, :, None], flat[:, None, :]].reshape(groups, 6, 2, 2)
        eig = torch.linalg.eigvalsh(qdd)
        if not bool((eig[..., 0] > 0).all()):
            raise RuntimeError("A Q_DD principal submatrix is not positive definite")
        condition = eig[..., 1] / eig[..., 0]
        inverse_qdd = torch.linalg.inv(qdd)
        condition_max = float(condition.max())
    else:
        inverse_qdd = torch.empty(0, 6, 2, 2, device=h.device, dtype=h.dtype)
        condition_max = 0.0
    if not bool(torch.isfinite(q).all() and torch.isfinite(upper).all()
                and torch.isfinite(inverse_qdd).all()):
        raise FloatingPointError("Non-finite GPT candidate system")
    return CandidateFactors(inverse_qdd, h, q, upper, float(damp), condition_max)


def candidate_costs(
    weight: Tensor,
    probes: Tensor,
    inverse_qdd: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return GPT and APBR costs as ``[rows, groups, 6]``.

    ``probes`` is ``[K, rows, columns]``.  K may be one during training or
    eight during formal pruning.  The APBR coefficient is exactly 1/(2K).
    """
    if weight.ndim != 2 or probes.ndim != 3 or probes.shape[1:] != weight.shape:
        raise ValueError("Weight/probe coordinate mismatch")
    if probes.shape[0] < 1:
        raise ValueError("At least one APBR probe is required")
    groups = weight.shape[1] // 4
    if inverse_qdd.shape != (groups, 6, 2, 2):
        raise ValueError("Candidate factor coordinate mismatch")
    inverse_qdd = inverse_qdd.detach()  # Q/H are frozen training-time statistics.
    dtype, device = inverse_qdd.dtype, inverse_qdd.device
    w = weight.to(device=device, dtype=dtype)
    e = probes.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(w).all() and torch.isfinite(e).all()):
        raise ValueError("Non-finite weight or probe")
    pair = _pairs(device)
    wb = w[:, : groups * 4].reshape(w.shape[0], groups, 4)
    eb = e[:, :, : groups * 4].reshape(e.shape[0], e.shape[1], groups, 4)
    wd = wb[..., pair]
    ed = eb[..., pair]
    gpt = 0.5 * _quadratic(wd, inverse_qdd[None])
    apbr = 0.5 * _quadratic(ed, inverse_qdd[None, None]).mean(dim=0)
    if not bool(torch.isfinite(gpt).all() and torch.isfinite(apbr).all()):
        raise FloatingPointError("Non-finite joint deletion cost")
    return gpt, apbr


def softmin_pruning_regularizer(
    weight: Tensor,
    probe: Tensor,
    inverse_qdd: Tensor,
    reference_cost: float | Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Differentiable normalized six-candidate soft-min for one layer."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    gpt, apbr = candidate_costs(weight, probe[None], inverse_qdd)
    reference = torch.as_tensor(reference_cost, dtype=gpt.dtype, device=gpt.device)
    if reference.numel() != 1 or not bool(torch.isfinite(reference)) or float(reference) <= 0:
        raise ValueError("reference_cost must be a positive frozen scalar")
    normalized = (gpt + apbr) / reference
    result = -temperature * (
        torch.logsumexp(-normalized / temperature, dim=-1) - torch.log(
            torch.tensor(6.0, dtype=normalized.dtype, device=normalized.device)
        )
    )
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Non-finite pruning regularizer")
    return result.mean()


def shuffle_probes_within_groups(probes: Tensor, *, seed: int) -> Tensor:
    """Shuffle coordinates independently per probe/output-row/2:4 group."""
    if probes.ndim != 3:
        raise ValueError("Probes must have shape [K, rows, columns]")
    end = probes.shape[-1] // 4 * 4
    groups = end // 4
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.rand((*probes.shape[:2], groups, 4), generator=generator, dtype=torch.float64)
    permutation = keys.argsort(dim=-1, stable=True).to(probes.device)
    blocks = probes[..., :end].reshape(*probes.shape[:2], groups, 4)
    result = probes.clone()
    result[..., :end] = blocks.gather(-1, permutation).reshape(*probes.shape[:2], end)
    return result


def dynamic_joint_obs_matrix(
    weight: Tensor,
    hessian: Tensor,
    probes: Tensor,
    *,
    shuffled_seed: int | None = None,
    keep_n: int = 2,
    m: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float32,
) -> DynamicJointOBSResult:
    """Dynamic group-level selection embedded in SparseGPT's OBS order."""
    if keep_n != 2 or m != 4 or blocksize % 4:
        raise ValueError("This implementation is frozen to 2:4 and aligned blocks")
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Weight/Hessian coordinate mismatch")
    if probes.ndim != 3 or probes.shape[1:] != weight.shape or probes.shape[0] < 1:
        raise ValueError("Probe coordinate mismatch")
    w = weight.detach().to(compute_dtype).clone()
    e = probes.detach().to(device=w.device, dtype=compute_dtype).clone()
    if shuffled_seed is not None:
        e = shuffle_probes_within_groups(e, seed=shuffled_seed)
    h = hessian.detach().to(compute_dtype).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    e[:, :, dead] = 0
    damp = percdamp * torch.diag(h).mean()
    h.diagonal().add_(damp)
    _sync(w.device)
    started = time.perf_counter()
    lower, info = torch.linalg.cholesky_ex(h)
    if int(info.max()) != 0:
        raise RuntimeError("Damped activation Gram is not positive definite")
    factor = torch.linalg.cholesky(torch.cholesky_inverse(lower), upper=True)
    _sync(w.device)
    decomposition_seconds = time.perf_counter() - started

    started = time.perf_counter()
    keep = torch.ones_like(w, dtype=torch.bool)
    groups = w.shape[1] // 4
    selected = torch.empty(w.shape[0], groups, dtype=torch.long, device=w.device)
    selected_gpt = torch.empty(w.shape[0], groups, dtype=w.dtype, device=w.device)
    selected_apbr = torch.empty_like(selected_gpt)
    all_gpt = torch.empty(w.shape[0], groups, 6, dtype=w.dtype, device=w.device)
    all_apbr = torch.empty_like(all_gpt)
    pair = _pairs(w.device)
    complete_end = groups * 4

    # Each four-column group is solved as one exact multi-coordinate
    # constraint under the current trailing inverse factor.  The whole group
    # is then frozen; the next group starts from U[i+4:, i+4:].
    for group_index in range(groups):
        global_i = 4 * group_index
        cols = global_i + torch.arange(4, device=w.device)
        delete_cols = cols[pair]
        trailing_factor = factor[global_i:, global_i:]
        trailing_deleted = factor[global_i:, delete_cols]
        qdd = torch.einsum("npi,npj->pij", trailing_deleted, trailing_deleted)
        rq = torch.linalg.inv(qdd)
        current_w = w[:, cols]
        current_e = e[:, :, cols]
        wd = current_w[..., pair]
        ed = current_e[..., pair]
        gpt = 0.5 * _quadratic(wd, rq[None])
        apbr = 0.5 * _quadratic(ed, rq[None, None]).mean(dim=0)
        choice = (gpt + apbr).argmin(dim=-1)  # lexicographic first on ties
        selected[:, group_index] = choice
        all_gpt[:, group_index] = gpt
        all_apbr[:, group_index] = apbr
        selected_gpt[:, group_index] = gpt.gather(-1, choice[:, None]).squeeze(-1)
        selected_apbr[:, group_index] = apbr.gather(-1, choice[:, None]).squeeze(-1)

        for pair_index in range(6):
            rows = torch.nonzero(choice == pair_index, as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                continue
            deleted = delete_cols[pair_index]
            # Q_R,D = U_R^T U_D for the current trailing system.
            cross = trailing_factor.T @ trailing_deleted[:, pair_index]
            row_values = w[rows][:, deleted]
            coefficients = row_values @ rq[pair_index].T
            updated = w[rows, global_i:] - coefficients @ cross.T
            w[rows, global_i:] = updated
            w[rows[:, None], deleted[None, :]] = 0
            keep[rows[:, None], deleted[None, :]] = False

            probe_values = e[:, rows][:, :, deleted]
            probe_coefficients = probe_values @ rq[pair_index].T
            e[:, rows, global_i:] -= probe_coefficients @ cross.T
            e[:, rows[:, None], deleted[None, :]] = 0

    validate_keep_mask(keep, keep_n=2, m=4)
    if torch.count_nonzero(w[~keep]) or not bool(torch.isfinite(w).all()):
        raise FloatingPointError("Dynamic Joint OBS output invariant failed")
    if not bool(torch.isfinite(e).all()):
        raise FloatingPointError("Dynamic Joint OBS probe propagation failed")
    _sync(w.device)
    return DynamicJointOBSResult(
        w.to(weight.dtype), keep, e.to(probes.dtype), selected,
        selected_gpt, selected_apbr, all_gpt, all_apbr, float(damp),
        decomposition_seconds, time.perf_counter() - started,
    )
