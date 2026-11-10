"""Native SparseGPT first score, conditional second score, native OBS update.

At each 2:4 group entry the first deletion uses exactly the pinned SparseGPT
score ``w_j**2 / U_jj**2`` and stable tie order. Its effect is then *virtually*
applied to the group's conditional inverse R. On the three surviving coordinates
we rebuild the tiny upper Cholesky factor and reapply the same SparseGPT score
form. The virtual state is discarded; the selected pair is processed by the
original forward-column OBS loop, in original column order.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from baseline_adapters.common.nm import validate_keep_mask


@dataclass(frozen=True)
class GPTScoreConditionalResult:
    weight: Tensor
    keep_mask: Tensor
    first_choices: Tensor
    native_second_choices: Tensor
    second_choices: Tensor
    damp: float
    decomposition_seconds: float
    selection_compensation_seconds: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _conditional_second_denominators(group_factor: Tensor) -> Tensor:
    """Return U' diagonal squares for each possible first deletion.

The remaining coordinates stay in original ascending column order. The factor
is shared across output rows; only the selected first coordinate varies by row.
"""
    inverse = group_factor.T @ group_factor
    denominators = torch.full((4, 4), torch.inf, dtype=inverse.dtype, device=inverse.device)
    for first in range(4):
        rest = torch.tensor([j for j in range(4) if j != first], device=inverse.device)
        selected = inverse.index_select(0, rest).index_select(1, rest)
        cross = inverse.index_select(0, rest)[:, first]
        reduced = selected - torch.outer(cross, cross) / inverse[first, first]
        reduced = (reduced + reduced.T) * 0.5
        upper = torch.linalg.cholesky(reduced, upper=True)
        denominators[first, rest] = torch.diag(upper).square()
    return denominators


@torch.no_grad()
def gpt_score_conditional_prune(
    weight: Tensor,
    hessian: Tensor,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> GPTScoreConditionalResult:
    """Run a 2:4 SparseGPT selector with conditional second-coordinate choice.

    All preprocessing, relative damping and physical OBS weight updates match
    ``sparsegpt_matrix(..., keep_n=2, m=4, compute_dtype=torch.float64)``.
    Only the mask decision at each group entry changes.
    """
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Weight/Hessian shapes are inconsistent")
    if weight.device != hessian.device or blocksize <= 0 or blocksize % 4 or percdamp < 0:
        raise ValueError("Device, blocksize, or damping mismatch")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(hessian).all()):
        raise ValueError("Nonfinite weight or Hessian")
    w = weight.detach().double().clone()
    h = hessian.detach().double().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    damp_tensor = percdamp * torch.diag(h).mean()
    diag = torch.arange(h.shape[0], device=h.device)
    h[diag, diag] += damp_tensor

    _sync(h.device)
    decomposition_started = time.perf_counter()
    try:
        chol = torch.linalg.cholesky(h)
    except RuntimeError as exc:
        raise RuntimeError("Damped Hessian is not positive definite") from exc
    factor = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    _sync(h.device)
    decomposition_seconds = time.perf_counter() - decomposition_started

    rows, width = w.shape
    output = torch.zeros_like(w)
    keep = torch.ones_like(w, dtype=torch.bool)
    complete_end = width // 4 * 4
    first_choices = torch.empty((rows, complete_end // 4), dtype=torch.uint8, device=w.device)
    native_second_choices = torch.empty_like(first_choices)
    second_choices = torch.empty_like(first_choices)
    row_index = torch.arange(rows, device=w.device)
    _sync(h.device)
    started = time.perf_counter()
    for i1 in range(0, width, blocksize):
        i2 = min(i1 + blocksize, width)
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        local_factor = factor[i1:i2, i1:i2]
        local_prune = torch.zeros_like(w1, dtype=torch.bool)
        for local_i in range(i2 - i1):
            global_i = i1 + local_i
            if global_i < complete_end and global_i % 4 == 0:
                group = w1[:, local_i:local_i + 4]
                upper4 = local_factor[local_i:local_i + 4, local_i:local_i + 4]

                # Exactly the original GPT score and its stable lower-index tie.
                first_metric = group.square() / torch.diag(upper4).square()[None, :]
                native_order = first_metric.argsort(dim=1, stable=True)
                first = native_order[:, 0]

                # Virtual full OBS deletion for the selected first coordinate.
                inverse4 = upper4.T @ upper4
                wi = group[row_index, first]
                rii = inverse4[first, first]
                virtual = group - (wi / rii)[:, None] * inverse4[first, :]
                updated_denominators = _conditional_second_denominators(upper4)
                second_metric = virtual.square() / updated_denominators[first, :]
                second_metric[row_index, first] = torch.inf
                if not bool(torch.isfinite(first_metric).all() and
                            torch.isfinite(second_metric[second_metric != torch.inf]).all()):
                    raise FloatingPointError(f"Nonfinite selector metric at group {global_i // 4}")
                second = second_metric.argsort(dim=1, stable=True)[:, 0]
                first_choices[:, global_i // 4] = first.to(torch.uint8)
                native_second_choices[:, global_i // 4] = native_order[:, 1].to(torch.uint8)
                second_choices[:, global_i // 4] = second.to(torch.uint8)
                selected = torch.stack((first, second), dim=1)
                local_prune.scatter_(1, local_i + selected, True)

            # This is the native SparseGPT column-by-column OBS update.
            current = w1[:, local_i]
            quantized = current.clone()
            quantized[local_prune[:, local_i]] = 0
            q1[:, local_i] = quantized
            keep[:, global_i] = ~local_prune[:, local_i]
            error = (current - quantized) / local_factor[local_i, local_i]
            w1[:, local_i:] -= error[:, None] @ local_factor[local_i, local_i:][None, :]
            err1[:, local_i] = error
        output[:, i1:i2] = q1
        w[:, i2:] -= err1 @ factor[i1:i2, i2:]
    validate_keep_mask(keep, 2, 4)
    if not bool(torch.isfinite(output).all()) or int(torch.count_nonzero(output[~keep])):
        raise FloatingPointError("Invalid conditional SparseGPT output")
    _sync(h.device)
    selection_compensation_seconds = time.perf_counter() - started
    return GPTScoreConditionalResult(
        output.to(weight.dtype), keep, first_choices, native_second_choices, second_choices,
        float(damp_tensor), decomposition_seconds, selection_compensation_seconds,
    )
