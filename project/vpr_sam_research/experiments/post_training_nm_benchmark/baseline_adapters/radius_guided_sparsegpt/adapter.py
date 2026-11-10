"""SparseGPT dynamic N:M selection guided by a frozen APBR log-radius.

This module is intentionally isolated from the frozen original SparseGPT
adapter.  Relative to ``baseline_adapters.sparsegpt.adapter.sparsegpt_matrix``,
the only algorithmic change is the per-decision ranking metric

    C_hybrid = C_gpt * exp(-log_radius).

Hessian construction, damping, factorization, block traversal, OBS error
propagation, dense-tail behavior, and coordinate writeback are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from baseline_adapters.common.nm import validate_keep_mask


@dataclass(frozen=True)
class RadiusGuidedSparseGPTResult:
    weight: Tensor
    keep_mask: Tensor
    hessian: Tensor
    damp: float
    decomposition_seconds: float
    compensation_seconds: float
    original_metric: Tensor
    hybrid_metric: Tensor


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def radius_guided_sparsegpt_matrix(
    weight: Tensor,
    hessian: Tensor,
    log_radius: Tensor,
    *,
    keep_n: int = 2,
    m: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float64,
    radius_lambda: float = 1.0,
) -> RadiusGuidedSparseGPTResult:
    """Run dynamic 2:4 SparseGPT with APBR radius used only for ranking.

    ``log_radius`` must have the same row/column coordinates as ``weight``.
    It is frozen for the entire sequential OBS pass.  ``radius_lambda`` is
    exposed solely for the registered lambda=0 regression gate; formal runs
    must use lambda=1.
    """
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Weight/Hessian shapes are inconsistent")
    if log_radius.shape != weight.shape:
        raise ValueError("Log-radius must exactly match matrix coordinates")
    if not 0 < keep_n < m or blocksize % m:
        raise ValueError("Require 0 < keep_n < m and blocksize divisible by m")
    if percdamp < 0:
        raise ValueError("percdamp must be non-negative")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be torch.float32 or torch.float64")
    if not isinstance(radius_lambda, (int, float)) or not torch.isfinite(torch.tensor(float(radius_lambda))):
        raise ValueError("radius_lambda must be finite")

    w = weight.detach().to(compute_dtype).clone()
    h = hessian.detach().to(compute_dtype).clone()
    s = log_radius.detach().to(device=w.device, dtype=compute_dtype).clone()
    if not bool(torch.isfinite(h).all()):
        raise ValueError("Non-finite Hessian")
    if not bool(torch.isfinite(s).all()):
        raise ValueError("Non-finite APBR log-radius")
    # Positivity is guaranteed structurally: exp(s)>0 for every finite s.
    radius = torch.exp(s)
    if not bool(torch.isfinite(radius).all()) or not bool((radius > 0).all()):
        raise ValueError("APBR radius is non-positive or not representable")

    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    damp_tensor = percdamp * torch.diag(h).mean()
    diag = torch.arange(h.shape[0], device=h.device)
    h[diag, diag] += damp_tensor
    _synchronize(h.device)
    decomposition_started = time.perf_counter()
    try:
        chol = torch.linalg.cholesky(h)
    except RuntimeError as exc:
        raise RuntimeError("Damped Hessian is not positive definite") from exc
    hinv_factor = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    _synchronize(h.device)
    decomposition_seconds = time.perf_counter() - decomposition_started

    compensation_started = time.perf_counter()
    output = torch.zeros_like(w)
    keep = torch.ones_like(w, dtype=torch.bool)
    original_metric_all = torch.full_like(w, torch.nan)
    hybrid_metric_all = torch.full_like(w, torch.nan)
    complete_end = w.shape[1] // m * m
    zeros = m - keep_n
    for i1 in range(0, w.shape[1], blocksize):
        i2 = min(i1 + blocksize, w.shape[1])
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        local_factor = hinv_factor[i1:i2, i1:i2]
        local_prune = torch.zeros_like(w1, dtype=torch.bool)
        for local_i in range(i2 - i1):
            global_i = i1 + local_i
            if global_i < complete_end and global_i % m == 0:
                width = min(m, complete_end - global_i)
                original_metric = w1[:, local_i : local_i + width].square() / torch.diag(local_factor)[
                    local_i : local_i + width
                ].square()[None, :]
                # exp(-lambda*s) is exactly r**(-lambda); lambda=1 is formal.
                hybrid_metric = original_metric * torch.exp(
                    -float(radius_lambda) * s[:, global_i : global_i + width]
                )
                if not bool(torch.isfinite(original_metric).all() and torch.isfinite(hybrid_metric).all()):
                    raise FloatingPointError("Non-finite SparseGPT ranking metric")
                original_metric_all[:, global_i : global_i + width] = original_metric
                hybrid_metric_all[:, global_i : global_i + width] = hybrid_metric
                removed = hybrid_metric.argsort(dim=1, stable=True)[:, :zeros]
                local_prune.scatter_(1, local_i + removed, True)
            current = w1[:, local_i]
            quantized = current.clone()
            quantized[local_prune[:, local_i]] = 0
            q1[:, local_i] = quantized
            keep[:, global_i] = ~local_prune[:, local_i]
            error = (current - quantized) / local_factor[local_i, local_i]
            w1[:, local_i:] -= error[:, None] @ local_factor[local_i, local_i:][None, :]
            err1[:, local_i] = error
        output[:, i1:i2] = q1
        w[:, i2:] -= err1 @ hinv_factor[i1:i2, i2:]

    validate_keep_mask(keep, keep_n, m)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Radius-guided SparseGPT produced NaN/Inf")
    _synchronize(h.device)
    compensation_seconds = time.perf_counter() - compensation_started
    return RadiusGuidedSparseGPTResult(
        output.to(weight.dtype), keep, hessian.detach().clone(), float(damp_tensor),
        decomposition_seconds, compensation_seconds, original_metric_all, hybrid_metric_all,
    )
