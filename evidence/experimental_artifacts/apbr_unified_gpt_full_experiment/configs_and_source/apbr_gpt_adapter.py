"""Geometry-regularized SparseGPT using Q in selection and OBS compensation.

The APBR geometry enters only through

    Q_o = H_base + lambda * mu_H * diag(q_o / mean(q_eligible)).

Exact mode factorizes one Q per output row. Shared mode is an explicitly
labelled approximation using the output-row mean q. Lambda zero dispatches to
the frozen Original SparseGPT implementation for an exact regression gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.sparsegpt import sparsegpt_matrix


@dataclass(frozen=True)
class APBRGPTResult:
    weight: Tensor
    keep_mask: Tensor
    original_damp: float
    mu_h: float
    decomposition_seconds: float
    compensation_seconds: float
    mode: str


def normalized_inverse_square(log_radius_by_layer: dict[str, Tensor]) -> tuple[dict[str, Tensor], float]:
    """Return exp(-2s)/global eligible mean without layer/row recentering."""
    if not log_radius_by_layer:
        raise ValueError("No eligible APBR geometry")
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    raw = {}
    for name, value in log_radius_by_layer.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite log-radius: {name}")
        q = torch.exp(-2 * value.double())
        if not bool(torch.isfinite(q).all() and (q > 0).all()):
            raise ValueError(f"Invalid inverse-square geometry: {name}")
        raw[name] = q
        total += q.sum().cpu()
        count += q.numel()
    mean = float(total / count)
    return {name: value / mean for name, value in raw.items()}, mean


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _q_core(
    weight: Tensor,
    hessian: Tensor,
    q_normalized: Tensor,
    *,
    lambda_value: float,
    keep_n: int,
    m: int,
    blocksize: int,
    percdamp: float,
    compute_dtype: torch.dtype,
) -> APBRGPTResult:
    w = weight.detach().to(compute_dtype).clone()
    h_raw = hessian.detach().to(compute_dtype).clone()
    q = q_normalized.detach().to(device=w.device, dtype=compute_dtype)
    if weight.ndim != 2 or q.shape != weight.shape or h_raw.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Weight/Hessian/q coordinate mismatch")
    if not bool(torch.isfinite(h_raw).all() and torch.isfinite(q).all() and (q > 0).all()):
        raise ValueError("Non-finite Hessian or invalid q")
    mu_h = torch.diag(h_raw).mean()
    dead = torch.diag(h_raw) == 0
    h_raw[dead, dead] = 1
    w[:, dead] = 0
    damp = percdamp * torch.diag(h_raw).mean()
    h_base = h_raw.clone()
    h_base.diagonal().add_(damp)
    # q is shared across these rows in this core; exact mode calls one row.
    if not bool(torch.equal(q, q[0:1].expand_as(q))):
        raise ValueError("_q_core requires one shared q vector")
    q_matrix = h_base.clone()
    q_matrix.diagonal().add_(float(lambda_value) * mu_h * q[0])
    _sync(w.device)
    decomposition_started = time.perf_counter()
    chol, info = torch.linalg.cholesky_ex(q_matrix)
    if int(info) != 0:
        raise RuntimeError("APBR Q is not positive definite")
    factor = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    _sync(w.device)
    decomposition_seconds = time.perf_counter() - decomposition_started

    compensation_started = time.perf_counter()
    output = torch.zeros_like(w)
    keep = torch.ones_like(w, dtype=torch.bool)
    complete_end = w.shape[1] // m * m
    zeros = m - keep_n
    for i1 in range(0, w.shape[1], blocksize):
        i2 = min(i1 + blocksize, w.shape[1])
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        local_factor = factor[i1:i2, i1:i2]
        local_prune = torch.zeros_like(w1, dtype=torch.bool)
        for local_i in range(i2 - i1):
            global_i = i1 + local_i
            if global_i < complete_end and global_i % m == 0:
                metric = w1[:, local_i:local_i + m].square() / torch.diag(local_factor)[
                    local_i:local_i + m
                ].square()[None, :]
                removed = metric.argsort(dim=1, stable=True)[:, :zeros]
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
        w[:, i2:] -= err1 @ factor[i1:i2, i2:]
    validate_keep_mask(keep, keep_n, m)
    if not bool(torch.isfinite(output).all()) or torch.count_nonzero(output[~keep]):
        raise FloatingPointError("APBR-GPT output invariant failed")
    _sync(w.device)
    return APBRGPTResult(
        output.to(weight.dtype), keep, float(damp), float(mu_h),
        decomposition_seconds, time.perf_counter() - compensation_started, "core",
    )


def apbr_gpt_matrix(
    weight: Tensor,
    hessian: Tensor,
    q_normalized: Tensor,
    *,
    lambda_value: float,
    mode: str = "exact_rowwise",
    keep_n: int = 2,
    m: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float64,
) -> APBRGPTResult:
    if lambda_value < 0:
        raise ValueError("lambda must be non-negative")
    if mode not in {"exact_rowwise", "shared_geometry"}:
        raise ValueError("Unknown APBR-GPT mode")
    if q_normalized.shape != weight.shape:
        raise ValueError("q must map every weight coordinate exactly")
    if lambda_value == 0:
        original = sparsegpt_matrix(
            weight, hessian, keep_n=keep_n, m=m, blocksize=blocksize,
            percdamp=percdamp, compute_dtype=compute_dtype,
        )
        return APBRGPTResult(
            original.weight, original.keep_mask, original.damp,
            float(torch.diag(hessian).mean()), original.decomposition_seconds,
            original.compensation_seconds, mode,
        )
    if mode == "shared_geometry":
        shared = q_normalized.mean(dim=0, keepdim=True).expand_as(weight)
        result = _q_core(
            weight, hessian, shared, lambda_value=lambda_value, keep_n=keep_n, m=m,
            blocksize=blocksize, percdamp=percdamp, compute_dtype=compute_dtype,
        )
        return APBRGPTResult(
            result.weight, result.keep_mask, result.original_damp, result.mu_h,
            result.decomposition_seconds, result.compensation_seconds, mode,
        )

    weights, masks = [], []
    decomp = comp = 0.0
    damp = mu = 0.0
    for row in range(weight.shape[0]):
        result = _q_core(
            weight[row:row + 1], hessian, q_normalized[row:row + 1],
            lambda_value=lambda_value, keep_n=keep_n, m=m, blocksize=blocksize,
            percdamp=percdamp, compute_dtype=compute_dtype,
        )
        weights.append(result.weight)
        masks.append(result.keep_mask)
        decomp += result.decomposition_seconds
        comp += result.compensation_seconds
        damp, mu = result.original_damp, result.mu_h
    return APBRGPTResult(torch.cat(weights), torch.cat(masks), damp, mu, decomp, comp, mode)
