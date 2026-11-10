"""Exploratory fixed-support OBS compensation using SparseGPT's update order.

This module never selects a mask.  The caller supplies a canonical bool
keep-mask, and every deletion decision in the sequential update is read from
that immutable mask.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor, nn

from baseline_adapters.common.conv2d import grouped_patch_matrices, weight_to_matrix
from baseline_adapters.common.nm import validate_keep_mask


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class FixedMaskOBSResult:
    weight: Tensor
    keep_mask: Tensor
    damp: float
    decomposition_seconds: float
    compensation_seconds: float
    survivor_changed_count: int


@torch.no_grad()
def fixedmask_obs_matrix(
    weight: Tensor,
    hessian: Tensor,
    keep_mask: Tensor,
    *,
    keep_n: int = 2,
    m: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float64,
    compensate: bool = True,
) -> FixedMaskOBSResult:
    """Apply SparseGPT-ordered OBS updates on an externally fixed support.

    ``keep_mask`` is validated but never derived from ``weight`` or Hessian
    scores.  With ``compensate=False`` this is exactly mask-only projection,
    which is useful for auditing the APBR Raw-Hard reconstruction path.
    """
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Weight/Hessian shapes are inconsistent")
    if keep_mask.shape != weight.shape or keep_mask.dtype != torch.bool:
        raise ValueError("Fixed keep-mask shape/dtype mismatch")
    if not 0 < keep_n < m or blocksize % m:
        raise ValueError("Require 0 < keep_n < m and blocksize divisible by m")
    if percdamp < 0:
        raise ValueError("percdamp must be non-negative")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be torch.float32 or torch.float64")
    validate_keep_mask(keep_mask, keep_n, m)
    fixed_keep = keep_mask.detach().clone()
    original = weight.detach().to(compute_dtype).clone()
    if not compensate:
        output = original * fixed_keep.to(original.dtype)
        return FixedMaskOBSResult(
            output.to(weight.dtype), fixed_keep, 0.0, 0.0, 0.0, 0
        )

    h = hessian.detach().to(compute_dtype).clone()
    if not bool(torch.isfinite(h).all()):
        raise ValueError("Non-finite Hessian")
    work = original.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    work[:, dead] = 0
    damp_tensor = percdamp * torch.diag(h).mean()
    diagonal = torch.arange(h.shape[0], device=h.device)
    h[diagonal, diagonal] += damp_tensor

    _synchronize(h.device)
    decomposition_started = time.perf_counter()
    try:
        chol = torch.linalg.cholesky(h)
    except RuntimeError as exc:
        raise RuntimeError("Damped Hessian is not positive definite") from exc
    inverse_factor = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    _synchronize(h.device)
    decomposition_seconds = time.perf_counter() - decomposition_started

    _synchronize(h.device)
    compensation_started = time.perf_counter()
    output = torch.zeros_like(work)
    for block_start in range(0, work.shape[1], blocksize):
        block_end = min(block_start + blocksize, work.shape[1])
        block = work[:, block_start:block_end].clone()
        quantized = torch.zeros_like(block)
        errors = torch.zeros_like(block)
        local_factor = inverse_factor[block_start:block_end, block_start:block_end]
        local_keep = fixed_keep[:, block_start:block_end]
        for local_column in range(block_end - block_start):
            current = block[:, local_column]
            projected = current.clone()
            projected[~local_keep[:, local_column]] = 0
            quantized[:, local_column] = projected
            error = (current - projected) / local_factor[local_column, local_column]
            block[:, local_column:] -= error[:, None] @ local_factor[local_column, local_column:][None, :]
            errors[:, local_column] = error
        output[:, block_start:block_end] = quantized
        work[:, block_end:] -= errors @ inverse_factor[block_start:block_end, block_end:]
    output[~fixed_keep] = 0
    _synchronize(h.device)
    compensation_seconds = time.perf_counter() - compensation_started
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Fixed-mask OBS produced NaN/Inf")
    if torch.count_nonzero(output[~fixed_keep]):
        raise RuntimeError("Fixed-mask OBS regrew a deleted coordinate")
    survivor_changed = int(torch.count_nonzero(output[fixed_keep] != original[fixed_keep]))
    return FixedMaskOBSResult(
        output.to(weight.dtype),
        fixed_keep,
        float(damp_tensor),
        decomposition_seconds,
        compensation_seconds,
        survivor_changed,
    )


class FixedMaskOBSAdapter:
    """Collect Conv/Linear Hessians and compensate an externally fixed mask."""

    def __init__(self, layer: nn.Linear | nn.Conv2d, *, compute_dtype: torch.dtype = torch.float64):
        if not isinstance(layer, (nn.Linear, nn.Conv2d)):
            raise TypeError("FixedMaskOBSAdapter supports Linear and Conv2d")
        if compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("compute_dtype must be torch.float32 or torch.float64")
        self.layer = layer
        self.compute_dtype = compute_dtype
        self.groups = layer.groups if isinstance(layer, nn.Conv2d) else 1
        features = layer.weight[0].numel()
        self.xtx = torch.zeros(self.groups, features, features, dtype=compute_dtype, device=layer.weight.device)
        self.vectors = 0
        self._gpu_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_hessian_seconds = 0.0

    @torch.no_grad()
    def add_batch(self, inputs: Tensor) -> None:
        if isinstance(self.layer, nn.Linear):
            matrices = (inputs.reshape(-1, inputs.shape[-1]).T,)
        else:
            matrices = grouped_patch_matrices(self.layer, inputs)
        count = matrices[0].shape[1]
        if any(matrix.shape[1] != count for matrix in matrices):
            raise RuntimeError("Grouped patch counts differ")
        for group, matrix in enumerate(matrices):
            x = matrix.to(self.compute_dtype)
            if x.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                self.xtx[group] += 2 * (x @ x.T)
                end.record()
                self._gpu_events.append((start, end))
            else:
                started = time.perf_counter()
                self.xtx[group] += 2 * (x @ x.T)
                self._cpu_hessian_seconds += time.perf_counter() - started
        self.vectors += count

    def hessians(self) -> Tensor:
        if self.vectors == 0:
            raise RuntimeError("No calibration inputs were collected")
        return self.xtx / self.vectors

    def hessian_build_seconds(self) -> float:
        if self.layer.weight.device.type != "cuda":
            return self._cpu_hessian_seconds
        _synchronize(self.layer.weight.device)
        return sum(start.elapsed_time(end) for start, end in self._gpu_events) / 1000.0

    @torch.no_grad()
    def prune(
        self,
        keep_mask: Tensor,
        *,
        keep_n: int = 2,
        m: int = 4,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> tuple[Tensor, tuple[FixedMaskOBSResult, ...]]:
        matrix = self.layer.weight if isinstance(self.layer, nn.Linear) else weight_to_matrix(self.layer.weight)
        matrix_keep = keep_mask.reshape_as(matrix).to(matrix.device)
        validate_keep_mask(matrix_keep, keep_n, m)
        rows_per_group = matrix.shape[0] // self.groups
        results: list[FixedMaskOBSResult] = []
        for group, hessian in enumerate(self.hessians()):
            rows = slice(group * rows_per_group, (group + 1) * rows_per_group)
            result = fixedmask_obs_matrix(
                matrix[rows],
                hessian,
                matrix_keep[rows],
                keep_n=keep_n,
                m=m,
                blocksize=blocksize,
                percdamp=percdamp,
                compute_dtype=self.compute_dtype,
            )
            matrix[rows].copy_(result.weight)
            if not torch.equal(result.keep_mask, matrix_keep[rows]):
                raise RuntimeError("Fixed support changed during OBS compensation")
            results.append(result)
        return matrix_keep.reshape_as(self.layer.weight).detach().clone(), tuple(results)
