"""SparseGPT core update adapted to exact Linear/Conv2d local matrices."""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor, nn

from .conv2d import grouped_patch_matrices, weight_to_matrix
from .nm import validate_keep_mask


@dataclass(frozen=True)
class SparseGPTResult:
    weight: Tensor
    keep_mask: Tensor
    hessian: Tensor
    damp: float
    decomposition_seconds: float
    compensation_seconds: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sparsegpt_matrix(
    weight: Tensor,
    hessian: Tensor,
    *,
    keep_n: int = 2,
    m: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float64,
) -> SparseGPTResult:
    """Run the official sequential OBS update with a dense incomplete tail.

    The official ``prunen`` is ``m-keep_n``. Masks are selected from the
    compensated current block, as in pinned SparseGPT, not precomputed from
    the dense weight.
    """
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Weight/Hessian shapes are inconsistent")
    if not 0 < keep_n < m or blocksize % m:
        raise ValueError("Require 0 < keep_n < m and blocksize divisible by m")
    if percdamp < 0:
        raise ValueError("percdamp must be non-negative")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be torch.float32 or torch.float64")
    w = weight.detach().to(compute_dtype).clone()
    h = hessian.detach().to(compute_dtype).clone()
    if not bool(torch.isfinite(h).all()):
        raise ValueError("Non-finite Hessian")
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
                metric = w1[:, local_i : local_i + width].square() / torch.diag(local_factor)[
                    local_i : local_i + width
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
        w[:, i2:] -= err1 @ hinv_factor[i1:i2, i2:]
    validate_keep_mask(keep, keep_n, m)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("SparseGPT produced NaN/Inf")
    _synchronize(h.device)
    compensation_seconds = time.perf_counter() - compensation_started
    return SparseGPTResult(
        output.to(weight.dtype), keep, hessian.detach().clone(), float(damp_tensor),
        decomposition_seconds, compensation_seconds,
    )


class SparseGPTAdapter:
    def __init__(self, layer: nn.Linear | nn.Conv2d, *, compute_dtype: torch.dtype = torch.float64):
        if not isinstance(layer, (nn.Linear, nn.Conv2d)):
            raise TypeError("SparseGPTAdapter supports Linear and Conv2d")
        self.layer = layer
        if compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("compute_dtype must be torch.float32 or torch.float64")
        self.compute_dtype = compute_dtype
        self.groups = layer.groups if isinstance(layer, nn.Conv2d) else 1
        features = layer.weight[0].numel()
        self.xtx = torch.zeros(self.groups, features, features, dtype=compute_dtype, device=layer.weight.device)
        self.vectors = 0

    @torch.no_grad()
    def add_batch(self, inputs: Tensor) -> None:
        if isinstance(self.layer, nn.Linear):
            matrices = (inputs.reshape(-1, inputs.shape[-1]).T,)
        else:
            matrices = grouped_patch_matrices(self.layer, inputs)
        count = matrices[0].shape[1]
        for group, matrix in enumerate(matrices):
            x = matrix.to(self.compute_dtype)
            self.xtx[group] += 2 * (x @ x.T)
        self.vectors += count

    def hessians(self) -> Tensor:
        if self.vectors == 0:
            raise RuntimeError("No calibration inputs were collected")
        return self.xtx / self.vectors

    @torch.no_grad()
    def prune(self, keep_n: int = 2, m: int = 4, blocksize: int = 128, percdamp: float = 0.01) -> Tensor:
        matrix = self.layer.weight if isinstance(self.layer, nn.Linear) else weight_to_matrix(self.layer.weight)
        rows_per_group = matrix.shape[0] // self.groups
        all_keep = torch.empty_like(matrix, dtype=torch.bool)
        for group, hessian in enumerate(self.hessians()):
            rows = slice(group * rows_per_group, (group + 1) * rows_per_group)
            result = sparsegpt_matrix(
                matrix[rows], hessian, keep_n=keep_n, m=m, blocksize=blocksize,
                percdamp=percdamp, compute_dtype=self.compute_dtype,
            )
            matrix[rows].copy_(result.weight)
            all_keep[rows] = result.keep_mask
        return all_keep.reshape_as(self.layer.weight)
