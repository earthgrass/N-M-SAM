"""Controlled paper-faithful ROSE adapter for unified N:M groups.

Pinned ROSE commit 88d922b has an undefined ``reorder_column`` in its N:M
branch and uses max rather than Equation (7)'s mean denominator. This module is
therefore an independent repair, not an unmodified official reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.sparsegpt.adapter import SparseGPTAdapter, sparsegpt_matrix


@dataclass(frozen=True)
class ROSEOrder:
    permutation: Tensor
    inverse: Tensor
    block_loss: Tensor
    relative_range: float
    reordered: bool


def rose_permutation(weight: Tensor, activation_norm: Tensor, *, keep_n: int = 2, m: int = 4, threshold: float = 0.5) -> ROSEOrder:
    if weight.ndim != 2 or activation_norm.shape != (weight.shape[1],):
        raise ValueError("Weight and activation norm shapes are inconsistent")
    end = weight.shape[1] // m * m
    if end == 0:
        identity = torch.arange(weight.shape[1], device=weight.device)
        return ROSEOrder(identity, identity, torch.empty(0, device=weight.device), 0.0, False)
    score = weight.detach().abs().to(torch.float64) * activation_norm.to(torch.float64)[None, :]
    blocks = score[:, :end].reshape(weight.shape[0], -1, m)
    removed = blocks.argsort(dim=-1, stable=True)[..., : m - keep_n]
    candidate = torch.zeros_like(blocks, dtype=torch.bool)
    candidate.scatter_(-1, removed, True)
    loss = blocks * candidate
    block_loss = loss.sum(dim=(0, 2))
    mean = block_loss.mean()
    relative = 0.0 if float(mean) == 0.0 else float((block_loss.max() - block_loss.min()) / mean)
    if relative <= threshold:
        permutation = torch.arange(weight.shape[1], device=weight.device)
        return ROSEOrder(permutation, permutation, block_loss, relative, False)
    column_loss = loss.sum(dim=0)
    within = column_loss.argsort(dim=1, descending=True, stable=True)
    group_order = block_loss.argsort(descending=True, stable=True)
    bases = group_order[:, None] * m
    complete = (bases + within[group_order]).reshape(-1)
    tail = torch.arange(end, weight.shape[1], device=weight.device)
    permutation = torch.cat((complete, tail))
    inverse = permutation.argsort(stable=True)
    return ROSEOrder(permutation, inverse, block_loss, relative, True)


class ROSEAdapter(SparseGPTAdapter):
    @torch.no_grad()
    def prune(
        self,
        keep_n: int = 2,
        m: int = 4,
        blocksize: int = 128,
        percdamp: float = 0.01,
        threshold: float = 0.5,
    ) -> tuple[Tensor, tuple[ROSEOrder, ...]]:
        matrix = self.layer.weight if isinstance(self.layer, nn.Linear) else weight_to_matrix(self.layer.weight)
        rows_per_group = matrix.shape[0] // self.groups
        keep = torch.empty_like(matrix, dtype=torch.bool)
        orders: list[ROSEOrder] = []
        # sqrt(sum x^2) is the paper's ||X_j||_2; a global normalization would
        # cancel in ordering and in the relative-range ratio.
        norms = self.xtx.diagonal(dim1=1, dim2=2).clamp_min(0).sqrt()
        for group, hessian in enumerate(self.hessians()):
            rows = slice(group * rows_per_group, (group + 1) * rows_per_group)
            order = rose_permutation(matrix[rows], norms[group], keep_n=keep_n, m=m, threshold=threshold)
            perm = order.permutation
            result = sparsegpt_matrix(
                matrix[rows][:, perm],
                hessian[perm][:, perm],
                keep_n=keep_n,
                m=m,
                blocksize=m if order.reordered else blocksize,
                percdamp=percdamp,
                compute_dtype=self.compute_dtype,
            )
            matrix[rows].copy_(result.weight[:, order.inverse])
            keep[rows] = result.keep_mask[:, order.inverse]
            orders.append(order)
        return keep.reshape_as(self.layer.weight), tuple(orders)
