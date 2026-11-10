"""Paper-faithful Wanda scoring with an exact Conv2d activation mapping."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from baseline_adapters.common.conv2d import grouped_patch_matrices, weight_to_matrix
from baseline_adapters.common.nm import keep_mask_from_prune_metric, validate_keep_mask


class WandaAdapter:
    """Collect ||X_j||_2 and prune |W_ij|*||X_j||_2 per output row.

    Retained weights are never updated. Grouped convolutions maintain one
    statistic vector per convolution group.
    """

    def __init__(self, layer: nn.Linear | nn.Conv2d):
        if not isinstance(layer, (nn.Linear, nn.Conv2d)):
            raise TypeError("WandaAdapter supports Linear and Conv2d")
        self.layer = layer
        self.groups = layer.groups if isinstance(layer, nn.Conv2d) else 1
        features = layer.weight[0].numel()
        self.input_sumsq = torch.zeros(self.groups, features, dtype=torch.float64, device=layer.weight.device)
        self.vectors = 0

    @torch.no_grad()
    def add_batch(self, inputs: Tensor) -> None:
        if isinstance(self.layer, nn.Linear):
            matrix = inputs.reshape(-1, inputs.shape[-1]).T
            matrices = (matrix,)
        else:
            matrices = grouped_patch_matrices(self.layer, inputs)
        count = matrices[0].shape[1]
        if any(matrix.shape[1] != count for matrix in matrices):
            raise RuntimeError("Grouped patch counts differ")
        for group, matrix in enumerate(matrices):
            self.input_sumsq[group] += matrix.to(torch.float64).square().sum(dim=1)
        self.vectors += count

    def scores(self) -> Tensor:
        if self.vectors == 0:
            raise RuntimeError("No calibration inputs were collected")
        weights = self.layer.weight if isinstance(self.layer, nn.Linear) else weight_to_matrix(self.layer.weight)
        rows_per_group = weights.shape[0] // self.groups
        score = torch.empty_like(weights, dtype=torch.float64)
        for group in range(self.groups):
            rows = slice(group * rows_per_group, (group + 1) * rows_per_group)
            score[rows] = weights[rows].detach().abs().to(torch.float64) * self.input_sumsq[group].sqrt()[None, :]
        return score

    @torch.no_grad()
    def prune(self, keep_n: int = 2, m: int = 4) -> Tensor:
        matrix = self.layer.weight if isinstance(self.layer, nn.Linear) else weight_to_matrix(self.layer.weight)
        keep = keep_mask_from_prune_metric(self.scores(), keep_n, m)
        validate_keep_mask(keep, keep_n, m)
        original = matrix.detach().clone()
        matrix.mul_(keep.to(matrix.device))
        if not torch.equal(matrix[keep], original[keep]):
            raise RuntimeError("Wanda changed retained weights")
        return keep.reshape_as(self.layer.weight)
