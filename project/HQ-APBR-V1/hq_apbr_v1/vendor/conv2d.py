"""Exact OIHW <-> local-linear mapping for torch.nn.Conv2d.

PyTorch's unfold order is channel-major then kernel-row/kernel-column, exactly
the order produced by ``weight.reshape(out_channels, -1)``.  Grouped
convolutions are represented by one independent matrix multiplication per
group; combining their input covariances would be mathematically incorrect.
"""
from __future__ import annotations

from collections.abc import Iterator
from math import floor

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def output_hw(layer: nn.Conv2d, input_hw: tuple[int, int]) -> tuple[int, int]:
    h, w = input_hw
    kh, kw = _pair(layer.kernel_size)
    sh, sw = _pair(layer.stride)
    ph, pw = _pair(layer.padding)
    dh, dw = _pair(layer.dilation)
    return (
        floor((h + 2 * ph - dh * (kh - 1) - 1) / sh + 1),
        floor((w + 2 * pw - dw * (kw - 1) - 1) / sw + 1),
    )


def weight_to_matrix(weight: Tensor) -> Tensor:
    if weight.ndim != 4:
        raise ValueError("Conv2d weight must be OIHW")
    return weight.reshape(weight.shape[0], -1)


def matrix_to_weight(matrix: Tensor, shape: torch.Size | tuple[int, ...]) -> Tensor:
    if len(shape) != 4 or matrix.shape != (shape[0], shape[1] * shape[2] * shape[3]):
        raise ValueError("Matrix and OIHW shape are inconsistent")
    return matrix.reshape(shape)


def grouped_patch_matrices(layer: nn.Conv2d, inputs: Tensor) -> tuple[Tensor, ...]:
    """Return G matrices of shape [Cin/G*kH*kW, N*L]."""
    if inputs.ndim != 4 or inputs.shape[1] != layer.in_channels:
        raise ValueError("Expected NCHW input matching layer.in_channels")
    patches = F.unfold(
        inputs,
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=layer.padding,
        stride=layer.stride,
    )
    n, _, locations = patches.shape
    features = (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
    patches = patches.reshape(n, layer.groups, features, locations)
    return tuple(patches[:, group].permute(1, 0, 2).reshape(features, n * locations) for group in range(layer.groups))


def iter_grouped_patch_chunks(
    layer: nn.Conv2d, inputs: Tensor, max_vectors: int
) -> Iterator[tuple[Tensor, ...]]:
    """Yield column views of unfolded patches.

    This bounds downstream matrix/Hessian temporaries. ``F.unfold`` itself is
    still materialized once; CIFAR-32 makes that allocation modest. A large
    image production runner must spatially tile before unfold.
    """
    if max_vectors <= 0:
        raise ValueError("max_vectors must be positive")
    groups = grouped_patch_matrices(layer, inputs)
    total = groups[0].shape[1]
    for start in range(0, total, max_vectors):
        yield tuple(matrix[:, start : start + max_vectors] for matrix in groups)


def conv2d_via_unfold(layer: nn.Conv2d, inputs: Tensor, max_vectors: int = 4096) -> Tensor:
    """Evaluate Conv2d through the exact grouped local-linear mapping."""
    matrices = grouped_patch_matrices(layer, inputs)
    weights = weight_to_matrix(layer.weight)
    rows_per_group = layer.out_channels // layer.groups
    outputs: list[Tensor] = []
    for group, patches in enumerate(matrices):
        rows = weights[group * rows_per_group : (group + 1) * rows_per_group]
        pieces = [rows @ patches[:, start : start + max_vectors] for start in range(0, patches.shape[1], max_vectors)]
        outputs.append(torch.cat(pieces, dim=1))
    flat = torch.cat(outputs, dim=0)
    if layer.bias is not None:
        flat = flat + layer.bias[:, None]
    oh, ow = output_hw(layer, (inputs.shape[-2], inputs.shape[-1]))
    return flat.reshape(layer.out_channels, inputs.shape[0], oh, ow).permute(1, 0, 2, 3).contiguous()
