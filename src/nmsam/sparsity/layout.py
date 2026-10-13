"""Reversible per-output grouping for Linear and Conv2d weights."""
from dataclasses import dataclass
from math import prod
import torch


@dataclass(frozen=True)
class MaskSpec:
    n: int = 2
    m: int = 4

    def __post_init__(self):
        if type(self.n) is not int or type(self.m) is not int or not 0 < self.n <= self.m:
            raise ValueError("integer 0 < N <= M required")


@dataclass(frozen=True)
class MaskLayout:
    shape: tuple[int, ...]
    spec: MaskSpec = MaskSpec()

    def __post_init__(self):
        if len(self.shape) not in (2, 4) or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError("nonempty Linear [out,in] or Conv2d [out,in,kh,kw] weight required")
        if self.row_width % self.spec.m:
            raise ValueError("incomplete N:M tail: reject entire target layer")

    @property
    def row_width(self) -> int:
        return prod(self.shape[1:])

    @property
    def group_shape(self) -> tuple[int, int, int]:
        return self.shape[0], self.row_width // self.spec.m, self.spec.m

    def groups(self, weights: torch.Tensor) -> torch.Tensor:
        if tuple(weights.shape) != self.shape:
            raise ValueError("tensor shape differs from layout")
        # reshape respects logical C-order even for noncontiguous input.
        return weights.reshape(self.group_shape)

    def restore(self, groups: torch.Tensor) -> torch.Tensor:
        if tuple(groups.shape) != self.group_shape:
            raise ValueError("group tensor shape differs from layout")
        return groups.reshape(self.shape)

