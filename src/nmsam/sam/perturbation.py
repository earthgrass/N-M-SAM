"""One global L2 norm across all allowed gradient tensors."""
import math
from typing import Protocol
import torch
from nmsam.core.step_types import TensorTuple


class PerturbationPolicy(Protocol):
    def directions(self, gradients: TensorTuple) -> TensorTuple: ...


class DensePolicy:
    def directions(self, gradients: TensorTuple) -> TensorTuple:
        return gradients


def global_norm(tensors: TensorTuple) -> torch.Tensor:
    if not tensors:
        raise ValueError("empty gradient vector")
    return torch.sqrt(sum(t.square().sum() for t in tensors))


def perturbation(gradients: TensorTuple, rho: float,
                 policy: PerturbationPolicy) -> tuple[TensorTuple, bool]:
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    directions = policy.directions(gradients)
    if len(directions) != len(gradients) or any(
        d.shape != g.shape or d.dtype != g.dtype or d.device != g.device
        for d, g in zip(directions, gradients)
    ):
        raise ValueError("policy must preserve gradient structure, dtype and device")
    norm = global_norm(directions)
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite direction norm")
    zero = norm.item() == 0
    if zero:
        return tuple(torch.zeros_like(g) for g in directions), True
    return tuple((g / norm) * rho for g in directions), False


def clip_by_global_norm(gradients: TensorTuple, threshold: float) -> TensorTuple:
    """Official threshold=0 disables clipping, no additive epsilon denominator."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("clip threshold must be finite and nonnegative")
    norm = global_norm(gradients)
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite gradient norm")
    if threshold > 0 and norm.item() >= threshold:
        return tuple((g / norm) * threshold for g in gradients)
    return gradients

