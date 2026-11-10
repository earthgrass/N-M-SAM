"""One-shot preregistered lambda_p calibration with FP64 accumulation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class LambdaCalibration:
    lambda_p: float
    ce_gradient_norm: float
    prune_gradient_norm: float
    achieved_ratio: float
    target_ratio: float
    lower_bound: float
    upper_bound: float
    eligible_parameter_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_gradient_sets(
    ce_gradients: Sequence[Tensor | None],
    prune_gradients: Sequence[Tensor | None],
    parameter_names: Sequence[str],
) -> None:
    if len(ce_gradients) != len(prune_gradients) or len(ce_gradients) != len(parameter_names):
        raise ValueError("CE/pruning eligible parameter sets differ")
    if len(set(parameter_names)) != len(parameter_names) or not parameter_names:
        raise ValueError("Eligible parameter names must be unique and nonempty")
    for name, ce, prune in zip(parameter_names, ce_gradients, prune_gradients):
        if ce is None or prune is None:
            raise ValueError(f"Missing eligible gradient: {name}")
        if ce.shape != prune.shape:
            raise ValueError(f"Eligible gradient shape mismatch: {name}")
        if not bool(torch.isfinite(ce).all() and torch.isfinite(prune).all()):
            raise FloatingPointError(f"Non-finite eligible gradient: {name}")


def _global_norm(values: Iterable[Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for value in values:
        total += value.detach().cpu().double().square().sum()
    result = float(total.sqrt())
    if not math.isfinite(result):
        raise FloatingPointError("Non-finite global gradient norm")
    return result


def calibrate_lambda_p(
    ce_gradients: Sequence[Tensor | None],
    prune_gradients: Sequence[Tensor | None],
    parameter_names: Sequence[str],
    *,
    target_ratio: float = 0.05,
    lower_bound: float = 1.0e-6,
    upper_bound: float = 1.0e6,
    ratio_tolerance: float = 5.0e-5,
) -> LambdaCalibration:
    """Calibrate once; never clips or silently substitutes a boundary."""
    _validate_gradient_sets(ce_gradients, prune_gradients, parameter_names)
    if not 0 < target_ratio < 1 or not 0 < lower_bound <= upper_bound:
        raise ValueError("Invalid preregistered lambda calibration bounds")
    ce_norm = _global_norm(value for value in ce_gradients if value is not None)
    prune_norm = _global_norm(value for value in prune_gradients if value is not None)
    if ce_norm == 0:
        raise RuntimeError("CE gradient norm is zero")
    if prune_norm == 0:
        raise RuntimeError("Pruning gradient norm is zero")
    lambda_p = target_ratio * ce_norm / prune_norm
    if not lower_bound <= lambda_p <= upper_bound:
        raise RuntimeError(
            f"Calibrated lambda_p {lambda_p:.17g} is outside [{lower_bound:.17g}, {upper_bound:.17g}]"
        )
    achieved = lambda_p * prune_norm / ce_norm
    if abs(achieved - target_ratio) > ratio_tolerance:
        raise RuntimeError("lambda_p gradient-ratio acceptance failed")
    return LambdaCalibration(
        lambda_p, ce_norm, prune_norm, achieved, target_ratio, lower_bound,
        upper_bound, tuple(parameter_names),
    )
