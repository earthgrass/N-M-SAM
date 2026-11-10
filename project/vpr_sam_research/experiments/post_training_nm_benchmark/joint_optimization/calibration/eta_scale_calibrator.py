"""One-shot eta_s calibration under the global APBR log-volume constraint."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence

import torch
from torch import Tensor


Projection = Callable[[Sequence[Tensor]], list[Tensor]]


@dataclass(frozen=True)
class EtaCalibration:
    eta_s: float
    lambda_p: float
    energy_update_norm: float
    independent_prune_update_norm: float
    achieved_ratio: float
    target_ratio: float
    cosine_similarity: float
    combined_update_norm: float
    projection: str
    iterations: int
    analytic: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _same_structure(left: Sequence[Tensor], right: Sequence[Tensor], label: str) -> None:
    if not left or len(left) != len(right):
        raise ValueError(f"Scale structure mismatch: {label}")
    for a, b in zip(left, right):
        if a.shape != b.shape:
            raise ValueError(f"Scale shape mismatch: {label}")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise FloatingPointError(f"Non-finite scale state: {label}")


def _dot(left: Sequence[Tensor], right: Sequence[Tensor]) -> Tensor:
    return sum((a.detach().cpu().double() * b.detach().cpu().double()).sum()
               for a, b in zip(left, right))


def _norm(values: Sequence[Tensor]) -> float:
    return float(_dot(values, values).sqrt())


def _subtract(left: Sequence[Tensor], right: Sequence[Tensor]) -> list[Tensor]:
    return [a - b for a, b in zip(left, right)]


def affine_zero_volume_projection(values: Sequence[Tensor]) -> list[Tensor]:
    """Orthogonal affine projection onto sum_i s_i = 0 in global space."""
    if not values:
        raise ValueError("No APBR scale coordinates")
    total = sum(value.double().sum() for value in values)
    count = sum(value.numel() for value in values)
    mean = total / count
    result = [(value.double() - mean).to(value.dtype) for value in values]
    residual = float(sum(value.double().sum() for value in result))
    tolerance = 32 * torch.finfo(torch.float64).eps * max(
        1.0, sum(float(value.double().abs().sum()) for value in result)
    )
    if abs(residual) > tolerance:
        raise FloatingPointError("Affine log-volume projection residual")
    return result


def _independent_update(
    scales: Sequence[Tensor], scale_gradients: Sequence[Tensor], eta: float,
    lambda_p: float, projection: Projection,
) -> list[Tensor]:
    projected = projection([
        scale.double() - float(eta) * float(lambda_p) * gradient.detach().double()
        for scale, gradient in zip(scales, scale_gradients)
    ])
    return _subtract(projected, scales)


def combine_scale_updates(
    scales: Sequence[Tensor], energy_raw: Sequence[Tensor],
    scale_gradients: Sequence[Tensor], *, eta_s: float, lambda_p: float,
    projection: Projection = affine_zero_volume_projection,
) -> tuple[list[Tensor], list[Tensor]]:
    _same_structure(scales, energy_raw, "energy")
    _same_structure(scales, scale_gradients, "prune")
    proposed = [s.double() + d.detach().double() - eta_s * lambda_p * g.detach().double()
                for s, d, g in zip(scales, energy_raw, scale_gradients)]
    updated = projection(proposed)
    delta = _subtract(updated, scales)
    if not all(bool(torch.isfinite(value).all()) for value in updated):
        raise FloatingPointError("Non-finite combined scale update")
    return updated, delta


def calibrate_eta_s(
    scales: Sequence[Tensor],
    energy_raw: Sequence[Tensor],
    scale_gradients: Sequence[Tensor],
    *,
    lambda_p: float,
    target_ratio: float = 0.05,
    projection: Projection = affine_zero_volume_projection,
    projection_name: str = "global_affine_zero_volume",
    affine: bool = True,
    relative_tolerance: float = 1.0e-3,
    max_expand: int = 80,
    max_bisection: int = 100,
) -> EtaCalibration:
    """Calibrate eta analytically for affine projection or by bracket/bisection."""
    _same_structure(scales, energy_raw, "energy")
    _same_structure(scales, scale_gradients, "prune")
    if not math.isfinite(lambda_p) or lambda_p <= 0 or not 0 < target_ratio < 1:
        raise ValueError("Invalid eta calibration constants")
    projected_energy_state = projection([s.double() + d.detach().double()
                                         for s, d in zip(scales, energy_raw)])
    energy = _subtract(projected_energy_state, scales)
    energy_norm = _norm(energy)
    if energy_norm == 0:
        raise RuntimeError("Original APBR energy update is zero")
    target = target_ratio * energy_norm

    unit = _independent_update(scales, scale_gradients, 1.0, lambda_p, projection)
    unit_norm = _norm(unit)
    if unit_norm == 0:
        raise RuntimeError("Projected pruning scale gradient is zero")
    iterations = 0
    if affine:
        eta = target / unit_norm
    else:
        low, high = 0.0, 1.0
        previous = 0.0
        for _ in range(max_expand):
            current = _norm(_independent_update(scales, scale_gradients, high, lambda_p, projection))
            iterations += 1
            if current + 1e-15 < previous:
                raise RuntimeError("Projected update norm is not monotone in eta")
            if current >= target:
                break
            previous, high = current, high * 2
        else:
            raise RuntimeError("Target pruning scale update is unreachable")
        for _ in range(max_bisection):
            eta = 0.5 * (low + high)
            current = _norm(_independent_update(scales, scale_gradients, eta, lambda_p, projection))
            iterations += 1
            if abs(current - target) / target <= relative_tolerance:
                break
            if current < target:
                low = eta
            else:
                high = eta
        else:
            raise RuntimeError("eta_s bisection did not converge")
    prune = _independent_update(scales, scale_gradients, eta, lambda_p, projection)
    prune_norm = _norm(prune)
    achieved = prune_norm / energy_norm
    if abs(achieved - target_ratio) / target_ratio > relative_tolerance:
        raise RuntimeError("eta_s projected-norm acceptance failed")
    cosine = float(_dot(energy, prune) / (energy_norm * prune_norm))
    _updated, combined = combine_scale_updates(
        scales, energy_raw, scale_gradients, eta_s=eta, lambda_p=lambda_p,
        projection=projection,
    )
    return EtaCalibration(
        float(eta), float(lambda_p), energy_norm, prune_norm, achieved,
        target_ratio, cosine, _norm(combined), projection_name, iterations, affine,
    )
