"""Independent HQ-APBR mathematical reference for CPU numerical gates.

No function here mutates a model, optimizer, formal adapter, or checkpoint.
The inverse diagonal routine is exact and intended for bounded matrices; it
does not turn a 4608-dimensional factorization into a cheap operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import torch
from torch import Tensor

@dataclass(frozen=True)
class GeometrySpec:
    """A Conv2d weight and shared inverse diagonal in OIHW/unfold order.

    `q_diag` is [groups, input_features_per_group], not one Hessian per row.
    Output rows in each connection group receive the same feature target.
    """

    name: str
    weight_shape: tuple[int, int, int, int]
    groups: int
    q_diag: Tensor

    @property
    def output_rows(self) -> int:
        return self.weight_shape[0]

    @property
    def features(self) -> int:
        return math.prod(self.weight_shape[1:])

    def validate(self) -> None:
        if (not self.name.endswith(".weight") or self.groups < 1
                or len(self.weight_shape) != 4 or any(d < 1 for d in self.weight_shape)):
            raise ValueError("Expected named Conv2d weight and positive groups")
        if self.output_rows < 1 or self.output_rows % self.groups:
            raise ValueError("Output rows must partition evenly into groups")
        if self.q_diag.ndim != 2 or self.q_diag.shape != (self.groups, self.features):
            raise ValueError("Inverse diagonal does not match Conv2d connections")
        if not bool(torch.isfinite(self.q_diag).all() and (self.q_diag > 0).all()):
            raise ValueError("Inverse diagonal must be positive and finite")


def inverse_diagonal_cholesky(
    hessian: Tensor, *, percdamp: float = 0.01, dead_policy: str = "reject"
) -> tuple[Tensor, Tensor, float]:
    """Return diag((H+delta I)^-1), damped H, and delta.

    A near-symmetric Gram is explicitly averaged with its transpose; a
    materially asymmetric matrix is rejected. The returned H records the
    exact matrix used by Cholesky. `sparsegpt` dead policy reproduces the existing adapter's zero-diagonal
    replacement before relative damping. `reject` retains the stated H.
    Neither policy silently changes damping or falls back after failure.
    """
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or not hessian.shape[0]:
        raise ValueError("Hessian must be a nonempty square matrix")
    if not math.isfinite(percdamp) or percdamp < 0:
        raise ValueError("Invalid relative damping")
    if dead_policy not in {"reject", "sparsegpt"}:
        raise ValueError("Unknown dead-coordinate policy")
    if hessian.dtype not in {torch.float32, torch.float64}:
        raise ValueError("Hessian must be FP32 or FP64")
    if not bool(torch.isfinite(hessian).all()):
        raise ValueError("Nonfinite Hessian")
    h = hessian.detach().clone()
    # Matmul may introduce roundoff asymmetry. Cholesky reads one triangle,
    # so return the explicitly symmetrized matrix it actually factorizes.
    if not bool(torch.allclose(h, h.T, rtol=1e-6, atol=1e-8)):
        raise ValueError("Hessian is materially asymmetric")
    h = .5 * h + .5 * h.T
    if dead_policy == "sparsegpt":
        dead = torch.diagonal(h) == 0
        indices = torch.arange(h.shape[0], device=h.device)[dead]
        h[indices, indices] = 1
    delta = percdamp * torch.diagonal(h).mean()
    if not bool(torch.isfinite(delta)):
        raise FloatingPointError("Nonfinite damping")
    indices = torch.arange(h.shape[0], device=h.device)
    h[indices, indices] += delta
    lower = torch.linalg.cholesky(h)
    identity = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    inverse_lower = torch.linalg.solve_triangular(lower, identity, upper=False)
    q_diag = inverse_lower.square().sum(dim=0)
    if not bool(torch.isfinite(q_diag).all() and (q_diag > 0).all()):
        raise FloatingPointError("Invalid inverse diagonal")
    return q_diag, h, float(delta)


def obs_singleton_cost(weight: Tensor, inverse_diagonal: Tensor) -> Tensor:
    """Damped quadratic deletion cost for one active coordinate."""
    if weight.shape != inverse_diagonal.shape:
        raise ValueError("Weight and inverse diagonal shape mismatch")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(inverse_diagonal).all()
                and (inverse_diagonal > 0).all()):
        raise ValueError("Invalid OBS inputs")
    cost = weight.to(torch.float64).square() / (2 * inverse_diagonal.to(torch.float64))
    if not bool(torch.isfinite(cost).all()):
        raise FloatingPointError("OBS cost overflow")
    return cost


def target_from_inverse_diagonals(
    specs: list[GeometrySpec],
) -> tuple[dict[str, Tensor], Tensor, int]:
    """Build globally centered 0.5 log(Qii) per eligible weight coordinate.

    Centering weights each shared input feature by its number of connected
    output rows. Merely averaging layer or group means would be incorrect.
    """
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Require unique nonempty eligible layer list")
    devices = {spec.q_diag.device for spec in specs}
    if len(devices) != 1:
        raise ValueError("All inverse diagonals must share a device")
    total = torch.zeros((), dtype=torch.float64, device=specs[0].q_diag.device)
    coordinates = 0
    logs: dict[str, Tensor] = {}
    for spec in specs:
        spec.validate()
        u = .5 * torch.log(spec.q_diag.to(torch.float64))
        logs[spec.name] = u
        rows_per_group = spec.output_rows // spec.groups
        total += rows_per_group * u.sum()
        coordinates += spec.output_rows * spec.features
    center = total / coordinates
    targets: dict[str, Tensor] = {}
    for spec in specs:
        rows_per_group = spec.output_rows // spec.groups
        target = (logs[spec.name] - center).repeat_interleave(rows_per_group, dim=0)
        targets[spec.name] = target.reshape(spec.weight_shape)
    return targets, center, coordinates


def historical_geometry_update(
    previous: Mapping[str, Tensor], target: Mapping[str, Tensor], eta: float
) -> dict[str, Tensor]:
    """One immutable EMA step; eta=0 is the SAM control, eta=1 Direct-Q."""
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be in [0, 1]")
    if previous.keys() != target.keys():
        raise ValueError("Geometry key mismatch")
    updated = {}
    for label, mapping in (("previous", previous), ("target", target)):
        values = [value.to(torch.float64).reshape(-1) for value in mapping.values()]
        if not values:
            raise ValueError("Require nonempty geometry")
        total = sum(value.sum() for value in values)
        absolute = sum(value.abs().sum() for value in values)
        if not bool(torch.isfinite(total) and torch.isfinite(absolute)):
            raise ValueError(f"Nonfinite {label} geometry")
        if float(total.abs()) > 1e-12 * max(1.0, float(absolute)):
            raise ValueError(f"{label} geometry violates global zero-sum constraint")
    for name, value in target.items():
        old = previous[name]
        if old.shape != value.shape or old.device != value.device:
            raise ValueError(f"Geometry tensor mapping mismatch: {name}")
        if not bool(torch.isfinite(old).all() and torch.isfinite(value).all()):
            raise ValueError("Nonfinite geometry")
        candidate = (1 - eta) * old.to(torch.float64) + eta * value.to(torch.float64)
        if not bool(torch.isfinite(candidate).all()):
            raise FloatingPointError("Geometry update overflow")
        updated[name] = candidate
    return updated


def hq_global_l2_perturbation(
    parameters: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor | None],
    geometry: Mapping[str, Tensor],
    rho: float,
    *,
    radius_rtol: float = 2e-5,
) -> tuple[dict[str, Tensor], float, bool]:
    """Compute rho * exp(2s)g / ||exp(2s)g|| over all trainable params.

    A log-domain calculation avoids intermediate exp(2s) overflow. The
    returned tensors have parameter dtype, and their actual L2 norm is checked.
    Missing/None gradients contribute zero; noneligible coordinates have s=0.
    """
    if not parameters or not math.isfinite(rho) or rho <= 0:
        raise ValueError("Require positive rho and nonempty parameters")
    if not math.isfinite(radius_rtol) or radius_rtol < 0:
        raise ValueError("Invalid radius tolerance")
    if not geometry.keys() <= parameters.keys() or not gradients.keys() <= parameters.keys():
        raise ValueError("Unknown gradient or geometry parameter")
    exponents: dict[str, Tensor] = {}
    signs: dict[str, Tensor] = {}
    maxima = []
    for name, parameter in parameters.items():
        if parameter.dtype not in {torch.float32, torch.float64}:
            raise ValueError("Only FP32/FP64 trainable parameters supported")
        grad = gradients.get(name)
        g = torch.zeros_like(parameter) if grad is None else grad.detach()
        if g.shape != parameter.shape or g.device != parameter.device:
            raise ValueError(f"Gradient mapping mismatch: {name}")
        s = geometry.get(name)
        if s is not None and (s.shape != parameter.shape or s.device != parameter.device):
            raise ValueError(f"Geometry mapping mismatch: {name}")
        if not bool(torch.isfinite(g).all() and (s is None or torch.isfinite(s).all())):
            raise ValueError("Nonfinite gradient or geometry")
        logabs = g.to(torch.float64).abs().log()
        exponent = logabs if s is None else logabs + 2 * s.to(torch.float64)
        if bool(torch.isinf(exponent[g != 0]).any()):
            raise FloatingPointError("Scaled direction exponent overflow")
        exponents[name] = exponent
        signs[name] = g.to(torch.float64).sign()
        maxima.append(exponent.max())
    maximum = torch.stack(maxima).max()
    if bool(torch.isneginf(maximum)):
        return {name: torch.zeros_like(p) for name, p in parameters.items()}, 0.0, True
    norm = sum(torch.exp(2 * (value - maximum)).sum() for value in exponents.values()).sqrt()
    epsilon = {
        name: (rho * signs[name] * torch.exp(value - maximum) / norm).to(parameters[name].dtype)
        for name, value in exponents.items()
    }
    actual = float(sum(e.to(torch.float64).square().sum() for e in epsilon.values()).sqrt())
    if not math.isfinite(actual) or not math.isclose(actual, rho, rel_tol=radius_rtol, abs_tol=0):
        raise FloatingPointError(f"Global perturbation norm {actual} differs from rho {rho}")
    return epsilon, actual, False


def final_obs_keep_mask(weight_matrix: Tensor, q_diag: Tensor, *, groups: int = 1) -> Tensor:
    """Static final 2:4 Mask from |w|/sqrt(Qii), with stable ties and dense tail."""
    if weight_matrix.ndim != 2 or groups < 1 or weight_matrix.shape[0] % groups:
        raise ValueError("Invalid weight matrix/groups")
    q = q_diag.reshape(1, -1) if q_diag.ndim == 1 else q_diag
    if q.shape != (groups, weight_matrix.shape[1]):
        raise ValueError("Q diagonal does not match layer input features")
    if not bool(torch.isfinite(q).all() and (q > 0).all() and torch.isfinite(weight_matrix).all()):
        raise ValueError("Invalid final OBS score inputs")
    rows_per_group = weight_matrix.shape[0] // groups
    broadcast = q.repeat_interleave(rows_per_group, dim=0)
    # Log ranking is order-equivalent for positive scores and also handles
    # finite extreme inputs whose direct ratio would overflow to infinity.
    # log(0)=-inf is intentional and puts zero weights first for deletion.
    metric = weight_matrix.abs().to(torch.float64).log() - .5 * broadcast.to(torch.float64).log()
    rows, columns = metric.shape
    complete_end = columns // 4 * 4
    keep = torch.ones_like(metric, dtype=torch.bool)
    if complete_end:
        quartets = metric[:, :complete_end].reshape(rows, -1, 4)
        # The canonical project convention deletes lower-index coordinates
        # first on exact score ties and leaves an incomplete tail dense.
        removed = quartets.argsort(dim=-1, stable=True)[..., :2]
        block_keep = torch.ones_like(quartets, dtype=torch.bool)
        block_keep.scatter_(-1, removed, False)
        keep[:, :complete_end] = block_keep.reshape(rows, complete_end)
    return keep
