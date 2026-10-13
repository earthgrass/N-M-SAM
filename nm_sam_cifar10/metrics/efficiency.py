"""CPU/software execution metrics (not hardware speedup claims)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EfficiencyMetrics:
    time_per_epoch: float
    total_time: float
    iterations_per_second: float
    images_per_second: float


def compute_efficiency(epoch_time: float, total_time: float, iterations: int, images: int) -> EfficiencyMetrics:
    return EfficiencyMetrics(
        time_per_epoch=float(epoch_time),
        total_time=float(total_time),
        iterations_per_second=float(iterations / max(epoch_time, 1e-12)),
        images_per_second=float(images / max(epoch_time, 1e-12)),
    )


__all__ = ["EfficiencyMetrics", "compute_efficiency"]
