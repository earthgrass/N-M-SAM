"""HQ-APBR V1 CPU mathematical reference; no formal training entry point."""

from .math_core import (
    GeometrySpec,
    final_obs_keep_mask,
    hq_global_l2_perturbation,
    historical_geometry_update,
    inverse_diagonal_cholesky,
    obs_singleton_cost,
    target_from_inverse_diagonals,
)

__all__ = [
    "GeometrySpec", "final_obs_keep_mask", "hq_global_l2_perturbation",
    "historical_geometry_update", "inverse_diagonal_cholesky",
    "obs_singleton_cost", "target_from_inverse_diagonals",
]
