"""Isolated exact v=w*exp(-s) score variant of sequential SparseGPT.

Only the within-quartet ranking is changed to C_GPT*exp(-2s). The existing
radius-guided adapter owns the identical damping, factorization, traversal,
and physical-w OBS update used by its historical exp(-s) variant.
"""
from __future__ import annotations

import torch
from torch import Tensor

from baseline_adapters.radius_guided_sparsegpt.adapter import (
    RadiusGuidedSparseGPTResult,
    radius_guided_sparsegpt_matrix,
)


def geo_rank_sparsegpt_matrix(
    weight: Tensor,
    hessian: Tensor,
    log_scale: Tensor,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float32,
) -> RadiusGuidedSparseGPTResult:
    """Select two deletions per quartet by (w*exp(-s))**2/U_jj**2.

    The compensation remains the original physical-weight columnwise OBS.
    There is no six-pair enumeration or groupwise compensation.
    """
    return radius_guided_sparsegpt_matrix(
        weight,
        hessian,
        log_scale,
        keep_n=2,
        m=4,
        blocksize=blocksize,
        percdamp=percdamp,
        compute_dtype=compute_dtype,
        radius_lambda=2.0,
    )
