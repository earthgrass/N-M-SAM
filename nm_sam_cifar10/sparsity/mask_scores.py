"""Mask-score names and their mathematical definitions."""
from __future__ import annotations

from typing import Optional

import torch


SCORE_TYPES = ("magnitude", "taylor_g1", "taylor_g2")


def compute_mask_score(
    weight: torch.Tensor,
    score_type: str,
    grad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if score_type == "magnitude":
        return weight.abs()
    if score_type in ("taylor_g1", "taylor_g2", "taylor", "sam_taylor"):
        if grad is None:
            raise ValueError(f"grad is required for score_type='{score_type}'")
        return (weight * grad).abs()
    raise ValueError(f"Unknown score_type '{score_type}'")


__all__ = ["SCORE_TYPES", "compute_mask_score"]
