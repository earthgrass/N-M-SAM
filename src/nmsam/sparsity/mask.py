"""Legal masks and reproducible top-N magnitude; no training or scoring proxy."""
from itertools import combinations
import torch
from .layout import MaskSpec, MaskLayout


def enumerate_masks(spec: MaskSpec = MaskSpec(), *, device=None) -> torch.Tensor:
    """[C(M,N), M] bool masks, lexicographic kept-index order."""
    supports = list(combinations(range(spec.m), spec.n))
    masks = torch.zeros(len(supports), spec.m, dtype=torch.bool, device=device)
    for index, kept in enumerate(supports):
        masks[index, list(kept)] = True
    return masks


def validate_mask(mask: torch.Tensor, layout: MaskLayout) -> None:
    """Reject invalid shape, nonbinary values and groups not containing exactly N."""
    groups = layout.groups(mask)
    if not ((groups == 0) | (groups == 1)).all():
        raise ValueError("mask must be binary")
    if not (groups.to(torch.int64).sum(-1) == layout.spec.n).all():
        raise ValueError("each complete group must contain exactly N ones")


def build_magnitude_mask(weights: torch.Tensor, spec: MaskSpec = MaskSpec()) -> torch.Tensor:
    """Select abs(weight) descending; exact ties favor lower within-group index.

    Caller explicitly supplies one target weight; other layers are not touched.
    No silent padding/exclusion. Return bool tensor of identical shape/device.
    """
    if not weights.is_floating_point() or not torch.isfinite(weights).all():
        raise ValueError("weights must be finite real floating-point tensors")
    layout = MaskLayout(tuple(weights.shape), spec)
    scores = layout.groups(weights.detach()).abs()
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, order[..., :spec.n], True)
    result = layout.restore(mask)
    validate_mask(result, layout)
    return result


def mask_delta(weights: torch.Tensor, old: torch.Tensor, new: torch.Tensor,
               spec: MaskSpec = MaskSpec()) -> torch.Tensor:
    """Exact q=(P_new-P_old)W for one target layer, not SAM epsilon."""
    layout = MaskLayout(tuple(weights.shape), spec)
    validate_mask(old, layout)
    validate_mask(new, layout)
    if old.device != weights.device or new.device != weights.device:
        raise ValueError("weights and masks must share device")
    return (new.to(weights.dtype)-old.to(weights.dtype))*weights.detach()

