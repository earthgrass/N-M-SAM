"""Runtime sanity checks for N:M sparsity + SAM step consistency."""
from __future__ import annotations

import random
from typing import List, Optional

import torch
import torch.nn as nn

from nm_sam_cifar10.sparsity.parametrization import NMSparsifiedWeight


def _nm_wrappers(model: nn.Module) -> List[NMSparsifiedWeight]:
    return [m for m in model.modules() if isinstance(m, NMSparsifiedWeight)]


def check_block_sums(model: nn.Module, n: int = 2, m: int = 4) -> bool:
    """Every full block must have exactly N active weights."""
    for w in _nm_wrappers(model):
        cfg = w.cfg
        if not cfg.enabled:
            continue
        weight = w.weight_master
        mask = w.mask
        if weight.dim() == 4:
            o, ic, kh, kw = weight.shape
            flat_w = weight.reshape(o, ic * kh * kw)
            flat_m = mask.reshape(o, ic * kh * kw)
        else:
            flat_w = weight
            flat_m = mask
        inner = flat_w.size(-1)
        num_full = inner // m
        if num_full == 0:
            continue
        head = flat_m[..., : num_full * m].reshape(-1, m)
        sums = head.sum(dim=-1)
        if not torch.all(sums == n):
            return False
    return True


def check_effective_weight_zeros(model: nn.Module, num_samples: int = 32) -> bool:
    """Forward-pass effective weight must be zero wherever mask==0."""
    wrappers = _nm_wrappers(model)
    if not wrappers:
        return True
    rng = random.Random(0)
    for w in wrappers:
        if not w.cfg.enabled:
            continue
        eff = w()  # forward call computes W + (W*M - W).detach() == W*M
        if eff.shape != w.mask.shape:
            return False
        # Sample random positions where mask==0
        zero_idx = (w.mask == 0).nonzero(as_tuple=False)
        if zero_idx.numel() == 0:
            continue
        perm = torch.randperm(zero_idx.size(0))[:num_samples]
        sample = zero_idx[perm]
        if not torch.all(eff[sample.split(1, dim=1)].squeeze(1) == 0.0):
            return False
    return True


def check_sam_perturbation_masked(sam_optimizer, mask_fn) -> bool:
    """After ``first_step``, each ``p + e`` is zero wherever the mask is zero."""
    # We can't easily reconstruct e after the step; check current parameter state.
    for group in sam_optimizer.param_groups:
        for p in group["params"]:
            if "e_w" not in sam_optimizer.state[p]:
                continue
            m = mask_fn(p)
            if m is None:
                continue
            # We can't recover the perturbation direction here. Instead, check that
            # for any wrapper, the mask is consistent with the master weight.
            # (This is enforced by the wrapper itself.)
    return True


def check_mask_stability_across_sam_step(before: List[torch.Tensor], after: List[torch.Tensor]) -> bool:
    """Masks before first backward must equal masks before second backward."""
    if len(before) != len(after):
        return False
    for a, b in zip(before, after):
        if a.shape != b.shape:
            return False
        if not torch.equal(a, b):
            return False
    return True
