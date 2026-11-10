"""Differentiable pieces of the joint objective, isolated from APBR core."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def differentiable_apbr_perturbations(
    gradients: Sequence[Tensor],
    log_scales: Sequence[Tensor],
    *,
    rho: float,
) -> list[Tensor]:
    """Reproduce APBR epsilon while stopping gradients through first gradients.

    The returned perturbations retain a differentiable dependency on ``s``.
    This is the original global formula; there is no layer/block normalization.
    """
    if len(gradients) != len(log_scales) or not gradients or rho <= 0:
        raise ValueError("Invalid APBR perturbation inputs")
    h, b, signs = [], [], []
    for gradient, scale in zip(gradients, log_scales):
        if gradient.shape != scale.shape:
            raise ValueError("Gradient/scale coordinate mismatch")
        g = gradient.detach().double()
        if not bool(torch.isfinite(g).all() and torch.isfinite(scale).all()):
            raise FloatingPointError("Non-finite APBR perturbation input")
        logg = g.abs().log()
        h.append(scale.double() + logg)
        b.append(2 * scale.double() + logg)
        signs.append(g.sign())
    maximum = torch.stack([item.max() for item in h]).max()
    if bool(torch.isneginf(maximum)):
        return [scale.sum() * 0 for scale in log_scales]
    norm_log = maximum + 0.5 * torch.log(sum(torch.exp(2 * (item - maximum)).sum() for item in h))
    epsilon = [sign * torch.exp(math.log(rho) + exponent - norm_log)
               for exponent, sign in zip(b, signs)]
    if not all(bool(torch.isfinite(item).all()) for item in epsilon):
        raise FloatingPointError("Non-finite differentiable APBR perturbation")
    return epsilon


def joint_training_loss(adversarial_ce: Tensor, pruning_regularizer: Tensor, lambda_p: float) -> Tensor:
    if lambda_p < 0 or not bool(torch.isfinite(adversarial_ce) and torch.isfinite(pruning_regularizer)):
        raise ValueError("Invalid joint loss inputs")
    result = adversarial_ce + float(lambda_p) * pruning_regularizer
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("Non-finite joint training loss")
    return result
