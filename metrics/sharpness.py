"""SAM sharpness probe: first-order approximation of worst-case loss.

We follow the *same* mathematical definition as the training-time SAM
perturbation so that the reported ``S_rho(w)`` is comparable with the
quantity the optimizer is actually targeting.

Mathematical definition
-----------------------

```
g        = ∇_w L_b(w)
g_allowed = M ⊙ g                       # M is the N:M mask for sparse layers
norm     = sqrt(Σ_p ||g_allowed_p||²)   # GLOBAL L2 norm over allowed dirs
ε        = rho * g_allowed / norm
L_adv    = L_b(w + ε)
S_rho    = L_adv - L(w)
```

For dense layers ``M = 1`` so this collapses to the standard SAM probe.
For sparse layers the perturbation lives in the allowed sub-space,
matching the training-time SAM step exactly.

The probe restores:
* all parameter tensors to ``w`` after the perturbation,
* the model's training/eval mode,
* its BN running buffers (which are not touched because we run in
  ``model.eval()``).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from nm_sam_cifar10.sparsity.perturbation import (
    build_constrained_directions,
    global_direction_norm,
)


@torch.enable_grad()
def compute_sharpness(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    rho: float,
    device: torch.device,
    mask_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None,
    adaptive: bool = False,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Return ``(L_center, L_adv, sharpness, info_dict)``.

    ``info_dict`` contains ``g_norm``, ``eps_norm``, ``eps_norm_over_rho``
    and is intended for logging.
    """
    was_training = model.training
    model.eval()
    model.zero_grad()

    inputs = inputs.to(device)
    targets = targets.to(device)

    # L_center
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    directions = build_constrained_directions(model.parameters(), mask_fn, adaptive=adaptive)
    norm = global_direction_norm(directions)

    info: Dict[str, float] = {
        "g_norm": float(norm.item()),
        "rho": float(rho),
        "skipped": False,
    }
    if norm.item() < 1e-12:
        info["eps_norm"] = 0.0
        info["eps_norm_over_rho"] = 0.0
        model.zero_grad()
        if was_training:
            model.train()
        return float(loss.item()), float(loss.item()), 0.0, info

    scale = rho / norm
    # Save parameters, perturb, compute L_adv, restore.
    saved = {p: p.detach().clone() for p in model.parameters()}
    with torch.no_grad():
        for p, d in directions.items():
            p.add_(d * scale)

    with torch.no_grad():
        logits_adv = model(inputs)
        adv_loss = criterion(logits_adv, targets).item()

    # Compute effective epsilon norm for logging.
    eps_sq = torch.zeros(1)
    for d in directions.values():
        eps_sq = eps_sq + (d * scale).pow(2).sum()
    eps_norm = float(eps_sq.sqrt().item())
    info["eps_norm"] = eps_norm
    info["eps_norm_over_rho"] = eps_norm / float(rho)
    info["max_abs_logit"] = float(logits_adv.detach().abs().max().item())
    info["max_abs_logit_center"] = float(logits.detach().abs().max().item())

    # Restore parameters exactly.
    with torch.no_grad():
        for p, s in saved.items():
            p.copy_(s)

    model.zero_grad()
    if was_training:
        model.train()

    return float(loss.item()), float(adv_loss), float(adv_loss - loss.item()), info
