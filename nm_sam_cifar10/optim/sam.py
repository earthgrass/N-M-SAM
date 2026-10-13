"""Sharpness-Aware Minimization (SAM) optimizer.

Mathematical contract (uniform across dense and N:M sparse):

    g1   = ∇_w L_b(w)
    g_allowed = M ⊙ g1                 # per-parameter allowed direction
                                       # M = mask for N:M weights,
                                       # M = 1 for dense / BN / bias.
    norm = sqrt( Σ_p ||g_allowed_p||² )  # GLOBAL L2 norm over allowed dirs
    ε_p  = rho * g_allowed_p / norm     # applied to every parameter
    w'   = w + ε
    g2   = ∇_w L_b(w')
    w_new = w - lr * SGD_step(g2)

With this construction:

    ||ε||_2 = sqrt( Σ_p ||ε_p||² ) = rho * ||g_allowed|| / ||g_allowed|| = rho

for **any** combination of sparse and dense parameters, **as long as**
at least one element of g_allowed is non-zero.

Implementation notes
--------------------
* The base optimizer's ``.step()`` is invoked **exactly once per SAM
  iteration**, **after** restoring ε from the parameter tensors.
* For BatchNorm-equipped models we want the *first* forward pass to
  update running statistics normally and the *second* (adversarial)
  forward pass to use batch statistics WITHOUT updating the running
  buffers. We expose ``freeze_bn_running_stats`` for that purpose
  (sets ``track_running_stats=False`` for the block).
* Sparse master weights receive a mask-aware perturbation: the mask is
  applied to the direction ``g_allowed`` *before* the global norm is
  computed. This guarantees ``||ε|| ≈ rho`` even when many entries of
  ``g`` are masked to zero.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional

import torch
from torch.optim.optimizer import Optimizer

from ..sparsity.perturbation import (
    build_constrained_directions,
    global_direction_norm,
)


_BN_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


# --------------------------------------------------------------------------- #
# BatchNorm helpers
# --------------------------------------------------------------------------- #


@contextmanager
def disable_running_stats(model: torch.nn.Module) -> Iterator[None]:
    """Legacy helper kept for backwards compatibility.

    Sets BN ``momentum = 0`` so the running_mean / running_var are not
    updated, but ``num_batches_tracked`` still increments and the BN
    forward still uses batch statistics. Prefer
    :func:`freeze_bn_running_stats` for SAM second-forward passes.
    """
    backup_momentum: List[float] = []
    for module in model.modules():
        if isinstance(module, _BN_TYPES):
            backup_momentum.append(module.momentum)
            module.momentum = 0.0
    try:
        yield
    finally:
        i = 0
        for module in model.modules():
            if isinstance(module, _BN_TYPES):
                module.momentum = backup_momentum[i]
                i += 1


@contextmanager
def freeze_bn_running_stats(model: torch.nn.Module) -> Iterator[None]:
    """Freeze BN running buffers AND batch-counter for the block.

    During the SAM second forward we want the BN layers to (a) still use
    batch statistics (so the gradient is well-defined) but (b) NOT
    update ``running_mean``, ``running_var``, or ``num_batches_tracked``.

    PyTorch exposes this through ``track_running_stats``: when set to
    ``False``, BN uses batch statistics only and does not touch the
    running buffers.
    """
    backup: List[bool] = []
    for module in model.modules():
        if isinstance(module, _BN_TYPES):
            backup.append(module.track_running_stats)
            module.track_running_stats = False
    try:
        yield
    finally:
        i = 0
        for module in model.modules():
            if isinstance(module, _BN_TYPES):
                module.track_running_stats = backup[i]
                i += 1


@contextmanager
def enable_running_stats(model: torch.nn.Module) -> Iterator[None]:
    """No-op placeholder kept for API symmetry."""
    yield


# --------------------------------------------------------------------------- #
# Perturbation helpers (shared by training SAM and sharpness probe)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# SAM optimizer
# --------------------------------------------------------------------------- #


class SAM(Optimizer):
    """SAM wrapper around a base optimizer (SGD by default).

    Use :meth:`first_step` and :meth:`second_step` rather than
    ``.step()``; the wrapper does NOT support the latter.

    The base optimizer (e.g. ``torch.optim.SGD``) is constructed once
    and stored; ``SAM.param_groups`` aliases its ``param_groups`` so
    state-dict / zero_grad flow through the base optimizer.
    """

    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults.update(base_optimizer.defaults)
        for group in self.param_groups:
            group.setdefault("rho", rho)
            group.setdefault("adaptive", adaptive)

    @torch.no_grad()
    def first_step(
        self,
        zero_grad: bool = False,
        mask_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Apply the constrained SAM perturbation.

        Returns ``{"g_norm": float, "eps_norm": float,
        "eps_norm_over_rho": float, "skipped": bool}``.
        """
        rho = self.param_groups[0].get("rho", 0.05)
        adaptive = self.param_groups[0].get("adaptive", False)

        directions = build_constrained_directions(
            self.__get_model_module().parameters(), mask_fn, adaptive
        )
        norm = global_direction_norm(directions)

        info = {
            "g_norm": float(norm.item()),
            "rho": float(rho),
            "skipped": False,
        }
        if norm.item() < 1e-12:
            # No allowed gradient -> skip perturbation.
            info["eps_norm"] = 0.0
            info["eps_norm_over_rho"] = 0.0
            self.state["__sam_step_info__"] = info
            if zero_grad:
                self.zero_grad()
            return info

        scale = rho / norm
        masked_epsilon_max_abs = 0.0
        for p, d in directions.items():
            e = d * scale
            mask = mask_fn(p) if mask_fn is not None else None
            if mask is not None and bool((mask == 0).any().item()):
                masked_epsilon_max_abs = max(
                    masked_epsilon_max_abs,
                    float(e[mask == 0].abs().max().item()),
                )
            # Copying the unperturbed parameter permits bit-exact restoration.
            # A subtraction-based restore can incur a float32 round-off residue.
            self.state[p]["old_p"] = p.detach().clone()
            self.state[p]["e"] = e.detach().clone()
            p.add_(e)

        eps_sq = torch.zeros(1)
        device = None
        for d in directions.values():
            s = (d * scale).pow(2).sum()
            if device is None:
                device = s.device
                eps_sq = eps_sq.to(device)
            eps_sq = eps_sq + s
        eps_norm = float(eps_sq.sqrt().item())
        info["eps_norm"] = eps_norm
        info["eps_norm_over_rho"] = eps_norm / float(rho)
        info["masked_epsilon_max_abs"] = masked_epsilon_max_abs
        self.state["__sam_step_info__"] = info

        if zero_grad:
            self.zero_grad()
        return info

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> Dict[str, float]:
        """Restore parameters bit-exactly, then run base SGD once at ``w``."""
        restored = 0
        restore_max_abs_error = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if "e" not in self.state[p]:
                    continue
                old_p = self.state[p].pop("old_p")
                p.copy_(old_p)
                restore_max_abs_error = max(
                    restore_max_abs_error, float((p - old_p).abs().max().item())
                )
                del self.state[p]["e"]
                restored += 1
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
        info = {
            "restored_parameter_tensors": float(restored),
            "restore_max_abs_error": restore_max_abs_error,
            "leftover_epsilon_buffers": float(sum(
                1 for group in self.param_groups for p in group["params"]
                if "e" in self.state[p] or "old_p" in self.state[p]
            )),
        }
        self.state["__sam_restore_info__"] = info
        return info

    def last_step_info(self) -> Dict[str, float]:
        return dict(self.state.get("__sam_step_info__", {}))

    def last_restore_info(self) -> Dict[str, float]:
        return dict(self.state.get("__sam_restore_info__", {}))

    def __get_model_module(self) -> torch.nn.Module:
        """Find the nn.Module owning these parameters. Used only to
        iterate ``model.parameters()`` for building directions; SAM itself
        works on the union of all params."""
        # We rely on model.parameters() having been passed; reconstruct
        # via param_groups is enough because directions are computed
        # over those Parameters' .grad attributes.
        # This is a placeholder; in practice the caller passes the
        # model object via ``mask_fn`` or we accept that we just need
        # the params list.
        class _ParamOwner(torch.nn.Module):
            def __init__(self, params):
                super().__init__()
                self._seen = set()
                seen_list = []
                for i, p in enumerate(params):
                    if id(p) in self._seen:
                        continue
                    self._seen.add(id(p))
                    seen_list.append(p)
                self._params = seen_list

            def parameters(self):
                return self._params

        # Flatten param_groups into one unique list.
        all_params = []
        seen = set()
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in seen:
                    continue
                seen.add(id(p))
                all_params.append(p)
        return _ParamOwner(all_params)

    def step(self, closure=None):  # pragma: no cover
        raise NotImplementedError("Use first_step + second_step")

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


def make_sam_optimizer(
    model: torch.nn.Module,
    lr: float,
    momentum: float,
    weight_decay: float,
    rho: float,
    nesterov: bool = False,
    adaptive: bool = False,
) -> SAM:
    base = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    return SAM(model.parameters(), base_optimizer=base, rho=rho, adaptive=adaptive)
