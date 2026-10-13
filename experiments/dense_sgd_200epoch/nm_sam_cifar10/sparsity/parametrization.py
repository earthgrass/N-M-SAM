"""STE mask parametrization: ``W_eff = W + (W * mask - W).detach()``.

This implements "training-time N:M sparsity" without ever shrinking the
weight tensor. Forward sees ``W * mask`` (so masked weights contribute
zero), backward uses straight-through identity so the dense master weight
still receives gradient for ALL positions -- allowing previously masked
weights to re-enter the mask later.

The class also tracks:
* current mask buffer (so we can switch between magnitude and SAM-aware)
* a generation counter for logging / hysteresis (not yet active)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .mask import SparsityConfig, mask_for_weight, sam_aware_mask


class NMSparsifiedWeight(nn.Module):
    """Wraps a 2D / 4D weight tensor with N:M sparsity via STE mask."""

    def __init__(
        self,
        weight: torch.Tensor,
        cfg: SparsityConfig,
        name: str = "weight",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        # The dense master weight is trainable.
        self.weight_master = nn.Parameter(weight.detach().clone())
        # Initial mask ALWAYS uses magnitude — even for SAM-aware configs.
        # The configured score is consumed only when ``recompute_mask`` is
        # called with a gradient; at construction time no gradient exists.
        init_cfg = SparsityConfig(n=cfg.n, m=cfg.m, enabled=cfg.enabled,
                                  score="magnitude", kind=cfg.kind)
        if cfg.enabled:
            with torch.no_grad():
                init_mask = mask_for_weight(self.weight_master.data, init_cfg)
        else:
            init_mask = torch.ones_like(self.weight_master.data)
        self.register_buffer("mask", init_mask.clone())
        self.generation = 0  # increments every recompute
        # Cached grad from SAM's second backward (set externally for SAM-aware).
        self.last_grad: Optional[torch.Tensor] = None

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_master

    def forward(self) -> torch.Tensor:
        if not self.cfg.enabled:
            return self.weight_master
        # STE: forward = W * mask, backward ~ identity on W.
        w = self.weight_master
        m = self.mask
        w_eff = w + (w * m - w).detach()
        return w_eff

    @torch.no_grad()
    def recompute_mask(
        self,
        score: str = "magnitude",
        grad: Optional[torch.Tensor] = None,
    ) -> float:
        """Recompute the mask. Returns the fraction of weights that changed."""
        if not self.cfg.enabled:
            return 0.0
        new_mask = _build_mask(self.weight_master.data, self.cfg, score=score, grad=grad)
        change = (new_mask != self.mask).float().mean().item()
        self.mask.copy_(new_mask)
        self.generation += 1
        return change

    def active_ratio(self) -> float:
        if not self.cfg.enabled:
            return 1.0
        return float(self.mask.float().mean().item())

    def extra_repr(self) -> str:
        return f"name={self.name}, n={self.cfg.n}, m={self.cfg.m}, enabled={self.cfg.enabled}"


def _build_mask(
    weight: torch.Tensor,
    cfg: SparsityConfig,
    score: str = "magnitude",
    grad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if score == "magnitude":
        return mask_for_weight(weight, cfg)
    if score in ("taylor", "sam_taylor"):
        if grad is None:
            raise ValueError("Gradient required for score='" + score + "'")
        return sam_aware_mask(weight, grad, n=cfg.n, m=cfg.m, score=score)
    raise ValueError(f"Unknown score '{score}'")


def _grad_for_wrapper(wrapper: "NMSparsifiedWeight", grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """If grad was captured for the master weight (4D for convs, 2D for linear),
    return it unchanged. This helper is intentionally trivial; it exists so the
    build_mask dispatch is the single source of truth about shape semantics."""
    return grad


class _WrappedLinear(nn.Module):
    def __init__(self, wrapper: NMSparsifiedWeight, bias: Optional[torch.Tensor]) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.register_parameter("bias", bias if bias is None else nn.Parameter(bias.detach()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.wrapper(), self.bias)


class _WrappedConv2d(nn.Module):
    def __init__(
        self,
        wrapper: NMSparsifiedWeight,
        bias: Optional[torch.Tensor],
        stride,
        padding,
        dilation,
        groups: int,
    ) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.register_parameter("bias", bias if bias is None else nn.Parameter(bias.detach()))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x,
            self.wrapper(),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def attach_nm_sparsity(
    model: nn.Module,
    cfg: SparsityConfig,
    layer_filter: Optional[Callable[[nn.Module], bool]] = None,
    skip_first_conv: bool = True,
    layer_names: Optional[Set[str]] = None,
) -> Tuple[int, List[NMSparsifiedWeight]]:
    """Replace selected weight tensors with :class:`NMSparsifiedWeight`.

    ``layer_filter`` is a callable ``module -> bool`` that decides which
    layers to sparsify. Default: all ``nn.Linear`` and non-1x1
    ``nn.Conv2d`` except the first conv. New code should pass an explicit
    ``layer_names`` set produced by :class:`MaskManager`.

    Returns ``(num_wrapped, list_of_wrappers)``.
    """
    wrapped = 0
    wrappers: List[NMSparsifiedWeight] = []
    first_conv_seen = False

    for name, module in list(model.named_modules()):
        if layer_names is not None and name not in layer_names:
            continue
        if layer_filter is not None and not layer_filter(module):
            continue
        is_linear = isinstance(module, nn.Linear)
        is_conv = isinstance(module, nn.Conv2d)
        if not (is_linear or is_conv):
            continue
        if layer_names is None and is_conv and module.kernel_size == (1, 1):
            continue
        weight = module.weight.data
        inner = (
            weight.size(-1)
            if is_linear
            else weight.size(1) * weight.shape[2] * weight.shape[3]
        )
        if inner < cfg.m:
            continue
        if layer_names is None and is_conv and skip_first_conv and not first_conv_seen:
            first_conv_seen = True
            continue

        wrapper = NMSparsifiedWeight(weight, cfg, name=name)
        module.weight = None  # type: ignore[assignment]
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if is_linear:
            replacement = _WrappedLinear(wrapper, module.bias)
        else:
            replacement = _WrappedConv2d(
                wrapper,
                module.bias,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )
        setattr(parent, child_name, replacement)
        wrappers.append(wrapper)
        wrapped += 1

    model._nm_wrappers = wrappers  # type: ignore[attr-defined]
    return wrapped, wrappers


def make_param_mask_fn(model: nn.Module) -> Callable[[torch.Tensor], Optional[torch.Tensor]]:
    """Build a closure ``(param) -> mask tensor or None`` for SAM perturbation.

    Returns ``None`` for parameters that are not sparse master weights, so
    SAM's perturbation is unrestricted on biases / norms / dense weights.
    """
    wrappers: List[NMSparsifiedWeight] = list(getattr(model, "_nm_wrappers", []))
    lookup: Dict[int, NMSparsifiedWeight] = {id(w.weight_master): w for w in wrappers}

    def mask_fn(p: torch.Tensor) -> Optional[torch.Tensor]:
        w = lookup.get(id(p))
        if w is None or not w.cfg.enabled:
            return None
        return w.mask

    return mask_fn
