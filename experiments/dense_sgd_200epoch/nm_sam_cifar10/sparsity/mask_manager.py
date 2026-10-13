"""Sparse-layer selection, mask updates, and auditable N:M reporting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import math
import torch.nn as nn

from .mask import SparsityConfig
from .parametrization import attach_nm_sparsity


SPARSE_SCOPES = (
    "phase1",
    "all_supported",
    "exclude_first",
    "exclude_first_last",
    "conv_only",
    "conv_linear",
)


@dataclass(frozen=True)
class LayerDecision:
    name: str
    layer_type: str
    weight_shape: tuple[int, ...]
    sparse: bool
    reason: str
    blocks: int
    tail_weights: int


def _supported_layers(model: nn.Module):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def plan_sparse_layers(
    model: nn.Module,
    scope: str,
    n: int = 2,
    m: int = 4,
) -> List[LayerDecision]:
    if scope not in SPARSE_SCOPES:
        raise ValueError(f"Unknown sparse scope '{scope}'")
    supported = _supported_layers(model)
    first_conv = next((name for name, layer in supported if isinstance(layer, nn.Conv2d)), None)
    last_linear = next(
        (name for name, layer in reversed(supported) if isinstance(layer, nn.Linear)), None
    )
    decisions: List[LayerDecision] = []
    for name, layer in supported:
        weight = layer.weight
        is_conv = isinstance(layer, nn.Conv2d)
        inner = weight.shape[1] * weight.shape[2] * weight.shape[3] if is_conv else weight.shape[1]
        blocks = int(weight.shape[0] * (inner // m))
        tail_weights = int(weight.shape[0] * (inner % m))
        sparse = True
        reason = f"selected by scope={scope}"
        if inner < m:
            sparse, reason = False, f"inner dimension {inner} < M={m}"
        elif scope == "phase1":
            if name == first_conv:
                sparse, reason = False, "Phase-1 default excludes first convolution"
            elif is_conv and layer.kernel_size == (1, 1):
                sparse, reason = False, "Phase-1 default excludes 1x1/downsample convolution"
        elif scope == "exclude_first" and name == first_conv:
            sparse, reason = False, "scope excludes first convolution"
        elif scope == "exclude_first_last" and name in {first_conv, last_linear}:
            sparse, reason = False, "scope excludes first convolution and final linear"
        elif scope == "conv_only" and not is_conv:
            sparse, reason = False, "scope includes Conv2d only"
        decisions.append(
            LayerDecision(
                name=name,
                layer_type=type(layer).__name__,
                weight_shape=tuple(weight.shape),
                sparse=sparse,
                reason=reason,
                blocks=blocks if sparse else 0,
                tail_weights=tail_weights if sparse else 0,
            )
        )
    return decisions


class MaskManager:
    def __init__(self, model: nn.Module, cfg: SparsityConfig, scope: str = "phase1") -> None:
        self.model = model
        self.cfg = cfg
        self.scope = scope
        self.decisions = plan_sparse_layers(model, scope, cfg.n, cfg.m)
        selected: Set[str] = {decision.name for decision in self.decisions if decision.sparse}
        _, self.wrappers = attach_nm_sparsity(
            model, cfg, skip_first_conv=False, layer_names=selected
        )
        self.last_layer_change: Dict[str, float] = {}

    def recompute(self, score_type: str) -> float:
        changed = 0.0
        elements = 0
        layer_change: Dict[str, float] = {}
        for wrapper in self.wrappers:
            grad = wrapper.last_grad if score_type in ("taylor", "sam_taylor") else None
            change = wrapper.recompute_mask(score=score_type, grad=grad)
            layer_change[wrapper.name] = change
            changed += change * wrapper.mask.numel()
            elements += wrapper.mask.numel()
        self.last_layer_change = layer_change
        return float(changed / max(elements, 1))

    def layer_change_ratios(self) -> Dict[str, float]:
        return dict(self.last_layer_change)

    def selected_names(self) -> Set[str]:
        return {decision.name for decision in self.decisions if decision.sparse}

    def write_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        total_blocks = sum(decision.blocks for decision in self.decisions)
        sparse_weights = sum(wrapper.mask.numel() for wrapper in self.wrappers)
        sparse_zeros = sum(int((wrapper.mask == 0).sum().item()) for wrapper in self.wrappers)
        all_supported_weights = sum(math.prod(decision.weight_shape) for decision in self.decisions)
        overall = sparse_zeros / max(all_supported_weights, 1)
        lines = [
            "N:M SPARSE LAYER REPORT",
            f"scope: {self.scope}",
            f"configured N:M: {self.cfg.n}:{self.cfg.m}",
            f"theoretical full-block sparsity: {1.0 - self.cfg.n / self.cfg.m:.6f}",
            f"selected layers: {len(self.wrappers)}",
            f"total N:M blocks: {total_blocks}",
            f"selected weight elements: {sparse_weights}",
            f"actual overall Conv2d/Linear weight sparsity: {overall:.6f}",
            "",
            "layer\tstatus\ttype\tshape\tblocks\ttail_weights\treason",
        ]
        for decision in self.decisions:
            status = "sparse" if decision.sparse else "dense"
            lines.append(
                f"{decision.name}\t{status}\t{decision.layer_type}\t"
                f"{decision.weight_shape}\t{decision.blocks}\t{decision.tail_weights}\t{decision.reason}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["SPARSE_SCOPES", "LayerDecision", "MaskManager", "plan_sparse_layers"]
