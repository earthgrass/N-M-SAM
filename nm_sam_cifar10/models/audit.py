"""Generate a deterministic layer-by-layer ResNet18-CIFAR audit."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch.nn as nn

from ..sparsity.mask_manager import plan_sparse_layers


def _batchnorm_partner(name: str, modules: Dict[str, nn.Module]) -> str:
    if name == "conv1" and "bn1" in modules:
        return "bn1"
    if name.endswith(".conv1"):
        candidate = name[:-5] + "bn1"
        return candidate if candidate in modules else "-"
    if name.endswith(".conv2"):
        candidate = name[:-5] + "bn2"
        return candidate if candidate in modules else "-"
    if name.endswith(".shortcut.0"):
        candidate = name[:-1] + "1"
        return candidate if candidate in modules else "-"
    return "-"


def architecture_rows(model: nn.Module, sparse_scope: str = "phase1", n: int = 2, m: int = 4):
    modules = dict(model.named_modules())
    decisions = {d.name: d for d in plan_sparse_layers(model, sparse_scope, n, m)}
    rows = []
    for name, layer in modules.items():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        decision = decisions[name]
        if isinstance(layer, nn.Conv2d):
            layer_type = "Conv2d"
            channels = f"{layer.in_channels} -> {layer.out_channels}"
            kernel = str(tuple(layer.kernel_size))
            stride = str(tuple(layer.stride))
        else:
            layer_type = "Linear"
            channels = f"{layer.in_features} -> {layer.out_features}"
            kernel = "-"
            stride = "-"
        rows.append({
            "name": name,
            "type": layer_type,
            "channels": channels,
            "kernel": kernel,
            "stride": stride,
            "shape": str(tuple(layer.weight.shape)),
            "bias": "yes" if layer.bias is not None else "no",
            "batchnorm": _batchnorm_partner(name, modules),
            "sparse": "yes" if decision.sparse else "no",
            "blocks": decision.blocks,
            "tail": decision.tail_weights,
            "reason": decision.reason,
        })
    return rows


def write_architecture_audit(
    model: nn.Module,
    path: str | Path,
    sparse_scope: str = "phase1",
    n: int = 2,
    m: int = 4,
) -> None:
    rows = architecture_rows(model, sparse_scope, n, m)
    lines = [
        "# CNN Architecture Audit",
        "",
        "Model: ResNet18-CIFAR10. Input is `3 x 32 x 32`; the stem is a 3x3, stride-1, padding-1 convolution; there is no leading max-pool; the classifier has 10 outputs.",
        "",
        f"N:M layout: Conv2d `[out,in,kh,kw] -> [out,in*kh*kw]`, Linear `[out,in]`; full blocks use {n}:{m}, tails remain dense. Sparse scope shown: `{sparse_scope}`.",
        "",
        "| layer_name | type | input/output | kernel | stride | weight shape | bias | BatchNorm | N:M | blocks | tail weights | decision |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {type} | {channels} | {kernel} | {stride} | {shape} | "
            "{bias} | {batchnorm} | {sparse} | {blocks} | {tail} | {reason} |".format(**row)
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["architecture_rows", "write_architecture_audit"]
