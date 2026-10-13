"""Composable method specifications.

Each method module only declares which shared components to assemble.  No
method owns an optimizer implementation, mask implementation, or training
loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Optional


@dataclass(frozen=True)
class MethodSpec:
    name: str
    optimizer: str
    sparsity: str
    mask_score: Optional[str]
    training_step: str

    @property
    def is_sparse(self) -> bool:
        return self.sparsity == "nm"

    @property
    def is_sam(self) -> bool:
        return self.optimizer == "sam"


METHOD_NAMES = (
    "dense_sgd",
    "dense_sam",
    "nm_sgd",
    "nm_sam",
    "nm_sam_aware",
)


def get_method_spec(name: str) -> MethodSpec:
    if name not in METHOD_NAMES:
        raise ValueError(f"Unknown method '{name}'")
    module = import_module(f"{__name__}.{name}")
    spec = module.METHOD
    if not isinstance(spec, MethodSpec):
        raise TypeError(f"{name}.METHOD must be a MethodSpec")
    return spec


__all__ = ["MethodSpec", "METHOD_NAMES", "get_method_spec"]
