"""Bridge functional gradients into the legacy APBR ``parameter.grad`` API."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn


class GradientHandoff:
    """Temporarily install detached first-pass gradients without accumulation.

    ``None`` is retained exactly for unused parameters.  The adapter requires a
    clean destination and always restores every destination to ``None``.  It
    does not know about APBR and therefore cannot alter perturbation or geometry
    semantics.
    """

    def __init__(self, parameters: Sequence[nn.Parameter]) -> None:
        self.parameters = tuple(parameters)
        if not self.parameters or len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError("Gradient handoff requires unique parameters")
        self.active = False

    def _validate(self, gradients: Sequence[Tensor | None]) -> None:
        if len(gradients) != len(self.parameters):
            raise ValueError("Gradient handoff length mismatch")
        for parameter, gradient in zip(self.parameters, gradients):
            if gradient is None:
                continue
            if gradient.shape != parameter.shape or gradient.device != parameter.device:
                raise ValueError("Gradient handoff coordinate mismatch")
            if gradient.dtype != parameter.dtype:
                raise ValueError("Gradient handoff dtype mismatch")
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Non-finite first-pass gradient")

    @contextmanager
    def install(self, gradients: Sequence[Tensor | None]) -> Iterator[None]:
        if self.active:
            raise RuntimeError("Nested gradient handoff is forbidden")
        if any(parameter.grad is not None for parameter in self.parameters):
            raise RuntimeError("Gradient handoff destination is not clean")
        self._validate(gradients)
        self.active = True
        try:
            for parameter, gradient in zip(self.parameters, gradients):
                parameter.grad = None if gradient is None else gradient.detach().clone()
            yield
        finally:
            for parameter in self.parameters:
                parameter.grad = None
            self.active = False


def materialize_none_as_zeros(
    parameters: Sequence[nn.Parameter], gradients: Sequence[Tensor | None],
) -> list[Tensor]:
    """Return a separate dense view for differentiable probe construction."""
    if len(parameters) != len(gradients):
        raise ValueError("Gradient materialization length mismatch")
    return [torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients)]
