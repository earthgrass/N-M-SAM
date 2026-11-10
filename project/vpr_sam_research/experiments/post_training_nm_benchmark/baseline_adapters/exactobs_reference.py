"""Bounded, single-row reference using the pinned official TrueOBS kernel.

This is a diagnostic for one real Conv2d output row, not a full-layer pruning
method. The official selection, inverse update, and compensation are unchanged.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import time

import torch
from torch import Tensor, nn

from benchmark.obc import prune_with_hessian


MAX_REFERENCE_COLUMNS = 576


@dataclass(frozen=True)
class ExactOBSRowReference:
    output_channel: int
    original_weight: Tensor
    compensated_weight: Tensor
    keep_mask: Tensor
    physical_quadratic_cost: float
    damped_quadratic_cost: float
    runtime_seconds: float
    relative_damping: float
    damping_value: float
    dead_columns: int
    cholesky_info: int
    fallback_used: bool
    official_log: str
    upstream_receipt: dict


@torch.no_grad()
def run_official_exactobs_row(
    conv: nn.Conv2d,
    hessian: Tensor,
    output_channel: int,
    *,
    relative_damping: float = 0.01,
) -> ExactOBSRowReference:
    """Prune one real Conv2d row with official OBC/TrueOBS 2:4.

    ``hessian`` must use the same flattened OIHW input-coordinate order as
    ``conv.weight.flatten(1)``. Passing a Linear proxy prevents the official
    Conv2d channel-last permutation from changing the project's 2:4 groups.
    The Conv2d module and input Hessian are never mutated. The 576-column cap
    prevents an accidental full-size CPU ExactOBS run.
    """
    if not isinstance(conv, nn.Conv2d) or conv.groups != 1:
        raise ValueError("Expected an ungrouped Conv2d layer")
    if not 0 <= output_channel < conv.out_channels:
        raise ValueError("output_channel is outside the Conv2d output range")
    columns = conv.weight[output_channel].numel()
    if columns > MAX_REFERENCE_COLUMNS:
        raise ValueError(f"Diagnostic reference limited to K <= {MAX_REFERENCE_COLUMNS}")
    if columns % 4:
        raise ValueError("Official TrueOBS requires complete 2:4 groups")
    if hessian.shape != (columns, columns) or hessian.device != conv.weight.device:
        raise ValueError("Hessian shape/device does not match the Conv2d row")
    if not torch.isfinite(hessian).all() or not torch.isfinite(conv.weight).all():
        raise ValueError("Reference input contains a non-finite value")
    if relative_damping < 0:
        raise ValueError("relative_damping must be non-negative")

    original = conv.weight.detach().flatten(1)[output_channel].clone()

    # Mirror official TrueOBS.prepare for a transparent decomposition receipt.
    prepared = hessian.detach().float().clone()
    damping_value = 0.0
    if relative_damping > 0:
        damping_value = float((relative_damping * prepared.diagonal().mean()).item())
        prepared.diagonal().add_(damping_value)
    dead = prepared.diagonal() == 0
    prepared.diagonal()[dead] = 1
    _, info = torch.linalg.cholesky_ex(prepared)
    cholesky_info = int(info.item())
    if cholesky_info:
        raise RuntimeError(
            f"Official preflight Cholesky failed at minor {cholesky_info}; "
            "the official identity-fallback has not been invoked"
        )

    proxy = nn.Linear(
        columns, 1, bias=False, device=original.device, dtype=original.dtype
    )
    proxy.weight.copy_(original.unsqueeze(0))
    log = StringIO()
    if original.is_cuda:
        torch.cuda.synchronize(original.device)
    started = time.perf_counter()
    with redirect_stdout(log):
        result, keep, receipt = prune_with_hessian(
            proxy,
            hessian,
            n=2,
            m=4,
            parallel=1,
            relative_damping=relative_damping,
        )
    if original.is_cuda:
        torch.cuda.synchronize(original.device)
    runtime_seconds = time.perf_counter() - started
    official_log = log.getvalue()
    fallback_used = "Hessian not full rank." in official_log
    if fallback_used:
        raise RuntimeError("Official TrueOBS used its unregistered identity fallback")

    compensated = result[0].clone()
    keep_mask = keep[0].clone()
    if not torch.equal(conv.weight.detach().flatten(1)[output_channel], original):
        raise RuntimeError("Official row reference modified the input Conv2d")
    delta = (compensated - original).double()
    physical_cost = 0.5 * (delta @ hessian.double() @ delta)
    damped_cost = 0.5 * (delta @ prepared.double() @ delta)
    return ExactOBSRowReference(
        output_channel=output_channel,
        original_weight=original,
        compensated_weight=compensated,
        keep_mask=keep_mask,
        physical_quadratic_cost=float(physical_cost.item()),
        damped_quadratic_cost=float(damped_cost.item()),
        runtime_seconds=runtime_seconds,
        relative_damping=relative_damping,
        damping_value=damping_value,
        dead_columns=int(dead.sum().item()),
        cholesky_info=cholesky_info,
        fallback_used=False,
        official_log=official_log,
        upstream_receipt=receipt,
    )
