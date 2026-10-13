"""One shared dense or constrained-sparse SAM training step."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..optim.sam import SAM, freeze_bn_running_stats


@dataclass
class SAMStepResult:
    loss: float
    center_loss: float
    adversarial_loss: float
    accuracy: float
    batch_size: int
    pre_masks: List[torch.Tensor]
    post_masks: List[torch.Tensor]
    info: Dict[str, float]
    first_input_data_ptr: int
    second_input_data_ptr: int
    g1_finite: bool
    g2_finite: bool
    restore_info: Dict[str, float]
    mask_scores: List[torch.Tensor]
    center_weight_snapshots: Optional[List[torch.Tensor]] = None
    g1_snapshots: Optional[List[torch.Tensor]] = None
    g2_snapshots: Optional[List[torch.Tensor]] = None


def sam_step(
    model: nn.Module,
    optimizer: SAM,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    param_mask_fn=None,
    mask_score: Optional[str] = None,
    input_observer: Optional[Callable[[str, torch.Tensor], None]] = None,
    capture_gradients: bool = False,
    capture_masks: bool = True,
) -> SAMStepResult:
    """Run both SAM passes on the exact same already-augmented tensor.

    The mask is captured before the first pass and after the complete SAM
    step. Mask recomputation is intentionally not performed here; callers may
    update it only after this function returns, so both passes use M_t.
    """
    model.train()
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    wrappers = list(getattr(model, "_nm_wrappers", []))
    pre_masks = (
        [wrapper.mask.detach().clone() for wrapper in wrappers]
        if capture_masks else []
    )
    need_scores = mask_score is not None
    center_weights = (
        [wrapper.weight_master.detach().clone() for wrapper in wrappers]
        if need_scores or capture_gradients else []
    )
    mask_scores: List[torch.Tensor] = []
    if mask_score == "magnitude":
        mask_scores = [weight.abs() for weight in center_weights]
    elif mask_score not in (None, "taylor_g1", "taylor_g2"):
        raise ValueError(f"Unsupported mask_score '{mask_score}'")

    optimizer.zero_grad(set_to_none=True)
    if input_observer is not None:
        input_observer("first", inputs)
    first_ptr = inputs.data_ptr()
    logits = model(inputs)
    center_loss = criterion(logits, targets)
    center_loss.backward()
    g1 = []
    if mask_score == "taylor_g1" or capture_gradients:
        g1 = [
            wrapper.weight_master.grad.detach().clone()
            if wrapper.weight_master.grad is not None else torch.zeros_like(wrapper.weight_master)
            for wrapper in wrappers
        ]
    if mask_score == "taylor_g1":
        mask_scores = [(weight * grad).abs() for weight, grad in zip(center_weights, g1)]
    g1_finite = all(
        bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters() if parameter.grad is not None
    )
    info = optimizer.first_step(zero_grad=True, mask_fn=param_mask_fn)

    with freeze_bn_running_stats(model):
        if input_observer is not None:
            input_observer("second", inputs)
        second_ptr = inputs.data_ptr()
        adversarial_logits = model(inputs)
        adversarial_loss = criterion(adversarial_logits, targets)
        adversarial_loss.backward()
    g2_finite = all(
        bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters() if parameter.grad is not None
    )
    g2 = []
    if mask_score == "taylor_g2" or capture_gradients:
        g2 = [
            wrapper.weight_master.grad.detach().clone()
            if wrapper.weight_master.grad is not None else torch.zeros_like(wrapper.weight_master)
            for wrapper in wrappers
        ]
    if mask_score == "taylor_g2":
        mask_scores = [(weight * grad).abs() for weight, grad in zip(center_weights, g2)]

    restore_info = optimizer.second_step(zero_grad=True)
    post_masks = (
        [wrapper.mask.detach().clone() for wrapper in wrappers]
        if capture_masks else []
    )
    accuracy = (adversarial_logits.argmax(dim=1) == targets).float().mean().item()
    return SAMStepResult(
        loss=float(adversarial_loss.item()),
        center_loss=float(center_loss.item()),
        adversarial_loss=float(adversarial_loss.item()),
        accuracy=float(accuracy),
        batch_size=int(inputs.shape[0]),
        pre_masks=pre_masks,
        post_masks=post_masks,
        info=info,
        first_input_data_ptr=first_ptr,
        second_input_data_ptr=second_ptr,
        g1_finite=g1_finite,
        g2_finite=g2_finite,
        restore_info=restore_info,
        mask_scores=mask_scores,
        center_weight_snapshots=center_weights if capture_gradients else None,
        g1_snapshots=g1 if capture_gradients else None,
        g2_snapshots=g2 if capture_gradients else None,
    )


__all__ = ["SAMStepResult", "sam_step"]
