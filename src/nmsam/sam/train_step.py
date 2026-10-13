"""Transactional single-device dense SGD/SAM with official update order."""
from collections.abc import Callable
from copy import deepcopy
import math
import torch
from torch import nn
from nmsam.core.batch import StepBatch
from nmsam.core.loss import LossReport
from nmsam.core.step_types import SAMStepReport, StepTrace, TensorTuple
from nmsam.utils.device import validate_dense_device
from nmsam.utils.seed import capture_rng, restore_rng
from .perturbation import DensePolicy, PerturbationPolicy, perturbation, global_norm, clip_by_global_norm

LossFunction = Callable[[nn.Module, StepBatch], LossReport]


def _clone(tensors) -> TensorTuple:
    return tuple(t.detach().clone() for t in tensors)


@torch.no_grad()
def _copy(destination, source) -> None:
    for dst, src in zip(destination, source):
        dst.copy_(src)


def _validate_optimizer(model: nn.Module, optimizer: torch.optim.SGD) -> None:
    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError("bootstrap requires torch.optim.SGD")
    params = [p for group in optimizer.param_groups for p in group["params"]]
    wanted = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len(wanted) or {id(p) for p in params} != {id(p) for p in wanted}:
        raise ValueError("optimizer must own each trainable model parameter exactly once")
    for group in optimizer.param_groups:
        if group["weight_decay"] != 0 or group["dampening"] != 0 or group.get("maximize", False):
            raise ValueError("explicit loss L2 requires weight_decay=0, dampening=0, maximize=False")
        if not math.isfinite(group["lr"]) or group["lr"] <= 0:
            raise ValueError("optimizer lr must be finite and positive")
        if not math.isfinite(group["momentum"]) or not 0 <= group["momentum"] < 1:
            raise ValueError("momentum must be finite in [0,1)")


def train_step(model: nn.Module, optimizer: torch.optim.SGD, batch: StepBatch,
               loss_fn: LossFunction, *, rho: float = 0.05,
               gradient_clipping: float = 5.0,
               policy: PerturbationPolicy | None = None,
               capture_trace: bool = False) -> SAMStepReport:
    """Commit one update, or roll back on any exception including optimizer failure.

    Success: original modes/tracking switches, first-pass buffers/RNG, final grads.
    Failure: pre-step params/buffers/grads/optimizer/RNG/modes/tracking switches.
    Forward must not mutate batch, module registration or external state.
    """
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    if not math.isfinite(gradient_clipping) or gradient_clipping < 0:
        raise ValueError("gradient_clipping must be finite and nonnegative")
    validate_dense_device(model, batch.inputs.device)
    _validate_optimizer(model, optimizer)
    if torch.is_autocast_enabled(batch.inputs.device.type):
        raise ValueError("AMP/autocast is outside bootstrap scope")
    policy = DensePolicy() if policy is None else policy
    all_params = tuple(model.parameters())
    params = tuple(p for p in all_params if p.requires_grad)
    all_base = _clone(all_params)
    base = _clone(params)
    buffers = tuple(model.buffers())
    original_buffers = _clone(buffers)
    original_grads = tuple(None if p.grad is None else p.grad.detach().clone() for p in all_params)
    original_optimizer = deepcopy(optimizer.state_dict())
    original_rng = capture_rng()
    modes = [(m, m.training) for m in model.modules()]
    bn_tracking = [(m, m.track_running_stats) for m in model.modules()
                   if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def backward() -> tuple[LossReport, TensorTuple]:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        if loss.optimizer_loss.ndim != 0 or not torch.isfinite(loss.optimizer_loss):
            raise FloatingPointError("optimizer loss must be finite scalar")
        loss.optimizer_loss.backward()
        grads = tuple(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                      for p in params)
        if not torch.isfinite(global_norm(grads)):
            raise FloatingPointError("nonfinite gradient")
        return loss, grads

    try:
        model.train()
        first, g1 = backward()
        first_buffers = _clone(buffers)
        first_rng = capture_rng()
        epsilon = tuple(torch.zeros_like(g) for g in g1)
        zero_direction = global_norm(g1).item() == 0
        probe = base
        final, g2 = first, g1
        if rho > 0:
            epsilon, zero_direction = perturbation(g1, rho, policy)
            with torch.no_grad():
                for p, w, e in zip(params, base, epsilon):
                    p.copy_(w + e)
            probe = _clone(params) if capture_trace else ()
            # Both Flax forwards close over the original state collection.
            _copy(buffers, original_buffers)
            for bn, _ in bn_tracking:
                bn.track_running_stats = False  # training batch stats, no buffer updates
            restore_rng(original_rng)
            final, g2 = backward()
            _copy(buffers, first_buffers)
            restore_rng(first_rng)
        _copy(params, base)  # exact restoration before momentum update
        restored = _clone(params) if capture_trace else ()
        restoration_error = max((p.detach()-w).abs().max().item() for p, w in zip(params, base))
        clipped = clip_by_global_norm(g2, gradient_clipping)
        for p, g in zip(params, clipped):
            p.grad = g.clone()
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in all_params):
            raise FloatingPointError("optimizer produced nonfinite parameters")
        accuracy = None if final.logits is None else float(
            (final.logits.detach().argmax(dim=-1) == batch.targets).float().mean())
        trace = StepTrace(base, g1, epsilon, probe, g2, restored, clipped) if capture_trace else None
        return SAMStepReport(float(final.task_loss.detach()), float(final.optimizer_loss.detach()),
                             float(final.l2_penalty.detach()), accuracy,
                             float(global_norm(g1)), float(global_norm(g2)),
                             float(global_norm(epsilon)), float(global_norm(clipped)),
                             restoration_error, 2 if rho > 0 else 1, zero_direction, trace)
    except BaseException:
        _copy(all_params, all_base)
        _copy(buffers, original_buffers)
        for p, grad in zip(all_params, original_grads):
            p.grad = grad
        optimizer.load_state_dict(original_optimizer)
        restore_rng(original_rng)
        raise
    finally:
        for module, training in modes:
            module.training = training
        for bn, tracking in bn_tracking:
            bn.track_running_stats = tracking

