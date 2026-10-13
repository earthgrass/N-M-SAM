"""Tests for SAM BN handling (P1 fix).

Verifies the exact contract:

  * First forward of a SAM iteration: BN running stats updated.
  * Second (adversarial) forward: BN running stats FROZEN.
  * Per SAM iteration, ``num_batches_tracked`` increases by exactly 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nm_sam_cifar10.optim.sam import (
    freeze_bn_running_stats,
    make_sam_optimizer,
)


def _toy_bn_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.BatchNorm1d(8),
    )


def _snap_bn(model):
    return [
        (m.running_mean.detach().clone(), m.running_var.detach().clone(), int(m.num_batches_tracked.item()))
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]


def _sam_step_correct(model, opt, crit, x, y):
    """Reference SAM step using the fixed BN pattern."""
    model.train()
    # First forward: normal BN behavior (running stats updated).
    logits = model(x)
    loss = crit(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.first_step(zero_grad=True)
    # Second forward: freeze BN running buffers.
    with freeze_bn_running_stats(model):
        logits2 = model(x)
        loss2 = crit(logits2, y)
        loss2.backward()
    opt.second_step(zero_grad=True)
    return loss.item(), loss2.item()


def test_bn_running_stats_updated_on_first_forward():
    model = _toy_bn_model()
    opt = make_sam_optimizer(model, lr=0.0, momentum=0.0, weight_decay=0.0, rho=0.05)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(64, 8)
    y = torch.randint(0, 8, (64,))

    s0 = _snap_bn(model)
    _sam_step_correct(model, opt, crit, x, y)
    s1 = _snap_bn(model)

    # Every BN module's running_mean should have moved (any non-zero update).
    moved = sum(
        1
        for a, b in zip(s0, s1)
        if not torch.allclose(a[0], b[0]) or not torch.allclose(a[1], b[1])
    )
    assert moved == len(s0), f"Expected all {len(s0)} BN modules to update; only {moved} did."
    # num_batches_tracked should increment by exactly 1 per BN module per SAM step.
    for a, b in zip(s0, s1):
        assert b[2] == a[2] + 1, (
            f"num_batches_tracked went from {a[2]} to {b[2]}; expected +1 per SAM iteration"
        )


def test_bn_running_stats_frozen_on_second_forward():
    """After first forward (which updates), the second forward must NOT touch running stats."""
    model = _toy_bn_model()
    opt = make_sam_optimizer(model, lr=0.0, momentum=0.0, weight_decay=0.0, rho=0.05)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(64, 8)
    y = torch.randint(0, 8, (64,))

    # First forward only.
    model.train()
    logits = model(x); loss = crit(logits, y); opt.zero_grad(set_to_none=True); loss.backward()
    s_pre = _snap_bn(model)

    # Now do a second forward under freeze.
    with freeze_bn_running_stats(model):
        out = model(x)
        out.sum().backward()
    s_post = _snap_bn(model)

    for a, b in zip(s_pre, s_post):
        assert torch.allclose(a[0], b[0]), "running_mean changed during frozen forward"
        assert torch.allclose(a[1], b[1]), "running_var changed during frozen forward"
        assert a[2] == b[2], "num_batches_tracked changed during frozen forward"


def test_bn_counter_increments_once_per_sam_iteration():
    """Across multiple SAM iterations, num_batches_tracked grows by exactly 1 per iteration."""
    model = _toy_bn_model()
    opt = make_sam_optimizer(model, lr=0.01, momentum=0.0, weight_decay=0.0, rho=0.05)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(64, 8)
    y = torch.randint(0, 8, (64,))
    s0 = _snap_bn(model)
    for _ in range(3):
        _sam_step_correct(model, opt, crit, x, y)
    s1 = _snap_bn(model)
    for a, b in zip(s0, s1):
        assert b[2] == a[2] + 3, (
            f"After 3 SAM iterations num_batches_tracked should grow by 3; got {b[2] - a[2]}"
        )


if __name__ == "__main__":
    test_bn_running_stats_updated_on_first_forward()
    test_bn_running_stats_frozen_on_second_forward()
    test_bn_counter_increments_once_per_sam_iteration()
    print("All BN-handling tests passed.")
