"""Tests for the sparse SAM perturbation radius (P-fix).

For both sparse and dense models the global epsilon norm must equal rho.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nm_sam_cifar10.models import SimpleCNN
from nm_sam_cifar10.optim.sam import (
    SAM,
    build_constrained_directions,
    freeze_bn_running_stats,
    global_direction_norm,
    make_sam_optimizer,
)
from nm_sam_cifar10.sparsity import SparsityConfig, attach_nm_sparsity
from nm_sam_cifar10.sparsity.parametrization import make_param_mask_fn


def _toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_dense_global_epsilon_norm_equals_rho():
    model = _toy_model()
    opt = make_sam_optimizer(model, lr=0.0, momentum=0.0, weight_decay=0.0, rho=0.05)
    x = torch.randn(8, 8)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    crit = nn.CrossEntropyLoss()

    model.train()
    loss = crit(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    info = opt.first_step(zero_grad=True)

    # Validate that the cached eps_norm matches rho.
    assert abs(info["eps_norm"] - 0.05) < 1e-5, (
        f"Global epsilon norm = {info['eps_norm']}, expected 0.05"
    )


def test_sparse_global_epsilon_norm_equals_rho():
    """Even when ~50% of weights are masked out, ||epsilon|| == rho."""
    torch.manual_seed(0)
    model = SimpleCNN()
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    attach_nm_sparsity(model, cfg)
    mask_fn = make_param_mask_fn(model)

    opt = make_sam_optimizer(model, lr=0.0, momentum=0.0, weight_decay=0.0, rho=0.05)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    crit = nn.CrossEntropyLoss()

    model.train()
    loss = crit(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    info = opt.first_step(zero_grad=True, mask_fn=mask_fn)

    assert abs(info["eps_norm"] - 0.05) < 1e-5, (
        f"Sparse global epsilon norm = {info['eps_norm']}, expected 0.05"
    )


def test_sparse_epsilon_zero_at_masked_positions():
    """The perturbation is zero wherever the mask is zero (master weight untouched there)."""
    torch.manual_seed(0)
    model = SimpleCNN()
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    attach_nm_sparsity(model, cfg)
    mask_fn = make_param_mask_fn(model)

    opt = make_sam_optimizer(model, lr=0.0, momentum=0.0, weight_decay=0.0, rho=0.05)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    crit = nn.CrossEntropyLoss()

    model.train()
    loss = crit(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.first_step(zero_grad=True, mask_fn=mask_fn)

    # For each sparse wrapper, the cached epsilon should be zero at masked positions.
    for group in opt.param_groups:
        for p in group["params"]:
            if "e" not in opt.state[p]:
                continue
            e = opt.state[p]["e"]
            m = mask_fn(p)
            if m is None:
                continue
            assert torch.all(e[m == 0] == 0.0), "Sparse epsilon non-zero at masked positions"


def test_parameter_restore_to_float_precision():
    """After SAM.first_step + manually subtracting cached epsilon, the
    parameter values are exactly the original values (modulo the
    base-optimizer's own numerical noise at lr=0).

    This test isolates the restore mechanism from the SGD step's
    floating-point noise by manually performing only the sub_epsilon part
    of second_step.
    """
    torch.manual_seed(0)
    model = SimpleCNN()
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    attach_nm_sparsity(model, cfg)
    mask_fn = make_param_mask_fn(model)

    opt = make_sam_optimizer(model, lr=0.0, momentum=0.0, weight_decay=0.0, rho=0.05)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    crit = nn.CrossEntropyLoss()

    w0 = [p.detach().clone() for p in model.parameters()]
    model.train()
    loss = crit(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.first_step(zero_grad=True, mask_fn=mask_fn)

    # Confirm perturbation actually moved the parameters.
    w_after_first = [p.detach().clone() for p in model.parameters()]
    for p, w0p, wfp in zip(model.parameters(), w0, w_after_first):
        assert (wfp - w0p).abs().max().item() > 0, (
            "Perturbation didn't actually move the parameter."
        )

    # Now manually perform the sub_epsilon part of second_step (no SGD noise).
    with torch.no_grad():
        for group in opt.param_groups:
            for p in group["params"]:
                if "e" in opt.state[p]:
                    p.sub_(opt.state[p]["e"])

    # Every parameter should equal w0 to float32 precision (~1e-6).
    for p, w0p in zip(model.parameters(), w0):
        diff = (p - w0p).abs().max().item()
        assert diff < 1e-5, f"Parameter not restored to float32 precision: max diff = {diff:.3e}"


if __name__ == "__main__":
    test_dense_global_epsilon_norm_equals_rho()
    test_sparse_global_epsilon_norm_equals_rho()
    test_sparse_epsilon_zero_at_masked_positions()
    test_parameter_restore_to_float_precision()
    print("All sparse SAM radius tests passed.")
