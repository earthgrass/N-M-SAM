"""Tests for the sharpness probe (shared constrained perturbation)."""
from __future__ import annotations

import torch
import torch.nn as nn

from nm_sam_cifar10.metrics.sharpness import compute_sharpness
from nm_sam_cifar10.models import SimpleCNN
from nm_sam_cifar10.sparsity import SparsityConfig, attach_nm_sparsity
from nm_sam_cifar10.sparsity.parametrization import make_param_mask_fn


def _toy():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_sharpness_restores_parameters():
    torch.manual_seed(0)
    model = _toy()
    crit = nn.CrossEntropyLoss()
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    w0 = [p.detach().clone() for p in model.parameters()]
    L_c, L_a, sharp, info = compute_sharpness(
        model, x, y, crit, rho=0.05, device=torch.device("cpu")
    )
    for p, w0p in zip(model.parameters(), w0):
        assert torch.allclose(p, w0p), "compute_sharpness did not restore params"


def test_sharpness_does_not_modify_bn_state():
    """Before / after a sharpness probe, BN buffers are identical."""
    torch.manual_seed(0)
    model = SimpleCNN()
    crit = nn.CrossEntropyLoss()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    bn_before = [
        (m.running_mean.detach().clone(), m.running_var.detach().clone(),
         int(m.num_batches_tracked.item()))
        for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    L_c, L_a, sharp, info = compute_sharpness(
        model, x, y, crit, rho=0.05, device=torch.device("cpu")
    )
    bn_after = [
        (m.running_mean.detach().clone(), m.running_var.detach().clone(),
         int(m.num_batches_tracked.item()))
        for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    for a, b in zip(bn_before, bn_after):
        assert torch.allclose(a[0], b[0]), "BN running_mean changed"
        assert torch.allclose(a[1], b[1]), "BN running_var changed"
        assert a[2] == b[2], "BN num_batches_tracked changed"


def test_sharpness_epsilon_norm_equals_rho():
    torch.manual_seed(0)
    model = _toy()
    crit = nn.CrossEntropyLoss()
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    L_c, L_a, sharp, info = compute_sharpness(
        model, x, y, crit, rho=0.05, device=torch.device("cpu")
    )
    assert abs(info["eps_norm"] - 0.05) < 1e-5


def test_sharpness_sparse_epsilon_norm_equals_rho():
    torch.manual_seed(0)
    model = SimpleCNN()
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    attach_nm_sparsity(model, cfg)
    mask_fn = make_param_mask_fn(model)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    L_c, L_a, sharp, info = compute_sharpness(
        model, x, y, crit, rho=0.05, device=torch.device("cpu"), mask_fn=mask_fn
    )
    assert abs(info["eps_norm"] - 0.05) < 1e-5


if __name__ == "__main__":
    test_sharpness_restores_parameters()
    test_sharpness_does_not_modify_bn_state()
    test_sharpness_epsilon_norm_equals_rho()
    test_sharpness_sparse_epsilon_norm_equals_rho()
    print("All sharpness tests passed.")
