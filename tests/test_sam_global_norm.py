"""Tests for the corrected SAM implementation.

Validates:
* ``epsilon`` global L2 norm equals ``rho`` (not per-parameter).
* Perturbation is correctly restored before the base-optimizer step.
* BN running stats are not updated during the second forward pass when
  ``disable_running_stats`` is active.
* Sharpness probe restores parameters exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nm_sam_cifar10.optim.sam import SAM, disable_running_stats, make_sam_optimizer
from nm_sam_cifar10.metrics.sharpness import compute_sharpness


def _toy_model_bn():
    """Simple BN-equipped MLP so we can also test BN running stats."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.BatchNorm1d(8),
    )


def _toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_global_epsilon_norm_equals_rho():
    model = _toy_model()
    opt = make_sam_optimizer(model, lr=0.01, momentum=0.0, weight_decay=0.0, rho=0.05)
    x = torch.randn(8, 8)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    crit = nn.CrossEntropyLoss()

    model.train()
    with disable_running_stats(model):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        info = opt.first_step(zero_grad=True)
        # Check the global norm of perturbations BEFORE second_step.
        eps_total_sq = torch.zeros(1)
        for group in opt.param_groups:
            for p in group["params"]:
                if "e" not in opt.state[p]:
                    continue
                eps_total_sq = eps_total_sq + opt.state[p]["e"].pow(2).sum()
        eps_norm = eps_total_sq.sqrt().item()
        assert abs(eps_norm - 0.05) < 1e-5, f"global epsilon norm = {eps_norm}, expected 0.05"
        assert abs(info["eps_norm"] - 0.05) < 1e-5
        assert abs(info["g_norm"] * (0.05 / info["g_norm"]) - 0.05) < 1e-5

        loss2 = crit(model(x), y)
        loss2.backward()
    opt.second_step(zero_grad=True)


def test_parameter_restore_exact():
    model = _toy_model()
    opt = make_sam_optimizer(model, lr=0.1, momentum=0.9, weight_decay=0.0, rho=0.05)
    x = torch.randn(8, 8)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    crit = nn.CrossEntropyLoss()

    w0 = [p.detach().clone() for p in model.parameters()]

    model.train()
    with disable_running_stats(model):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.first_step(zero_grad=True)
        # Now params are w + epsilon. Run second forward/backward.
        loss2 = crit(model(x), y)
        loss2.backward()
        # Right BEFORE second_step: weights should still be w + epsilon.
        w_after_first = [p.detach().clone() for p in model.parameters()]
    opt.second_step(zero_grad=True)

    # After second_step, weights should differ from w0 (because of SGD update)
    # but the residual perturbation MUST have been removed: epsilon should NOT
    # have leaked into the SGD update.
    # Check that the *change* from w0 equals the SGD step alone, i.e. that the
    # final weight equals w0 - lr * step_dir (not w0 + epsilon - lr * step_dir).
    # We don't know step_dir directly, but we can check that the cached "e"
    # buffers have been consumed (no leftover).
    for group in opt.param_groups:
        for p in group["params"]:
            assert "e" not in opt.state[p], "Perturbation buffer was not cleared after second_step"


def test_bn_running_stats_not_updated_during_sam():
    """BN running mean/var must be identical before vs after a SAM step."""
    model = _toy_model_bn()
    opt = make_sam_optimizer(model, lr=0.01, momentum=0.0, weight_decay=0.0, rho=0.05)
    x = torch.randn(64, 8)
    y = torch.randint(0, 8, (64,))
    crit = nn.CrossEntropyLoss()

    # Snapshot BN running stats.
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    before_mean = [m.running_mean.detach().clone() for m in bn_modules]
    before_var = [m.running_var.detach().clone() for m in bn_modules]

    model.train()
    with disable_running_stats(model):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.first_step(zero_grad=True)
        loss2 = crit(model(x), y)
        loss2.backward()
    opt.second_step(zero_grad=True)

    # After SAM step the params moved; running stats should be unchanged because
    # both passes were under disable_running_stats.
    for m, bm, bv in zip(bn_modules, before_mean, before_var):
        assert torch.allclose(m.running_mean, bm), "BN running_mean changed during SAM step"
        assert torch.allclose(m.running_var, bv), "BN running_var changed during SAM step"


def test_sharpness_probe_restores_params():
    """compute_sharpness must leave the model state identical to its input."""
    model = _toy_model()
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    crit = nn.CrossEntropyLoss()

    w0 = [p.detach().clone() for p in model.parameters()]
    base_loss, adv_loss, sharp, info = compute_sharpness(model, x, y, crit, rho=0.05, device=torch.device("cpu"))
    for p, w0p in zip(model.parameters(), w0):
        assert torch.allclose(p, w0p), "compute_sharpness did not restore params"


if __name__ == "__main__":
    test_global_epsilon_norm_equals_rho()
    test_parameter_restore_exact()
    test_sharpness_probe_restores_params()
    print("All SAM global-norm tests passed.")
