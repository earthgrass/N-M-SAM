import pytest
import torch

from baseline_adapters.apbr_gpt_joint import build_candidate_factors, softmin_pruning_regularizer
from joint_optimization.calibration import (
    affine_zero_volume_projection,
    calibrate_eta_s,
    calibrate_lambda_p,
    combine_scale_updates,
)
from joint_optimization.training_objective import (
    differentiable_apbr_perturbations,
    joint_training_loss,
)


NAMES = ("layer1.weight", "layer2.weight")


def test_lambda_normal_ratio_fp64_accumulation_and_repeatability():
    ce = [torch.tensor([3.0, 4.0], dtype=torch.float32), torch.tensor([12.0])]
    prune = [torch.tensor([1.0, 2.0], dtype=torch.float32), torch.tensor([2.0])]
    a = calibrate_lambda_p(ce, prune, NAMES)
    b = calibrate_lambda_p(ce, prune, NAMES)
    assert a == b
    assert abs(a.achieved_ratio - 0.05) <= 5e-5
    assert a.ce_gradient_norm == 13.0


@pytest.mark.parametrize("which", ["ce", "prune"])
def test_lambda_rejects_zero_gradient(which):
    ce = [torch.ones(2), torch.ones(1)]
    prune = [torch.ones(2), torch.ones(1)]
    (ce if which == "ce" else prune)[:] = [torch.zeros(2), torch.zeros(1)]
    with pytest.raises(RuntimeError, match=which.upper() if which == "ce" else "Pruning"):
        calibrate_lambda_p(ce, prune, NAMES)


def test_lambda_rejects_outside_interval_without_clip():
    with pytest.raises(RuntimeError, match="outside"):
        calibrate_lambda_p([torch.ones(1)], [torch.tensor([1e20])], ["x"])


def test_lambda_requires_identical_eligible_parameter_set():
    with pytest.raises(ValueError, match="sets differ"):
        calibrate_lambda_p([torch.ones(1)], [torch.ones(1), torch.ones(1)], ["x"])


def test_eta_affine_analytic_target_constraint_and_determinism():
    scales = [torch.tensor([0.2, -0.2], dtype=torch.float64)]
    energy = [torch.tensor([0.1, -0.1], dtype=torch.float64)]
    gradient = [torch.tensor([2.0, -1.0], dtype=torch.float64)]
    a = calibrate_eta_s(scales, energy, gradient, lambda_p=0.2)
    b = calibrate_eta_s(scales, energy, gradient, lambda_p=0.2)
    assert a == b
    assert abs(a.achieved_ratio - 0.05) / 0.05 <= 1e-3
    updated, _ = combine_scale_updates(scales, energy, gradient, eta_s=a.eta_s, lambda_p=0.2)
    assert abs(float(sum(item.sum() for item in updated))) < 1e-12


def test_eta_nonlinear_projection_bisection():
    def nonlinear(values):
        bounded = [torch.tanh(value) for value in values]
        return affine_zero_volume_projection(bounded)
    result = calibrate_eta_s(
        [torch.zeros(4, dtype=torch.float64)],
        [torch.tensor([0.1, -0.1, 0.05, -0.05], dtype=torch.float64)],
        [torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64)],
        lambda_p=0.1, projection=nonlinear, projection_name="test_nonlinear", affine=False,
    )
    assert not result.analytic and result.iterations > 0
    assert abs(result.achieved_ratio - 0.05) / 0.05 <= 1e-3


def test_eta_rejects_zero_energy_and_zero_projected_gradient():
    scales = [torch.zeros(2, dtype=torch.float64)]
    with pytest.raises(RuntimeError, match="energy update is zero"):
        calibrate_eta_s(scales, [torch.zeros(2)], [torch.tensor([1.0, -1.0])], lambda_p=1.0)
    with pytest.raises(RuntimeError, match="gradient is zero"):
        calibrate_eta_s(scales, [torch.tensor([1.0, -1.0])], [torch.ones(2)], lambda_p=1.0)


def test_eta_unreachable_target_stops():
    def constant_projection(values):
        return [torch.zeros_like(value) for value in values]
    with pytest.raises(RuntimeError):
        calibrate_eta_s(
            [torch.zeros(2)], [torch.tensor([1.0, -1.0])], [torch.tensor([1.0, -1.0])],
            lambda_p=1.0, projection=constant_projection, affine=False,
        )


def test_original_energy_update_is_unchanged_by_affine_projection():
    scales = [torch.tensor([0.3, -0.2, -0.1], dtype=torch.float64)]
    energy = [torch.tensor([0.04, -0.01, -0.03], dtype=torch.float64)]
    projected = affine_zero_volume_projection([scales[0] + energy[0]])
    assert torch.allclose(projected[0] - scales[0], energy[0], atol=1e-15, rtol=0)


def test_differentiable_epsilon_stops_g_and_keeps_s_gradient():
    g = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    s = torch.tensor([0.1, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    epsilon = differentiable_apbr_perturbations([g], [s], rho=0.05)[0]
    epsilon.square().sum().backward()
    assert g.grad is None
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_q_stop_gradient_and_finite_joint_loss_lambda_zero_degeneracy():
    torch.manual_seed(4)
    x = torch.randn(8, 8, dtype=torch.float64)
    factors = build_candidate_factors(x @ x.T + torch.eye(8, dtype=torch.float64), compute_dtype=torch.float64)
    q = factors.inverse_qdd.detach().clone().requires_grad_()
    w = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    e = torch.randn(2, 8, dtype=torch.float64, requires_grad=True) * 0.01
    regularizer = softmin_pruning_regularizer(w, e, q, 1.0)
    ce = w.square().mean()
    loss = joint_training_loss(ce, regularizer, 0.0)
    assert torch.equal(loss, ce)
    loss.backward()
    assert q.grad is None
    assert torch.isfinite(loss)
