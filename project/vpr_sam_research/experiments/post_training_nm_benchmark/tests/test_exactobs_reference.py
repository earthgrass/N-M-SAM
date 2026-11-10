import pytest
import torch
from torch import nn

from baseline_adapters.exactobs_reference import run_official_exactobs_row


def test_official_single_row_reference_uses_original_conv_weight_and_hessian():
    torch.manual_seed(7)
    conv = nn.Conv2d(4, 2, (1, 2), bias=False)
    original = conv.weight.detach().clone()
    x = torch.randn(8, 24)
    hessian = x @ x.T + torch.eye(8)
    hessian_before = hessian.clone()

    result = run_official_exactobs_row(conv, hessian, 1, relative_damping=0.01)

    assert torch.equal(conv.weight, original)
    assert torch.equal(hessian, hessian_before)
    assert result.original_weight.shape == (8,)
    assert result.compensated_weight.shape == (8,)
    assert result.keep_mask.dtype == torch.bool
    assert result.keep_mask.reshape(-1, 4).sum(1).tolist() == [2, 2]
    assert torch.count_nonzero(result.compensated_weight[~result.keep_mask]) == 0
    assert torch.isfinite(result.compensated_weight).all()
    assert result.cholesky_info == 0 and not result.fallback_used

    # ExactOBS's final surviving weights must solve the fixed-support local
    # quadratic for the support selected by the official greedy kernel.
    keep, drop = result.keep_mask, ~result.keep_mask
    damped = hessian + result.damping_value * torch.eye(8)
    optimum = result.original_weight[keep] + torch.linalg.solve(
        damped[keep][:, keep],
        damped[keep][:, drop] @ result.original_weight[drop],
    )
    assert torch.allclose(result.compensated_weight[keep], optimum, atol=3e-6, rtol=3e-5)
    delta = result.compensated_weight - result.original_weight
    physical_cost = 0.5 * delta @ hessian @ delta
    assert result.physical_quadratic_cost == pytest.approx(float(physical_cost), rel=1e-6)


def test_reference_rejects_unbounded_layer_and_bad_hessian():
    conv = nn.Conv2d(128, 1, 3, bias=False)
    with pytest.raises(ValueError, match="K <= 576"):
        run_official_exactobs_row(conv, torch.eye(1152), 0)
    small = nn.Conv2d(4, 1, (1, 2), bias=False)
    with pytest.raises(ValueError, match="shape/device"):
        run_official_exactobs_row(small, torch.eye(4), 0)
