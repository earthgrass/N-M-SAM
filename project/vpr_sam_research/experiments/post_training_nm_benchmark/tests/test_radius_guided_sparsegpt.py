import torch

from baseline_adapters.radius_guided_sparsegpt import radius_guided_sparsegpt_matrix
from baseline_adapters.sparsegpt import sparsegpt_matrix


def _case():
    torch.manual_seed(20261031)
    weight = torch.randn(5, 12, dtype=torch.float64)
    x = torch.randn(12, 41, dtype=torch.float64)
    hessian = 2 * x @ x.T / x.shape[1]
    return weight, hessian


def test_unit_radius_exactly_degenerates_to_original_sparsegpt():
    weight, hessian = _case()
    original = sparsegpt_matrix(weight, hessian, blocksize=8)
    hybrid = radius_guided_sparsegpt_matrix(weight, hessian, torch.zeros_like(weight), blocksize=8)
    assert torch.equal(hybrid.keep_mask, original.keep_mask)
    torch.testing.assert_close(hybrid.weight, original.weight, rtol=0, atol=0)


def test_constant_radius_preserves_original_mask_and_order():
    weight, hessian = _case()
    constant_log_radius = torch.full_like(weight, 1.75)
    result = radius_guided_sparsegpt_matrix(weight, hessian, constant_log_radius, blocksize=8)
    baseline = radius_guided_sparsegpt_matrix(weight, hessian, torch.zeros_like(weight), blocksize=8)
    assert torch.equal(result.keep_mask, baseline.keep_mask)
    valid = torch.isfinite(result.original_metric)
    original_order = result.original_metric[valid].reshape(-1, 4).argsort(dim=1, stable=True)
    hybrid_order = result.hybrid_metric[valid].reshape(-1, 4).argsort(dim=1, stable=True)
    assert torch.equal(original_order, hybrid_order)


def test_lambda_zero_exactly_degenerates_to_original_sparsegpt():
    weight, hessian = _case()
    log_radius = torch.linspace(-2, 2, weight.numel(), dtype=weight.dtype).reshape_as(weight)
    original = sparsegpt_matrix(weight, hessian, blocksize=8)
    hybrid = radius_guided_sparsegpt_matrix(
        weight, hessian, log_radius, blocksize=8, radius_lambda=0.0
    )
    assert torch.equal(hybrid.keep_mask, original.keep_mask)
    torch.testing.assert_close(hybrid.weight, original.weight, rtol=0, atol=0)


def test_radius_changes_only_selection_and_obs_remains_finite():
    weight, hessian = _case()
    log_radius = torch.zeros_like(weight)
    log_radius[:, 0::4] = 4.0
    log_radius[:, 1::4] = -4.0
    result = radius_guided_sparsegpt_matrix(weight, hessian, log_radius, blocksize=8)
    assert bool(torch.isfinite(result.weight).all())
    assert torch.count_nonzero(result.weight[~result.keep_mask]) == 0
    assert bool((torch.exp(log_radius) > 0).all())


def test_shape_and_finite_radius_gates():
    weight, hessian = _case()
    try:
        radius_guided_sparsegpt_matrix(weight, hessian, torch.zeros(1, dtype=weight.dtype))
    except ValueError as exc:
        assert "exactly match" in str(exc)
    else:
        raise AssertionError("shape mismatch was accepted")
    bad = torch.zeros_like(weight)
    bad[0, 0] = torch.nan
    try:
        radius_guided_sparsegpt_matrix(weight, hessian, bad)
    except ValueError as exc:
        assert "Non-finite" in str(exc)
    else:
        raise AssertionError("non-finite log-radius was accepted")
