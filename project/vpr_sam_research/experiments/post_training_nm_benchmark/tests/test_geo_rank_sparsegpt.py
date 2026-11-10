import torch

from baseline_adapters.geo_rank_sparsegpt import geo_rank_sparsegpt_matrix
from baseline_adapters.sparsegpt import sparsegpt_matrix


def _inputs():
    generator = torch.Generator().manual_seed(20261103)
    weight = torch.randn((3, 12), dtype=torch.float64, generator=generator)
    x = torch.randn((12, 32), dtype=torch.float64, generator=generator)
    hessian = 2 * (x @ x.T) / x.shape[1]
    return weight, hessian


def test_zero_geometry_bitwise_matches_native_sparsegpt():
    weight, hessian = _inputs()
    native = sparsegpt_matrix(weight, hessian, blocksize=8)
    guided = geo_rank_sparsegpt_matrix(
        weight, hessian, torch.zeros_like(weight), blocksize=8,
        compute_dtype=torch.float64,
    )
    assert torch.equal(guided.keep_mask, native.keep_mask)
    assert torch.equal(guided.weight, native.weight)


def test_exact_normalized_weight_score_and_physical_obs():
    weight, hessian = _inputs()
    s = torch.zeros_like(weight)
    s[:, 0::4] = 1.2
    s[:, 1::4] = -0.9
    guided = geo_rank_sparsegpt_matrix(
        weight, hessian, s, blocksize=8, compute_dtype=torch.float64,
    )
    valid = torch.isfinite(guided.original_metric)
    torch.testing.assert_close(
        guided.hybrid_metric[valid],
        guided.original_metric[valid] * torch.exp(-2 * s[valid]),
        rtol=1e-14, atol=1e-14,
    )
    assert guided.keep_mask.dtype == torch.bool
    assert torch.equal(guided.keep_mask.reshape(3, 3, 4).sum(-1),
                       torch.full((3, 3), 2))
    assert torch.count_nonzero(guided.weight[~guided.keep_mask]) == 0
    assert bool(torch.isfinite(guided.weight).all())


def test_constant_geometry_keeps_native_result():
    weight, hessian = _inputs()
    native = sparsegpt_matrix(weight, hessian, blocksize=8)
    guided = geo_rank_sparsegpt_matrix(
        weight, hessian, torch.full_like(weight, 0.3), blocksize=8,
        compute_dtype=torch.float64,
    )
    assert torch.equal(guided.keep_mask, native.keep_mask)
    assert torch.equal(guided.weight, native.weight)


def test_geometry_selects_two_individual_scores_without_pair_enumeration():
    weight = torch.tensor([[0.2, 0.3, 0.4, 0.5]], dtype=torch.float64)
    hessian = torch.eye(4, dtype=torch.float64)
    s = torch.tensor([[0.0, 0.0, 0.0, 2.0]], dtype=torch.float64)
    native = sparsegpt_matrix(weight, hessian, blocksize=4)
    guided = geo_rank_sparsegpt_matrix(
        weight, hessian, s, blocksize=4, compute_dtype=torch.float64,
    )
    assert torch.equal(native.keep_mask[0], torch.tensor([False, False, True, True]))
    assert torch.equal(guided.keep_mask[0], torch.tensor([False, True, True, False]))
    torch.testing.assert_close(
        guided.weight[0], torch.tensor([0.0, 0.3, 0.4, 0.0], dtype=torch.float64),
        rtol=0, atol=0,
    )
