"""CPU numerical gates for native-score conditional SparseGPT selection."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.gpt_score_conditional import gpt_score_conditional_prune
from baseline_adapters.sparsegpt.adapter import sparsegpt_matrix


def _upper(h: torch.Tensor, damping: float = 0.0) -> torch.Tensor:
    damped = h.double().clone()
    damped.diagonal().add_(damping * damped.diagonal().mean())
    return torch.linalg.cholesky(torch.linalg.inv(damped), upper=True)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    upper = torch.tensor([
        [1., -.9, 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ], dtype=torch.float64)
    hessian = torch.linalg.inv(upper.T @ upper)
    weight = torch.tensor([[.5, .6, .7, 2.]], dtype=torch.float64)
    return weight, hessian


def _native_fixed_mask_output(weight: torch.Tensor, upper: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Independent scalar replay of SparseGPT's original column update."""
    current = weight.clone()
    output = torch.empty_like(weight)
    for j in range(weight.shape[0]):
        output[j] = current[j] if keep[j] else 0
        error = (current[j] - output[j]) / upper[j, j]
        current[j:] -= error * upper[j, j:]
    return output


def test_first_choice_is_exact_native_score_with_stable_tie():
    g = torch.Generator().manual_seed(317)
    a = torch.randn(4, 4, generator=g, dtype=torch.float64)
    h = a @ a.T + torch.eye(4, dtype=torch.float64)
    w = torch.randn(8, 4, generator=g, dtype=torch.float64)
    w[0] = 1.0  # A stable tie can still arise with diagonal H in the next test.
    result = gpt_score_conditional_prune(w, h, blocksize=4, percdamp=.01)
    upper = _upper(h, .01)
    native_metric = w.square() / torch.diag(upper).square()[None, :]
    expected = native_metric.argsort(dim=1, stable=True)[:, 0]
    assert torch.equal(result.first_choices[:, 0].long(), expected)


def test_diagonal_hessian_matches_native_mask_and_output_with_dense_tail():
    w = torch.tensor([[.3, 2., .2, 1., 3., .6, .1, 4., -.7, .8]], dtype=torch.float64)
    h = torch.diag(torch.tensor([1., 2., 3., 4., 2., 3., 4., 5., 1., 1.], dtype=torch.float64))
    native = sparsegpt_matrix(w, h, blocksize=4, percdamp=0.0)
    variant = gpt_score_conditional_prune(w, h, blocksize=4, percdamp=0.0)
    assert torch.equal(variant.keep_mask, native.keep_mask)
    assert torch.equal(variant.weight, native.weight)
    assert bool(variant.keep_mask[:, 8:].all())
    validate_keep_mask(variant.keep_mask)


def test_correlated_second_choice_changes_while_first_is_native():
    w, h = _fixture()
    native = sparsegpt_matrix(w, h, blocksize=4, percdamp=0.0)
    variant = gpt_score_conditional_prune(w, h, blocksize=4, percdamp=0.0)
    assert variant.first_choices.item() == 0
    assert variant.native_second_choices.item() == 1
    assert variant.second_choices.item() == 2
    assert torch.equal(torch.where(~native.keep_mask[0])[0], torch.tensor([0, 1]))
    assert torch.equal(torch.where(~variant.keep_mask[0])[0], torch.tensor([0, 2]))
    upper = _upper(h)
    expected = _native_fixed_mask_output(w[0], upper, variant.keep_mask[0])
    torch.testing.assert_close(variant.weight[0], expected, rtol=0, atol=1e-12)


def test_second_choice_matches_independent_schur_and_cholesky_reference():
    g = torch.Generator().manual_seed(888)
    a = torch.randn(4, 4, generator=g, dtype=torch.float64)
    h = a @ a.T + torch.eye(4, dtype=torch.float64)
    w = torch.randn(3, 4, generator=g, dtype=torch.float64)
    result = gpt_score_conditional_prune(w, h, blocksize=4, percdamp=0.0)
    upper = _upper(h)
    inverse = torch.linalg.inv(h)
    for row in range(w.shape[0]):
        first = int(result.first_choices[row, 0])
        assert first == int((w[row].square() / torch.diag(upper).square()).argmin())
        rest = [j for j in range(4) if j != first]
        q_reduced = inverse[rest][:, rest] - torch.outer(inverse[rest, first], inverse[first, rest]) / inverse[first, first]
        u_reduced = torch.linalg.cholesky(q_reduced, upper=True)
        virtual = w[row, rest] - inverse[rest, first] * w[row, first] / inverse[first, first]
        expected_second = rest[int((virtual.square() / torch.diag(u_reduced).square()).argmin())]
        assert int(result.second_choices[row, 0]) == expected_second


@pytest.mark.parametrize("blocksize", [4, 8, 128])
def test_blocksize_invariance_and_exact_two_of_four(blocksize):
    g = torch.Generator().manual_seed(91)
    a = torch.randn(8, 8, generator=g, dtype=torch.float64)
    h = a @ a.T + torch.eye(8, dtype=torch.float64)
    w = torch.randn(3, 8, generator=g, dtype=torch.float64)
    result = gpt_score_conditional_prune(w, h, blocksize=blocksize)
    reference = gpt_score_conditional_prune(w, h, blocksize=128)
    assert torch.equal(result.keep_mask, reference.keep_mask)
    torch.testing.assert_close(result.weight, reference.weight, rtol=1e-12, atol=1e-12)
    assert bool(torch.isfinite(result.weight).all())
    assert int(torch.count_nonzero(result.weight[~result.keep_mask])) == 0
    validate_keep_mask(result.keep_mask)


def test_bad_inputs_fail_closed():
    w = torch.ones(1, 4, dtype=torch.float64)
    h = torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError):
        gpt_score_conditional_prune(w[:, :3], h)
    with pytest.raises(ValueError):
        gpt_score_conditional_prune(w, h, blocksize=6)
    with pytest.raises(ValueError):
        gpt_score_conditional_prune(w, h, percdamp=-.1)
    h[0, 0] = torch.nan
    with pytest.raises(ValueError):
        gpt_score_conditional_prune(w, h)
