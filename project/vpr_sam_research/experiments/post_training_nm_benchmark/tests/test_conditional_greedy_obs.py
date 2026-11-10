"""CPU numerical gates for approximate two-stage conditional 2:4 selection."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.conditional_greedy_obs import conditional_greedy_obs_prune
from baseline_adapters.geo_dynamic_obs import dynamic_group_obs_prune


def _matrix(seed=19, width=8, rows=3):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(width, width, generator=g, dtype=torch.float64)
    h = a @ a.T + torch.eye(width, dtype=torch.float64)
    w = torch.randn(rows, width, generator=g, dtype=torch.float64)
    return w, h


def _greedy_pair(w, q):
    """Independent scalar reference for one group and one output row."""
    singleton = [.5 * float(w[i] ** 2 / q[i, i]) for i in range(4)]
    first = min(range(4), key=lambda i: singleton[i])
    conditional = {}
    for j in range(4):
        if j == first:
            continue
        adjusted_w = w[j] - q[j, first] * w[first] / q[first, first]
        adjusted_q = q[j, j] - q[j, first] ** 2 / q[first, first]
        conditional[j] = .5 * float(adjusted_w ** 2 / adjusted_q)
    second = min(conditional, key=conditional.get)
    return first, second, singleton[first] + conditional[second]


def _explicit_sequential_reference(w, h, damping=.01):
    """Full reduced-Hessian solves after freezing every completed quartet."""
    h = h.double().clone()
    h.diagonal().add_(damping * h.diagonal().mean())
    output = w.double().clone()
    mask = torch.ones_like(output, dtype=torch.bool)
    for start in range(0, w.shape[1], 4):
        q = torch.linalg.inv(h[start:, start:])
        quartet = q[:4, :4]
        for row in range(output.shape[0]):
            first, second, _ = _greedy_pair(output[row, start:start + 4], quartet)
            d = torch.tensor([first, second])
            removed = output[row, start + d].clone()
            displacement = -(q[:, d] @ torch.linalg.solve(quartet[d][:, d], removed))
            output[row, start:] += displacement
            output[row, start + d] = 0
            mask[row, start + d] = False
    return output, mask


def test_diagonal_hessian_reduces_to_two_smallest_singleton_scores():
    w = torch.tensor([[.4, 3., -.2, 2., 4., -.5, .1, 3.]], dtype=torch.float64)
    h = torch.eye(8, dtype=torch.float64)
    result = conditional_greedy_obs_prune(w, h, blocksize=4, percdamp=0.0)
    expected = torch.tensor([[False, True, False, True, True, False, False, True]])
    assert torch.equal(result.keep_mask, expected)
    torch.testing.assert_close(result.weight, w * expected, rtol=0, atol=1e-12)
    validate_keep_mask(result.keep_mask, 2, 4)


def test_correlation_counterexample_exposes_greedy_approximation():
    q = torch.eye(4, dtype=torch.float64)
    q[1, 2] = q[2, 1] = .9
    h = torch.linalg.inv(q)
    w = torch.tensor([[.8, 1., 1., 2.]], dtype=torch.float64)
    greedy = conditional_greedy_obs_prune(w, h, blocksize=4, percdamp=0.0)
    oracle = dynamic_group_obs_prune(w, h, blocksize=4, percdamp=0.0)
    assert greedy.first_choices.item() == 0
    assert greedy.second_choices.item() in (1, 2)
    assert 0 in torch.where(~greedy.keep_mask[0])[0].tolist()
    assert torch.equal(torch.where(~oracle.keep_mask[0])[0], torch.tensor([1, 2]))
    torch.testing.assert_close(greedy.chosen_pair_cost[0, 0], torch.tensor(.82, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(oracle.candidate_costs[0, 0, 3], torch.tensor(1 / 1.9, dtype=torch.float64), atol=1e-12, rtol=0)


def test_conditioned_singleton_sum_equals_joint_two_coordinate_cost():
    _, h = _matrix(width=4)
    q = torch.linalg.inv(h)
    w = torch.tensor([.7, -.3, 1.2, .5], dtype=torch.float64)
    for first in range(4):
        for second in range(4):
            if first == second:
                continue
            a = .5 * w[first] ** 2 / q[first, first]
            updated_w = w[second] - q[second, first] * w[first] / q[first, first]
            updated_q = q[second, second] - q[second, first] ** 2 / q[first, first]
            sequential = a + .5 * updated_w ** 2 / updated_q
            d = torch.tensor([first, second])
            joint = .5 * w[d] @ torch.linalg.solve(q[d][:, d], w[d])
            torch.testing.assert_close(sequential, joint, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("blocksize", [4, 8, 128])
def test_full_small_matrix_matches_independent_frozen_group_reference(blocksize):
    w, h = _matrix()
    result = conditional_greedy_obs_prune(w, h, blocksize=blocksize)
    expected_weight, expected_mask = _explicit_sequential_reference(w, h)
    assert torch.equal(result.keep_mask, expected_mask)
    torch.testing.assert_close(result.weight, expected_weight, atol=1e-10, rtol=1e-10)
    assert bool(torch.isfinite(result.weight).all())
    assert int(torch.count_nonzero(result.weight[~result.keep_mask])) == 0
    validate_keep_mask(result.keep_mask)


def test_invalid_input_fails_closed():
    w = torch.ones(1, 8, dtype=torch.float64)
    h = torch.eye(8, dtype=torch.float64)
    with pytest.raises(ValueError):
        conditional_greedy_obs_prune(w[:, :7], h)
    with pytest.raises(ValueError):
        conditional_greedy_obs_prune(w, h, blocksize=6)
    h[0, 0] = torch.nan
    with pytest.raises(ValueError):
        conditional_greedy_obs_prune(w, h)
