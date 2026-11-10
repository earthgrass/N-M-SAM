from __future__ import annotations

import itertools

import pytest
import torch

from baseline_adapters.fixedmask_obs import fixedmask_obs_matrix
from baseline_adapters.group_joint_obs import (
    APBR_SHUFFLE_SEEDS,
    DELETION_PAIRS,
    apbr_pair_costs,
    prepare_obs_system,
    select_group_mask,
    static_group_obs_costs,
)
from baseline_adapters.sparsegpt import sparsegpt_matrix


def spd_case(rows: int = 3, columns: int = 8):
    generator = torch.Generator().manual_seed(20261031)
    weight = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    root = torch.randn(columns, columns, generator=generator, dtype=torch.float64)
    hessian = root.T @ root + 0.3 * torch.eye(columns, dtype=torch.float64)
    geometry = 0.1 * torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    return weight, hessian, geometry


def test_pair_enumeration_is_complete_and_deterministic():
    assert DELETION_PAIRS == tuple(itertools.combinations(range(4), 2))
    assert len(DELETION_PAIRS) == 6
    assert len(set(DELETION_PAIRS)) == 6


def test_obs_system_reconstructs_direct_inverse_and_upper_factor():
    _weight, hessian, _geometry = spd_case()
    system = prepare_obs_system(hessian, percdamp=0.01)
    direct = torch.linalg.inv(system.damped_hessian)
    assert torch.allclose(system.inverse_hessian, direct, rtol=1e-12, atol=1e-12)
    reconstructed = system.inverse_factor.T @ system.inverse_factor
    assert torch.allclose(reconstructed, direct, rtol=1e-12, atol=1e-12)
    # A Cholesky-factor submatrix is not generally the corresponding Q block.
    indices = torch.tensor([2, 3])
    assert not torch.allclose(
        system.inverse_factor[indices][:, indices],
        system.inverse_hessian[indices][:, indices],
    )


def test_all_joint_costs_match_explicit_constrained_optimum():
    weight, hessian, _geometry = spd_case(rows=2, columns=8)
    system = prepare_obs_system(hessian, percdamp=0.01)
    costs, conditions = static_group_obs_costs(weight, system.inverse_hessian)
    assert costs.shape == (2, 2, 6)
    assert conditions.shape == (2, 6)
    for row in range(weight.shape[0]):
        for group in range(2):
            for candidate, local_pair in enumerate(DELETION_PAIRS):
                indices = torch.tensor([4 * group + x for x in local_pair])
                qdd = system.inverse_hessian[indices][:, indices]
                wd = weight[row, indices]
                multiplier = torch.linalg.solve(qdd, wd)
                delta = -system.inverse_hessian[:, indices] @ multiplier
                assert torch.allclose(delta[indices], -wd, rtol=1e-11, atol=1e-11)
                objective = 0.5 * delta @ system.damped_hessian @ delta
                assert torch.allclose(costs[row, group, candidate], objective, rtol=1e-10, atol=1e-11)


def test_single_coordinate_formula_is_recovered():
    weight, hessian, _geometry = spd_case(rows=1, columns=8)
    system = prepare_obs_system(hessian)
    coordinate = 5
    expected = 0.5 * weight[0, coordinate].square() / system.inverse_hessian[coordinate, coordinate]
    index = torch.tensor([coordinate])
    actual = 0.5 * weight[0, index] @ torch.linalg.solve(
        system.inverse_hessian[index][:, index], weight[0, index]
    )
    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_apbr_pair_cost_matches_direct_scalar_reference():
    weight, _hessian, geometry = spd_case(rows=2, columns=8)
    costs = apbr_pair_costs(weight, geometry)
    for row in range(2):
        for group in range(2):
            for candidate, pair in enumerate(DELETION_PAIRS):
                coordinates = [4 * group + x for x in pair]
                expected = sum(
                    weight[row, index].square() * torch.exp(-2 * geometry[row, index])
                    for index in coordinates
                )
                assert torch.allclose(costs[row, group, candidate], expected)


def test_near_optimal_apbr_selection_and_zero_tau_degeneracy():
    weight = torch.tensor([[1.0, 1.0, 1.004, 10.0]], dtype=torch.float64)
    system = prepare_obs_system(torch.eye(4, dtype=torch.float64), percdamp=0.0)
    geometry = torch.tensor([[0.0, 0.0, 4.0, 0.0]], dtype=torch.float64)
    group_only = select_group_mask(weight, system)
    zero_tau = select_group_mask(weight, system, log_radius=geometry, tau=0.0)
    hybrid = select_group_mask(weight, system, log_radius=geometry, tau=0.01)
    assert group_only.selected_pair_index.item() == 0
    assert zero_tau.selected_pair_index.item() == 0
    assert hybrid.candidate_count.item() == 3
    assert hybrid.selected_pair_index.item() == 1  # pair (0, 2)
    assert hybrid.relative_gpt_increase.item() <= 0.01 + 1e-12


def test_zero_minimum_does_not_expand_multiplicatively():
    weight = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    system = prepare_obs_system(torch.eye(4, dtype=torch.float64), percdamp=0.0)
    geometry = torch.zeros_like(weight)
    result = select_group_mask(weight, system, log_radius=geometry, tau=0.02)
    assert result.minimum_gpt_cost.item() == 0
    assert result.candidate_count.item() == 1
    assert result.selected_pair_index.item() == 0
    assert result.relative_gpt_increase.item() == 0


def test_float32_zero_minimum_has_no_unrelated_absolute_window():
    weight = torch.tensor([[0.0, 0.0, 1e-3, 1e-3]], dtype=torch.float32)
    system = prepare_obs_system(
        torch.eye(4, dtype=torch.float32), percdamp=0.0, compute_dtype=torch.float32
    )
    result = select_group_mask(
        weight, system, log_radius=torch.zeros_like(weight), tau=0.02
    )
    assert result.minimum_gpt_cost.item() == 0
    assert result.candidate_count.item() == 1
    assert result.numerical_tolerance.item() < 1e-10


def test_exact_ties_use_frozen_pair_order():
    weight = torch.ones(1, 4, dtype=torch.float64)
    system = prepare_obs_system(torch.eye(4, dtype=torch.float64), percdamp=0.0)
    result = select_group_mask(weight, system)
    assert result.selected_pair_index.item() == 0
    assert torch.equal(result.keep_mask, torch.tensor([[False, False, True, True]]))


def test_shuffled_geometry_is_deterministic_and_preserves_block_multisets():
    weight = torch.ones(2, 8, dtype=torch.float64)
    geometry = torch.arange(16, dtype=torch.float64).reshape(2, 8) / 10
    first = apbr_pair_costs(weight, geometry, shuffled_seed=APBR_SHUFFLE_SEEDS[0])
    second = apbr_pair_costs(weight, geometry, shuffled_seed=APBR_SHUFFLE_SEEDS[0])
    third = apbr_pair_costs(weight, geometry, shuffled_seed=APBR_SHUFFLE_SEEDS[1])
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    # The sum over all six pair costs counts every coordinate three times and
    # is therefore an invariant of within-block permutation.
    original = apbr_pair_costs(weight, geometry)
    assert torch.allclose(first.sum(-1), original.sum(-1), rtol=1e-12, atol=1e-12)


def test_strict_2of4_and_dense_tail_are_preserved():
    generator = torch.Generator().manual_seed(12)
    weight = torch.randn(4, 10, generator=generator, dtype=torch.float64)
    root = torch.randn(10, 10, generator=generator, dtype=torch.float64)
    system = prepare_obs_system(root.T @ root + torch.eye(10, dtype=torch.float64))
    geometry = torch.randn(4, 10, generator=generator, dtype=torch.float64)
    result = select_group_mask(weight, system, log_radius=geometry, tau=0.02)
    assert result.keep_mask.dtype == torch.bool
    assert bool((result.keep_mask[:, :8].reshape(-1, 4).sum(1) == 2).all())
    assert bool(result.keep_mask[:, 8:].all())
    assert bool(torch.isfinite(result.gpt_costs).all())
    assert bool(torch.isfinite(result.apbr_costs).all())


def test_ill_conditioned_qdd_is_a_hard_failure():
    weight = torch.ones(1, 4, dtype=torch.float64)
    inverse = torch.diag(torch.tensor([1e-12, 1.0, 1.0, 1.0], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="condition number"):
        static_group_obs_costs(weight, inverse)


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64])
def test_native_mask_replay_is_bitwise_identical_to_sparsegpt(compute_dtype):
    weight, hessian, _geometry = spd_case(rows=5, columns=12)
    native = sparsegpt_matrix(
        weight, hessian, blocksize=8, percdamp=0.01, compute_dtype=compute_dtype
    )
    replay = fixedmask_obs_matrix(
        weight, hessian, native.keep_mask,
        blocksize=8, percdamp=0.01, compute_dtype=compute_dtype,
    )
    assert torch.equal(replay.keep_mask, native.keep_mask)
    assert torch.equal(replay.weight, native.weight)
    assert torch.count_nonzero(replay.weight[~replay.keep_mask]) == 0
    assert replay.survivor_changed_count > 0
