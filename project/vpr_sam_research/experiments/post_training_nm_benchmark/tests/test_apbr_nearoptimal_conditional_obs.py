"""Independent CPU gates for APBR's near-optimal conditional OBS selector."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.apbr_nearoptimal_conditional_obs import (
    apbr_nearoptimal_conditional_obs_prune,
    shuffle_log_scale_within_quartets,
)
from baseline_adapters.conditional_greedy_obs import conditional_greedy_obs_prune


def _spd(seed=20261103, width=8, rows=3, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(width, width, generator=generator, dtype=dtype)
    h = a @ a.T + torch.eye(width, dtype=dtype)
    w = torch.randn(rows, width, generator=generator, dtype=dtype)
    s = torch.randn(rows, width, generator=generator, dtype=dtype) * 0.2
    return w, h, s


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("blocksize", [4, 8, 128])
def test_tau_zero_bitwise_replays_historical_adapter_for_all_geometry_modes(dtype, blocksize):
    w, h, s = _spd(dtype=dtype)
    old = conditional_greedy_obs_prune(w, h, blocksize=blocksize)
    for mode in ("physical", "unit", "true", "shuffled"):
        new = apbr_nearoptimal_conditional_obs_prune(
            w, h, s, tau=0, geometry_mode=mode, shuffle_seed=7,
            blocksize=blocksize,
        )
        assert torch.equal(new.keep_mask, old.keep_mask)
        assert torch.equal(new.first_choices, old.first_choices)
        assert torch.equal(new.second_choices, old.second_choices)
        assert torch.equal(new.weight, old.weight)
        assert torch.equal(new.chosen_pair_cost, old.chosen_pair_cost)


def test_tau_zero_preserves_historical_argmin_on_exact_physical_ties():
    w = torch.tensor([[1., -1., 4., 5.]], dtype=torch.float64)
    h = torch.eye(4, dtype=torch.float64)
    s = torch.tensor([[-5., 5., 2., -2.]], dtype=torch.float64)
    old = conditional_greedy_obs_prune(w, h, blocksize=4, percdamp=0)
    new = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=0, geometry_mode="true", blocksize=4, percdamp=0,
    )
    assert int(new.first_choices[0, 0]) == 0
    assert torch.equal(new.first_choices, old.first_choices)
    assert torch.equal(new.second_choices, old.second_choices)
    assert torch.equal(new.weight, old.weight)


def test_physical_only_ignores_tau_and_log_geometry():
    w, h, s = _spd()
    old = conditional_greedy_obs_prune(w, h, blocksize=4)
    new = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=0.5, geometry_mode="physical", blocksize=4,
    )
    assert torch.equal(new.keep_mask, old.keep_mask)
    assert torch.equal(new.weight, old.weight)
    assert bool((new.first_candidate_counts >= 1).all())
    assert bool((new.second_candidate_counts >= 1).all())


def test_true_geometry_can_override_only_inside_near_optimal_window():
    w = torch.tensor([[1., 1.01, 3., 4.]], dtype=torch.float64)
    h = torch.eye(4, dtype=torch.float64)
    s = torch.tensor([[0., 0.1, 0., 0.]], dtype=torch.float64)
    inside = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=0.03, geometry_mode="true", blocksize=4, percdamp=0,
    )
    outside = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=0.02, geometry_mode="true", blocksize=4, percdamp=0,
    )
    assert int(inside.first_choices[0, 0]) == 1
    assert int(inside.first_candidate_counts[0, 0]) == 2
    assert int(outside.first_choices[0, 0]) == 0
    assert int(outside.first_candidate_counts[0, 0]) == 1
    assert inside.first_selected_cost[0, 0] <= 1.03 * inside.first_min_cost[0, 0]


def test_second_geometry_decision_uses_virtual_weight_and_conditioned_cost():
    w = torch.tensor([[1., 1.05, 1.06, 4.]], dtype=torch.float64)
    h = torch.eye(4, dtype=torch.float64)
    s = torch.tensor([[0., 0., 0.03, 0.]], dtype=torch.float64)
    baseline = apbr_nearoptimal_conditional_obs_prune(
        w, h, tau=0.05, geometry_mode="physical", blocksize=4, percdamp=0,
    )
    true = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=0.05, geometry_mode="true", blocksize=4, percdamp=0,
    )
    assert int(true.first_choices[0, 0]) == 0
    assert int(true.first_candidate_counts[0, 0]) == 1
    assert int(baseline.second_choices[0, 0]) == 1
    assert int(true.second_choices[0, 0]) == 2
    assert int(true.second_candidate_counts[0, 0]) == 2
    assert true.second_selected_cost[0, 0] <= 1.05 * true.second_min_cost[0, 0]


def test_second_geometry_tie_break_does_not_reuse_predeletion_weight():
    # After virtually deleting coordinate 0, the two physical candidates
    # become w'_1=0.4 and w'_2=0.6. Raw w/a would prefer 2, but w'/a prefers 1.
    q = torch.eye(4, dtype=torch.float64)
    q[0, 1] = q[1, 0] = 0.5
    q[0, 2] = q[2, 0] = -0.5
    h = torch.linalg.inv(q)
    w = torch.tensor([[0.2, 0.5, 0.5, 2.0]], dtype=torch.float64)
    s = torch.tensor([[0., 0., 0.2, 0.]], dtype=torch.float64)
    result = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=1.3, geometry_mode="true", blocksize=4, percdamp=0,
    )
    assert int(result.first_choices[0, 0]) == 0
    assert int(result.second_candidate_counts[0, 0]) == 2
    assert w[0, 2] * torch.exp(-s[0, 2]) < w[0, 1]
    assert 0.6 * torch.exp(-s[0, 2]) > 0.4
    assert int(result.second_choices[0, 0]) == 1


def test_unit_geometry_is_distinct_from_physical_only_when_tau_is_positive():
    # OBS prefers coordinate 1 because its inverse curvature is large; unit
    # geometry prefers coordinate 0 because its raw |w| is smaller.
    w = torch.tensor([[1., 2., 4., 5.]], dtype=torch.float64)
    h = torch.diag(torch.tensor([1., 0.125, 1., 1.], dtype=torch.float64))
    physical = apbr_nearoptimal_conditional_obs_prune(
        w, h, tau=1.1, geometry_mode="physical", blocksize=4, percdamp=0,
    )
    unit = apbr_nearoptimal_conditional_obs_prune(
        w, h, tau=1.1, geometry_mode="unit", blocksize=4, percdamp=0,
    )
    assert int(physical.first_choices[0, 0]) == 1
    assert int(unit.first_choices[0, 0]) == 0


def test_shuffled_geometry_is_seeded_and_preserves_each_quartet_multiset():
    s = torch.arange(24, dtype=torch.float64).reshape(3, 8)
    a = shuffle_log_scale_within_quartets(s, 11)
    b = shuffle_log_scale_within_quartets(s, 11)
    assert torch.equal(a, b)
    assert torch.equal(a.reshape(3, 2, 4).sort(-1).values,
                       s.reshape(3, 2, 4).sort(-1).values)
    w, h, _ = _spd()
    x = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=0.3, geometry_mode="shuffled", shuffle_seed=11,
    )
    y = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=0.3, geometry_mode="shuffled", shuffle_seed=11,
    )
    assert torch.equal(x.keep_mask, y.keep_mask)
    assert torch.equal(x.weight, y.weight)


def test_physical_joint_obs_update_matches_independent_constraint_solve():
    generator = torch.Generator().manual_seed(27)
    a = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    h = a @ a.T + torch.eye(4, dtype=torch.float64)
    w = torch.tensor([[0.7, -0.3, 1.2, 0.5]], dtype=torch.float64)
    s = torch.tensor([[0.2, -0.6, 0.9, -0.3]], dtype=torch.float64)
    result = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=1.0, geometry_mode="true", blocksize=4, percdamp=0,
    )
    q = torch.linalg.inv(h)
    deleted = torch.where(~result.keep_mask[0])[0]
    lam = torch.linalg.solve(q[deleted][:, deleted], w[0, deleted])
    expected = w[0] - q[:, deleted] @ lam
    expected[deleted] = 0
    torch.testing.assert_close(result.weight[0], expected, rtol=1e-12, atol=1e-12)
    assert result.keep_mask.dtype == torch.bool
    assert int(result.keep_mask.sum()) == 2
    assert int(torch.count_nonzero(result.weight[~result.keep_mask])) == 0


def test_multigroup_output_is_finite_and_strict_two_of_four():
    w, h, s = _spd(width=12, rows=5)
    result = apbr_nearoptimal_conditional_obs_prune(
        w, h, s, tau=0.05, geometry_mode="true", blocksize=8,
    )
    assert result.keep_mask.shape == w.shape
    assert torch.equal(result.keep_mask.reshape(5, 3, 4).sum(-1),
                       torch.full((5, 3), 2))
    assert bool(torch.isfinite(result.weight).all())
    assert int(torch.count_nonzero(result.weight[~result.keep_mask])) == 0
    assert bool((result.first_selected_cost <= 1.05 * result.first_min_cost).all())
    assert bool((result.second_selected_cost <= 1.05 * result.second_min_cost).all())


@pytest.mark.parametrize("bad_tau", [-0.01, float("nan"), float("inf")])
def test_invalid_tau_fails_closed(bad_tau):
    w = torch.ones((1, 4), dtype=torch.float64)
    with pytest.raises(ValueError):
        apbr_nearoptimal_conditional_obs_prune(w, torch.eye(4), tau=bad_tau)


def test_missing_or_nonfinite_geometry_and_shuffle_seed_fail_closed():
    w = torch.ones((1, 4), dtype=torch.float64)
    h = torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError):
        apbr_nearoptimal_conditional_obs_prune(w, h, geometry_mode="true")
    with pytest.raises(ValueError):
        apbr_nearoptimal_conditional_obs_prune(
            w, h, torch.zeros(1, 8), geometry_mode="true",
        )
    with pytest.raises(ValueError):
        apbr_nearoptimal_conditional_obs_prune(
            w, h, torch.full_like(w, float("nan")), geometry_mode="true",
        )
    with pytest.raises(ValueError):
        apbr_nearoptimal_conditional_obs_prune(
            w, h, torch.zeros_like(w), geometry_mode="shuffled",
        )
    with pytest.raises(ValueError):
        apbr_nearoptimal_conditional_obs_prune(
            w, h, percdamp=float("nan"),
        )
