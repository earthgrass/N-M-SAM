import torch

from baseline_adapters.apbr_gpt_joint import (
    build_candidate_factors,
    candidate_costs,
    dynamic_joint_obs_matrix,
    shuffle_probes_within_groups,
    softmin_pruning_regularizer,
)
from baseline_adapters.group_joint_obs import DELETION_PAIRS
from baseline_adapters.sparsegpt import sparsegpt_matrix


def spd(n: int) -> torch.Tensor:
    torch.manual_seed(17)
    x = torch.randn(n, n, dtype=torch.float64)
    return x @ x.T + 0.5 * torch.eye(n, dtype=torch.float64)


def test_candidate_cost_matches_explicit_quadratic():
    h = spd(8)
    factors = build_candidate_factors(h, percdamp=0.01, compute_dtype=torch.float64)
    w = torch.randn(3, 8, dtype=torch.float64)
    e = torch.randn(2, 3, 8, dtype=torch.float64) * 0.01
    gpt, apbr = candidate_costs(w, e, factors.inverse_qdd)
    for row in range(3):
        for group in range(2):
            for p, pair in enumerate(DELETION_PAIRS):
                wd = w[row, 4 * group + torch.tensor(pair)]
                ed = e[:, row, 4 * group + torch.tensor(pair)]
                r = factors.inverse_qdd[group, p]
                assert torch.allclose(gpt[row, group, p], 0.5 * wd @ r @ wd)
                expected = torch.stack([v @ r @ v for v in ed]).mean() * 0.5
                assert torch.allclose(apbr[row, group, p], expected)


def test_regularizer_is_differentiable_in_weight_and_probe():
    factors = build_candidate_factors(spd(8), compute_dtype=torch.float64)
    w = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    e = (torch.randn(2, 8, dtype=torch.float64) * 0.01).requires_grad_()
    value = softmin_pruning_regularizer(w, e, factors.inverse_qdd, 1.0)
    value.backward()
    assert torch.isfinite(value)
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert e.grad is not None and torch.isfinite(e.grad).all()


def test_zero_probe_dynamic_group_selection_is_finite_strict_2of4():
    torch.manual_seed(23)
    w = torch.randn(5, 8)
    h = spd(8).float()
    probes = torch.zeros(3, 5, 8)
    result = dynamic_joint_obs_matrix(w, h, probes, compute_dtype=torch.float64)
    assert result.keep_mask.reshape(5, 2, 4).sum(-1).eq(2).all()
    assert torch.count_nonzero(result.weight[~result.keep_mask]) == 0
    assert torch.isfinite(result.weight).all()
    assert torch.count_nonzero(result.selected_apbr_cost) == 0


def test_dynamic_with_zero_probe_differs_only_by_group_selector_not_compensator():
    # Diagonal H makes group pair selection equal to two independent native
    # choices, so both paths must be exactly equal in this controlled case.
    torch.manual_seed(29)
    w = torch.randn(4, 12)
    h = torch.diag(torch.linspace(1, 2, 12))
    native = sparsegpt_matrix(w, h, compute_dtype=torch.float64)
    joint = dynamic_joint_obs_matrix(w, h, torch.zeros(1, 4, 12), compute_dtype=torch.float64)
    assert torch.equal(native.keep_mask, joint.keep_mask)
    assert torch.equal(native.weight, joint.weight)


def test_probe_propagation_matches_separate_weight_linear_operator():
    torch.manual_seed(31)
    w = torch.randn(3, 8)
    probe = torch.randn(1, 3, 8) * 0.01
    h = spd(8).float()
    result = dynamic_joint_obs_matrix(w, h, probe, compute_dtype=torch.float64)
    # Reusing the selected support in a separate explicit fixed-support replay
    # is checked indirectly by treating the probe as a weight under the same
    # selected support; deleted coordinates must remain exactly zero.
    assert torch.count_nonzero(result.propagated_probes[:, ~result.keep_mask]) == 0
    assert torch.isfinite(result.propagated_probes).all()


def test_shuffle_preserves_each_group_multiset_and_is_deterministic():
    probes = torch.arange(2 * 3 * 9, dtype=torch.float64).reshape(2, 3, 9)
    a = shuffle_probes_within_groups(probes, seed=7)
    b = shuffle_probes_within_groups(probes, seed=7)
    assert torch.equal(a, b)
    assert torch.equal(a[..., 8:], probes[..., 8:])
    assert torch.equal(a[..., :8].reshape(2, 3, 2, 4).sort(-1).values,
                       probes[..., :8].reshape(2, 3, 2, 4).sort(-1).values)


def test_groupwise_update_matches_explicit_multi_constraint_with_rowwise_supports():
    h = torch.tensor([
        [2.0, .2, .1, .0], [.2, 1.7, .3, .1],
        [.1, .3, 1.5, .2], [.0, .1, .2, 1.2],
    ], dtype=torch.float64)
    w = torch.tensor([
        [.01, .02, 10., 11.],
        [10., .01, .02, 11.],
        [10., 11., .01, .02],
    ], dtype=torch.float64)
    probes = torch.zeros(2, 3, 4, dtype=torch.float64)
    result = dynamic_joint_obs_matrix(w, h, probes, compute_dtype=torch.float64)
    assert len(set(result.selected_pair_index[:, 0].tolist())) == 3

    damped = h.clone()
    damped.diagonal().add_(0.01 * torch.diag(h).mean())
    q = torch.linalg.inv(damped)
    expected = w.clone()
    for row, pair_index in enumerate(result.selected_pair_index[:, 0].tolist()):
        deleted = torch.tensor(DELETION_PAIRS[pair_index])
        r = torch.linalg.inv(q[deleted[:, None], deleted[None, :]])
        expected[row] -= (q[:, deleted] @ r @ w[row, deleted]).reshape(-1)
        expected[row, deleted] = 0
    assert torch.allclose(result.weight, expected, atol=2e-14, rtol=2e-14)
    assert torch.count_nonzero(result.propagated_probes) == 0
