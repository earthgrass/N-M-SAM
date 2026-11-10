"""Numerical gates for geometric six-pair support selection."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from baseline_adapters.fixedmask_obs import fixedmask_obs_matrix
from baseline_adapters.group_joint_obs import DELETION_PAIRS, prepare_obs_system, static_group_obs_costs
from run_geo_obs_seed0 import six_pair_keep


def test_congruence_cost_matches_explicit_rowwise_inverse():
    h = torch.tensor([[2.0, .2, .1, .0], [.2, 1.5, .1, .0],
                      [.1, .1, 1.2, .3], [.0, .0, .3, 1.8]], dtype=torch.float64)
    w = torch.tensor([[.4, -.6, .2, .9]], dtype=torch.float64)
    a = torch.tensor([[1.4, .8, 1.1, .7]], dtype=torch.float64)
    q = h.inverse()
    costs, _ = static_group_obs_costs(w / a, q)
    row_q = torch.diag(a[0]) @ q @ torch.diag(a[0])
    for k, pair in enumerate(DELETION_PAIRS):
        d = list(pair)
        expected = .5 * (w[0, d] @ torch.linalg.solve(row_q[d][:, d], w[0, d]))
        assert torch.allclose(costs[0, 0, k], expected, atol=1e-12, rtol=1e-12)


def test_identity_geometry_reduces_to_group_obs_and_tail_is_dense():
    h = torch.eye(5, dtype=torch.float64)
    w = torch.tensor([[.1, .2, 2., 3., 4.]], dtype=torch.float64)
    q = h.inverse()
    plain, _ = static_group_obs_costs(w, q)
    geo, _ = static_group_obs_costs(w / torch.ones_like(w), q)
    keep, chosen = six_pair_keep(geo, 5)
    assert torch.equal(plain, geo)
    assert chosen.item() == 0
    assert torch.equal(keep, torch.tensor([[False, False, True, True, True]]))


def test_fixedmask_obs_compensates_original_weight_on_geometric_support():
    h = torch.tensor([[2.0, .2, .1, .0], [.2, 1.5, .1, .0],
                      [.1, .1, 1.2, .3], [.0, .0, .3, 1.8]], dtype=torch.float64)
    w = torch.tensor([[.4, -.6, .2, .9]], dtype=torch.float64)
    a = torch.tensor([[1.4, .8, 1.1, .7]], dtype=torch.float64)
    q = prepare_obs_system(h, percdamp=.01, compute_dtype=torch.float64).inverse_hessian
    costs, _ = static_group_obs_costs(w / a, q)
    keep, _ = six_pair_keep(costs, 4)
    result = fixedmask_obs_matrix(w, h, keep, blocksize=4, compute_dtype=torch.float64)
    assert result.keep_mask.dtype == torch.bool
    assert torch.equal(result.weight[~keep], torch.zeros_like(result.weight[~keep]))
    assert torch.isfinite(result.weight).all()
    assert result.survivor_changed_count > 0
    # The compensator starts from physical w, not v=w/a.
    reference = fixedmask_obs_matrix(w, h, keep, blocksize=4, compute_dtype=torch.float64)
    assert torch.equal(result.weight, reference.weight)
