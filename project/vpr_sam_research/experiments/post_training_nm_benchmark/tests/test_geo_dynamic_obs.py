"""Independent numerical gates for sequential four-column group OBS."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.geo_dynamic_obs import dynamic_group_obs_prune


def _data(seed=41, k=8, rows=4):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(k, k, generator=generator, dtype=torch.float64)
    h = x.T @ x + 2 * torch.eye(k, dtype=torch.float64)
    w = torch.randn(rows, k, generator=generator, dtype=torch.float64)
    s = torch.randn(rows, k, generator=generator, dtype=torch.float64) * .2
    return w, h, s


def _explicit_frozen_group_reference(w, h, s, percdamp=.01):
    """Reference via full inverse then reduced Hessian on future coordinates."""
    w = w.double().clone() * torch.exp(-s)
    h = h.double().clone()
    h.diagonal().add_(percdamp * h.diagonal().mean())
    q_full = h.inverse()
    pairs = torch.tensor([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    keep = torch.ones_like(w, dtype=torch.bool)
    for group in range(2):
        start = group * 4
        # All four previously processed coordinates are frozen. The second
        # conditional inverse is the inverse of the future Hessian submatrix.
        q = q_full if group == 0 else h[4:, 4:].inverse()
        current = w[:, start:]
        q_block = q[:4, :4]
        for row in range(w.shape[0]):
            costs = []
            for d in pairs:
                qdd = q_block[d][:, d]
                vd = current[row, d]
                costs.append(.5 * vd @ torch.linalg.solve(qdd, vd))
            d = pairs[int(torch.argmin(torch.stack(costs)))]
            wd = current[row, d].clone()
            displacement = -(q[:, d] @ torch.linalg.solve(q_block[d][:, d], wd))
            w[row, start:] += displacement
            keep[row, start + d] = False
            w[row, start + d] = 0
    return w * torch.exp(s), keep


def test_frozen_group_schur_and_explicit_reference():
    w, h, s = _data()
    result = dynamic_group_obs_prune(w, h, log_scale=s, mode="geo_full", blocksize=4)
    reference, mask = _explicit_frozen_group_reference(w, h, s)
    assert torch.equal(result.keep_mask, mask)
    assert torch.allclose(result.weight, reference, atol=1e-10, rtol=1e-10)
    hd = h.clone()
    hd.diagonal().add_(.01 * hd.diagonal().mean())
    q = hd.inverse()
    upper = torch.linalg.cholesky(q, upper=True)
    conditional = upper[4:, 4:].T @ upper[4:, 4:]
    assert torch.allclose(conditional, hd[4:, 4:].inverse(), atol=1e-10)


def test_attached_six_coordinate_claim_is_not_sequential_frozen_group():
    w, h, _ = _data(seed=7)
    hd = h.clone()
    hd.diagonal().add_(.01 * hd.diagonal().mean())
    q = hd.inverse()
    upper = torch.linalg.cholesky(q, upper=True)
    frozen_conditional = upper[4:, 4:].T @ upper[4:, 4:]
    remaining = torch.tensor([2, 3, 4, 5, 6, 7])
    six_coordinate_inverse = hd[remaining][:, remaining].inverse()[2:, 2:]
    assert not torch.allclose(frozen_conditional, six_coordinate_inverse, atol=1e-10)


def test_zero_geometry_reduces_all_modes_to_same_output():
    w, h, _ = _data()
    zero = torch.zeros_like(w)
    results = [dynamic_group_obs_prune(w, h, log_scale=zero, mode=mode, blocksize=4)
               for mode in ("std_group_obs", "geo_select_std_comp", "geo_full")]
    assert all(torch.equal(results[0].keep_mask, x.keep_mask) for x in results[1:])
    assert all(torch.allclose(results[0].weight, x.weight, atol=1e-12, rtol=0) for x in results[1:])


def test_isotropic_hessian_matches_normalized_magnitude_with_no_survivor_change():
    w, _, s = _data()
    h = 2.5 * torch.eye(8, dtype=torch.float64)
    for mode in ("geo_select_std_comp", "geo_full"):
        result = dynamic_group_obs_prune(w, h, log_scale=s, mode=mode, blocksize=8)
        expected = torch.ones_like(result.keep_mask)
        scores = (w * torch.exp(-s)).abs().reshape(w.shape[0], 2, 4)
        for group in range(2):
            deleted = scores[:, group].argsort(dim=1, stable=True)[:, :2]
            expected[torch.arange(w.shape[0])[:, None], group * 4 + deleted] = False
        assert torch.equal(result.keep_mask, expected)
        assert torch.allclose(result.weight[result.keep_mask], w[result.keep_mask], atol=1e-12)


@pytest.mark.parametrize("mode", ["std_group_obs", "geo_select_std_comp", "geo_full"])
def test_blocksize_invariance(mode):
    w, h, s = _data(k=32)
    reference = dynamic_group_obs_prune(w, h, log_scale=s, mode=mode, blocksize=4)
    for size in (8, 128):
        result = dynamic_group_obs_prune(w, h, log_scale=s, mode=mode, blocksize=size)
        assert torch.equal(result.keep_mask, reference.keep_mask)
        assert torch.allclose(result.weight, reference.weight, atol=1e-11, rtol=1e-11)
