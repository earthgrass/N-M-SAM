from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
VPR_SRC = ROOT.parents[1] / "src"
sys.path.insert(0, str(VPR_SRC))

from blockwise_apbr import BlockwiseGeometry
from vpr_sam.optim.geometry import Geometry


def config():
    return {"gamma": 1e-4, "dtype": "float64", "warmup_steps": 0,
            "volume_atol": 1e-10, "volume_rtol": 1e-12, "static_std": 0.1,
            "block_degenerate_threshold": 1e-30}


def model():
    layer = nn.Sequential(nn.Conv2d(4, 2, 1, bias=False), nn.Linear(2, 2, bias=True)).float()
    for p in layer.parameters():
        p.grad = torch.arange(1, p.numel() + 1, dtype=torch.float32).reshape_as(p)
    return layer


def test_scheme4_perturbation_exactly_matches_frozen_original():
    a, b = model(), model()
    b.load_state_dict(a.state_dict())
    original = Geometry(a.named_parameters(), config(), "vpr", 0)
    scheme = BlockwiseGeometry(b.named_parameters(), config(), "global_block_feedback", 0, {"0.weight"})
    x, _ = original.perturbations(0.05)
    y, stats = scheme.perturbations(0.05)
    assert all(torch.equal(left, right) for left, right in zip(x, y))
    assert stats["perturbation_matches_original"] is True


def test_scheme1_equalizes_blocks_and_preserves_original_l2_and_residual():
    item = model()
    geometry = BlockwiseGeometry(item.named_parameters(), config(), "equal_block_energy", 0, {"0.weight"})
    original, _ = Geometry(model().named_parameters(), config(), "vpr", 0).perturbations(0.05)
    applied, stats = geometry.perturbations(0.05)
    groups = applied[0].reshape(2, 1, 4)
    norms = groups.double().square().sum(-1).sqrt().reshape(-1)
    torch.testing.assert_close(norms, norms[0].expand_as(norms), rtol=2e-6, atol=1e-12)
    assert torch.equal(applied[1], original[1])
    assert torch.equal(applied[2], original[2])
    original_l2 = sum(e.double().square().sum() for e in original).sqrt()
    applied_l2 = sum(e.double().square().sum() for e in applied).sqrt()
    torch.testing.assert_close(applied_l2, original_l2, rtol=2e-6, atol=1e-12)
    assert stats["perturbation_matches_original"] is False


def test_scheme1_preserves_zero_direction_and_equalizes_only_active_blocks():
    item = model()
    item[0].weight.grad[0].zero_()
    geometry = BlockwiseGeometry(item.named_parameters(), config(), "equal_block_energy", 0, {"0.weight"})
    original, _ = Geometry(item.named_parameters(), config(), "vpr", 0).perturbations(0.05)
    applied, stats = geometry.perturbations(0.05)
    norms = applied[0].reshape(2, 1, 4).double().square().sum(-1).sqrt().reshape(-1)
    assert norms[0] == 0
    assert norms[1] > 0
    assert torch.count_nonzero(applied[0][0]) == 0
    torch.testing.assert_close(
        sum(e.double().square().sum() for e in applied),
        sum(e.double().square().sum() for e in original), rtol=2e-6, atol=1e-12,
    )
    assert stats["degenerate_blocks"] == 1
    assert stats["active_blocks"] == 1


def test_zero_energy_block_has_neutral_feedback_while_active_block_uses_formula():
    item = model()
    geometry = BlockwiseGeometry(item.named_parameters(), config(), "global_block_feedback", 0, {"0.weight"})
    eps, _ = geometry.perturbations(0.05)
    eps[0][0].zero_()
    before = [s.clone() for s in geometry.logs]
    stats = geometry.prepare(eps, 0)
    after = geometry.candidate[0]
    delta = (after[0] - before[0]).reshape(2, 1, 4)
    assert torch.count_nonzero(delta[0]) == 0
    torch.testing.assert_close(delta[1].sum(-1), torch.zeros_like(delta[1].sum(-1)), atol=1e-12, rtol=0)
    assert stats["block_delta_sum_max"] < 1e-12


def test_block_and_residual_feedback_are_separately_zero_sum_and_finite():
    item = model()
    geometry = BlockwiseGeometry(item.named_parameters(), config(), "equal_block_energy", 0, {"0.weight"})
    eps, _ = geometry.perturbations(0.05)
    before = [s.clone() for s in geometry.logs]
    stats = geometry.prepare(eps, 0)
    after = geometry.candidate[0]
    block_delta = (after[0] - before[0]).reshape(2, 1, 4)
    torch.testing.assert_close(block_delta.sum(-1), torch.zeros_like(block_delta.sum(-1)), atol=1e-12, rtol=0)
    residual_sum = sum((a-b).double().sum() for a, b in zip(after[1:], before[1:]))
    assert abs(float(residual_sum)) < 1e-12
    assert stats["block_delta_sum_max"] < 1e-12
    assert all(bool(torch.isfinite(x).all()) for x in after)
