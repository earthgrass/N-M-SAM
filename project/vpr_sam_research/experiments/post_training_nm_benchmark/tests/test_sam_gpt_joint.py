from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from baseline_adapters.apbr_gpt_joint import build_candidate_factors
from joint_optimization.sam_joint_stepper import SAMJointStepper
from vpr_sam.optim import Stepper
from vpr_sam.optim.geometry import Geometry


GEOMETRY = {"gamma": 5.482145661475941e-8, "dtype": "float64", "warmup_steps": 0,
            "volume_atol": 1e-10, "volume_rtol": 1e-12, "static_std": .1}


def model():
    torch.manual_seed(41)
    return nn.Sequential(nn.Linear(4, 2))


def test_sam_joint_inactive_matches_original_sam_step():
    left = model(); right = copy.deepcopy(left)
    config = {"seed": 0, "optimizer": {"method": "sam", "lr": .01, "momentum": 0.,
        "weight_decay": 0., "nesterov": False, "rho": .05}, "geometry": GEOMETRY}
    original = Stepper(left, config)
    geometry = Geometry(right.named_parameters(), GEOMETRY, "sam", 0)
    optimizer = torch.optim.SGD(geometry.params, lr=.01)
    joint = SAMJointStepper(right, geometry, optimizer, eligible_parameter_names=("0.weight",),
                            lambda_p=1.0, temperature=.1, rho=.05)
    torch.manual_seed(43); x = torch.randn(8, 4); y = torch.randint(0, 2, (8,))
    a = original.step(x, y); b = joint.step(x, y, pruning_active=False)
    assert abs(a["eps_norm"] - b["eps_norm"]) < 1e-12
    for p, q in zip(left.parameters(), right.parameters()):
        assert torch.equal(p, q)


def test_sam_joint_active_uses_regularizer_without_geometry_scales():
    network = model(); geometry = Geometry(network.named_parameters(), GEOMETRY, "sam", 0)
    optimizer = torch.optim.SGD(geometry.params, lr=.01)
    stepper = SAMJointStepper(network, geometry, optimizer, eligible_parameter_names=("0.weight",),
                              lambda_p=1.0, temperature=.1, rho=.05)
    hessian = torch.eye(4, dtype=torch.float64)
    stepper.refresh({"0.weight": build_candidate_factors(hessian, compute_dtype=torch.float64)},
                    {"0.weight": 1.0})
    torch.manual_seed(47); result = stepper.step(torch.randn(8, 4), torch.randint(0, 2, (8,)),
                                                pruning_active=True)
    assert result["pruning_active"] is True
    assert result["eps_norm"] > 0
    assert result["pruning_weight_grad_norm"] > 0
    assert result["restoration_max_abs"] == 0
    assert geometry.logs == []
    assert all(torch.isfinite(parameter).all() for parameter in network.parameters())
