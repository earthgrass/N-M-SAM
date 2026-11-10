import json
import sys
from pathlib import Path

import pytest
import torch

from joint_optimization.orchestration import (
    AtomicCheckpointManager,
    FinalEvaluationGuard,
    FormalSchedule,
    formal_manifest,
)
from baseline_adapters.apbr_gpt_joint import build_candidate_factors
from joint_optimization.joint_stepper import JointStepper

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from vpr_sam.optim.geometry import Geometry


def test_frozen_schedule_warmup_and_refresh_epochs():
    schedule = FormalSchedule()
    assert not schedule.pruning_active(4)
    assert schedule.pruning_active(5)
    assert [e for e in range(200) if schedule.refresh_due(e)] == list(range(5, 200, 10))
    assert len(formal_manifest()["hessian_refresh_epochs_zero_based"]) == 20


def test_atomic_checkpoint_roundtrip_and_hash(tmp_path):
    manager = AtomicCheckpointManager(tmp_path)
    path, digest = manager.save("last.pt", {"value": torch.arange(4), "completed_step": True})
    assert not path.with_suffix(".pt.tmp").exists()
    assert torch.equal(manager.load("last.pt", expected_sha256=digest)["value"], torch.arange(4))
    with pytest.raises(RuntimeError, match="SHA256"):
        manager.load("last.pt", expected_sha256="0" * 64)


def test_final_evaluation_requires_freeze_and_is_single_use(tmp_path):
    freeze = tmp_path / "freeze.json"
    output = tmp_path / "final.json"
    freeze.write_text(json.dumps({"status": "training", "checkpoint_sha256": {}}))
    guard = FinalEvaluationGuard(freeze)
    with pytest.raises(RuntimeError, match="frozen"):
        guard.run_once(lambda: {"accuracy": 0.0}, output)
    freeze.write_text(json.dumps({"status": "CHECKPOINTS_FROZEN", "checkpoint_sha256": {"final.pt": "a"}}))
    result = guard.run_once(lambda: {"accuracy": 0.5}, output)
    assert result["single_final_evaluation"] and output.exists()
    with pytest.raises(RuntimeError, match="already"):
        guard.run_once(lambda: {"accuracy": 0.6}, output)


def test_joint_stepper_warmup_active_and_checkpoint_roundtrip():
    torch.manual_seed(9)
    model = torch.nn.Linear(8, 3)
    geometry = Geometry(model.named_parameters(), {
        "gamma": 1e-5, "dtype": "float64", "warmup_steps": 0,
        "volume_atol": 1e-10, "volume_rtol": 1e-12, "static_std": 0.1,
    }, "vpr", 0)
    optimizer = torch.optim.SGD(geometry.params, lr=0.01, momentum=0.9)
    stepper = JointStepper(model, geometry, optimizer,
        eligible_parameter_names=("weight",), lambda_p=0.2, eta_s=0.01)
    hessian = torch.eye(8)
    factors = build_candidate_factors(hessian, compute_dtype=torch.float64)
    stepper.refresh({"weight": factors}, {"weight": 1.0})
    x, y = torch.randn(4, 8), torch.tensor([0, 1, 2, 1])
    warmup = stepper.step(x, y, pruning_active=False)
    active = stepper.step(x, y, pruning_active=True)
    assert not warmup["pruning_active"] and warmup["pruning_regularizer"] == 0
    assert active["pruning_active"] and torch.isfinite(torch.tensor(active["joint_loss"]))
    state = stepper.state_dict()
    assert state["steps"] == 2 and geometry.diagnostics()["log_volume"] == pytest.approx(0, abs=1e-10)


def test_joint_stepper_restores_parameters_when_adversarial_forward_fails():
    class FailSecond(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.randn(3,8));self.bias=torch.nn.Parameter(torch.zeros(3));self.calls=0
        def forward(self,x):
            self.calls+=1
            if self.calls==2: raise RuntimeError("forced adversarial failure")
            return torch.nn.functional.linear(x,self.weight,self.bias)
    torch.manual_seed(11);model=FailSecond();before={n:p.detach().clone() for n,p in model.named_parameters()}
    geometry=Geometry(model.named_parameters(),{"gamma":1e-5,"dtype":"float64","warmup_steps":0,"volume_atol":1e-10,"volume_rtol":1e-12,"static_std":.1},"vpr",0)
    optimizer=torch.optim.SGD(geometry.params,lr=.01)
    stepper=JointStepper(model,geometry,optimizer,eligible_parameter_names=(),lambda_p=0.,eta_s=0.)
    with pytest.raises(RuntimeError,match="forced"):
        stepper.step(torch.randn(4,8),torch.tensor([0,1,2,1]),pruning_active=False)
    assert stepper.failed and geometry.candidate is None
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters())
