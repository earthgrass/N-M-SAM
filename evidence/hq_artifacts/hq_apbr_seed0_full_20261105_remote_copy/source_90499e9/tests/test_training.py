"""Bounded independent HQ runtime checks; no CIFAR Test is loaded."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from hq_apbr_v1.math_core import hq_global_l2_perturbation
from hq_apbr_v1.training import (
    HQStepper, atomic_checkpoint, build_checkpoint, capture_rng, cpu_copy,
    freeze_bn_running_stats, load_checkpoint, restore_rng, seed_all, train_epoch,
)


class TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 3)
        self.unused = nn.Parameter(torch.ones(2))

    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).mean((2, 3)))


@pytest.fixture(autouse=True)
def deterministic():
    seed_all(334)
    torch.set_num_threads(2)


def geometry(model, nonzero=False):
    value = torch.tensor([-1., -.5, .5, 1.], dtype=torch.float64).reshape_as(model.conv.weight)
    return {"conv.weight": value if nonzero else torch.zeros_like(value)}


def test_cached_direction_matches_log_reference_and_none_gradient():
    model = TinyBN()
    stepper = HQStepper(model, geometry(model, True))
    for name, parameter in model.named_parameters():
        parameter.grad = None if name == "unused" else torch.randn_like(parameter)
    before = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
    actual, norm, zero = stepper.perturbations()
    expected, reference_norm, reference_zero = hq_global_l2_perturbation(
        stepper.params, before, stepper.geometry, .05)
    assert not zero and not reference_zero
    assert norm == pytest.approx(reference_norm, abs=1e-10)
    for name in actual:
        assert torch.allclose(actual[name], expected[name], rtol=2e-7, atol=1e-9)
    assert model.unused.grad is None


def test_zero_geometry_matches_independent_standard_sam_and_bn_is_once():
    model = TinyBN()
    reference = copy.deepcopy(model)
    base_state = cpu_copy(model.state_dict())
    stepper = HQStepper(model, geometry(model))
    inputs, labels = torch.randn(5, 2, 3, 3), torch.tensor([0, 1, 2, 1, 0])
    actual = stepper.step(inputs, labels, audit=True)

    optimizer = torch.optim.SGD(reference.parameters(), lr=.1, momentum=.9, weight_decay=.0005)
    reference.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(reference(inputs), labels)
    loss.backward()
    params = dict(reference.named_parameters())
    g1 = {name: None if p.grad is None else p.grad.clone() for name, p in params.items()}
    norm = sum(p.grad.double().square().sum() for p in params.values() if p.grad is not None).sqrt()
    epsilon = {name: torch.zeros_like(p) if p.grad is None else (.05 * p.grad.double() / norm).to(p.dtype)
               for name, p in params.items()}
    saved = {name: p.detach().clone() for name, p in params.items()}
    with torch.no_grad():
        for name, p in params.items():
            p.add_(epsilon[name])
    optimizer.zero_grad(set_to_none=True)
    with freeze_bn_running_stats(reference):
        nn.functional.cross_entropy(reference(inputs), labels).backward()
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(saved[name])
    for name, p in params.items():
        if g1[name] is None:
            assert actual["audit"]["first_gradients"][name] is None
            assert actual["audit"]["optimizer_gradients"][name] is None
        else:
            assert torch.equal(actual["audit"]["first_gradients"][name], g1[name])
            assert torch.allclose(actual["audit"]["optimizer_gradients"][name], p.grad, atol=1e-7, rtol=1e-6)
        assert torch.allclose(actual["audit"]["epsilon"][name], epsilon[name], atol=1e-9, rtol=2e-7)
        assert torch.equal(actual["audit"]["restored_parameters"][name], base_state[name])
    optimizer.step()
    for name, value in model.state_dict().items():
        assert torch.allclose(value, reference.state_dict()[name], atol=1e-7, rtol=1e-6)
    assert int(model.bn.num_batches_tracked) == 1
    for name, value in model.named_buffers():
        assert torch.equal(value, actual["audit"]["bn_after_center"][name])
    assert all(p.grad is None for p in model.parameters())
    assert model.bn.track_running_stats
    assert actual["eps_norm"] == pytest.approx(.05, rel=2e-5)
    assert sum(actual["perturbation_energy_distribution"].values()) == pytest.approx(1., rel=1e-6)


def test_adversarial_exception_restores_parameters_buffers_and_poisons_stepper():
    model = TinyBN()
    initial = cpu_copy(model.state_dict())
    stepper = HQStepper(model, geometry(model, True))
    calls = 0

    def fail_second(_module, _inputs, _output):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-pass failure")

    handle = model.register_forward_hook(fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        stepper.step(torch.randn(5, 2, 3, 3), torch.tensor([0, 1, 2, 1, 0]))
    handle.remove()
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial[name])
    assert stepper.failed and not stepper.active and stepper.steps == 0
    assert model.bn.track_running_stats
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(RuntimeError, match="healthy"):
        stepper.state_dict()


def assert_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def test_runtime_checkpoint_resume_preserves_all_states(tmp_path):
    model = TinyBN()
    stepper = HQStepper(model, geometry(model, True))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stepper.optimizer, T_max=200)
    inputs = torch.randn(12, 2, 3, 3)
    labels = torch.tensor([0, 1, 2] * 4)

    def make_loader():
        order = torch.randperm(12)
        jitter = float(random.random()) + float(np.random.random())
        return [(inputs[order[i:i+4]] + .01 * jitter * torch.randn(4, 2, 3, 3),
                 labels[order[i:i+4]], order[i:i+4]) for i in range(0, 12, 4)]

    history = [train_epoch(stepper, make_loader(), "cpu", 0, expected_samples=12, expected_batches=3)]
    scheduler.step()
    identity = {"config_sha256": "frozen-unit-config", "source_commit": "unit"}
    payload = build_checkpoint(model, stepper, scheduler, epoch=0, history=history, identity=identity,
                               calibration_ids=[2, 5, 7], q_diags={"conv.weight": torch.tensor([1., 2.])},
                               calibration_report={"test_accessed": False})
    path = tmp_path / "last.pt"
    atomic_checkpoint(path, payload)
    continuation_loader = make_loader()
    row = train_epoch(stepper, continuation_loader, "cpu", 1, expected_samples=12, expected_batches=3)
    scheduler.step()
    expected = (cpu_copy(model.state_dict()), stepper.state_dict(), scheduler.state_dict(), capture_rng())

    restored_model = TinyBN()
    restored_stepper = HQStepper(restored_model, geometry(restored_model, True))
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_stepper.optimizer, T_max=200)
    restored = load_checkpoint(path, restored_model, restored_stepper, restored_scheduler, identity=identity)
    replay_loader = make_loader()
    assert_equal(continuation_loader, replay_loader)
    replay_row = train_epoch(restored_stepper, replay_loader, "cpu", restored["next_epoch"],
                             expected_samples=12, expected_batches=3)
    restored_scheduler.step()
    actual = (cpu_copy(restored_model.state_dict()), restored_stepper.state_dict(),
              restored_scheduler.state_dict(), capture_rng())
    assert_equal(actual, expected)
    assert replay_row["train_order_sha256"] == row["train_order_sha256"]
    assert restored["epoch_index"] == 0 and restored["completed_epochs"] == 1
    assert restored["test_accessed"] is False
    bad = copy.deepcopy(identity)
    bad["config_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, restored_model, restored_stepper, restored_scheduler, identity=bad)
