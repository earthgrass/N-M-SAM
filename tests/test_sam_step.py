from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from torch import nn
from nmsam.core.batch import StepBatch
from nmsam.core.loss import LossReport, classification_loss
from nmsam.sam.reference_sgd import create_optimizer
from nmsam.sam.train_step import train_step
from nmsam.utils.seed import capture_rng


class FailBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError("injected backward failure")


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("phase", ["forward1", "forward2", "backward1", "backward2", "optimizer"])
def test_exception_transaction_restores_all_state(phase):
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4, momentum=None),
                          nn.Dropout(0.2), nn.Linear(4, 3))
    model.register_buffer("custom_counter", torch.tensor(0))
    batch = StepBatch(torch.randn(6, 4), torch.tensor([0, 1, 2, 0, 1, 2]))
    optimizer = create_optimizer(model.parameters())
    train_step(model, optimizer, batch, classification_loss)  # pre-existing momentum
    model.eval()
    model[0].train()
    before_model = deepcopy(model.state_dict())
    before_optimizer = deepcopy(optimizer.state_dict())
    before_grads = [p.grad.clone() for p in model.parameters()]
    before_rng = capture_rng()
    before_modes = [m.training for m in model.modules()]
    calls = 0
    updates = 0
    real_step = optimizer.step
    def failing_step():
        nonlocal updates
        updates += 1
        real_step()  # throw AFTER mutating parameters and momentum
        raise RuntimeError("injected optimizer failure")
    if phase == "optimizer":
        optimizer.step = failing_step
    def loss(m, b):
        nonlocal calls
        calls += 1
        random.random(); np.random.random(); torch.rand(1)
        m.custom_counter.add_(1)
        report = classification_loss(m, b)
        if phase == f"forward{calls}":
            raise RuntimeError("injected forward failure")
        if phase == f"backward{calls}":
            return LossReport(report.task_loss, report.l2_penalty,
                              FailBackward.apply(report.optimizer_loss), report.logits)
        return report
    with pytest.raises(RuntimeError, match="injected"):
        train_step(model, optimizer, batch, loss)
    assert_nested_equal(model.state_dict(), before_model)
    assert_nested_equal(optimizer.state_dict(), before_optimizer)
    for p, previous in zip(model.parameters(), before_grads):
        assert torch.equal(p.grad, previous)
    after_rng = capture_rng()
    assert torch.equal(after_rng.cpu, before_rng.cpu)
    assert after_rng.python == before_rng.python
    assert np.array_equal(after_rng.numpy[1], before_rng.numpy[1])
    assert after_rng.numpy[2:] == before_rng.numpy[2:]
    assert [m.training for m in model.modules()] == before_modes
    assert model[1].track_running_stats is True
    assert updates == (1 if phase == "optimizer" else 0)


def test_zero_direction_finite_no_parameter_drift():
    model = nn.Linear(2, 1)
    before = deepcopy(model.state_dict())
    def loss(m, b):
        task = sum(p.sum()*0 for p in m.parameters())
        return LossReport(task, task.new_zeros(()), task)
    report = train_step(model, create_optimizer(model.parameters()),
        StepBatch(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long)), loss)
    assert report.zero_direction and report.epsilon_norm == 0 and report.forward_passes == 2
    assert_nested_equal(model.state_dict(), before)


@pytest.mark.parametrize("rho", [-1, float("nan"), float("inf")])
def test_bad_rho_rejected(rho):
    model = nn.Linear(2, 3)
    with pytest.raises(ValueError):
        train_step(model, create_optimizer(model.parameters()),
            StepBatch(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long)), classification_loss, rho=rho)


def test_nonfinite_loss_rolls_back():
    model = nn.Linear(2, 1)
    before = deepcopy(model.state_dict())
    def loss(m, b):
        scalar = sum(p.sum() for p in m.parameters()) * float("nan")
        return LossReport(scalar, scalar, scalar)
    with pytest.raises(FloatingPointError):
        train_step(model, create_optimizer(model.parameters()),
            StepBatch(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long)), loss)
    assert_nested_equal(model.state_dict(), before)
    assert all(p.grad is None for p in model.parameters())

