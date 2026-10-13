from copy import deepcopy
from functools import partial
import pytest
import torch
from torch import nn
from nmsam.core.batch import StepBatch
from nmsam.core.loss import LossReport, classification_loss
from nmsam.sam.perturbation import global_norm
from nmsam.sam.reference_sgd import create_optimizer
from nmsam.sam.train_step import train_step


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor([3.], dtype=torch.float64))


def quadratic_loss(model, batch):
    w = torch.cat(tuple(model.parameters()))
    h = w.new_tensor([[3., 1., 0.], [1., 4., 1.], [0., 1., 2.]])
    task = 0.5 * w @ h @ w
    return LossReport(task, task.new_zeros(()), task)


def toy_batch():
    return StepBatch(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long))


def cat(tensors):
    return torch.cat([t.flatten() for t in tensors])


def test_analytic_gradient_locations_global_epsilon_restore_and_clip():
    model = Quadratic()
    h = torch.tensor([[3., 1., 0.], [1., 4., 1.], [0., 1., 2.]], dtype=torch.float64)
    w = cat(tuple(model.parameters())).detach().clone()
    opt = create_optimizer(model.parameters(), lr=0.1, beta=0, nesterov=False)
    report = train_step(model, opt, toy_batch(), quadratic_loss, rho=0.3,
                        gradient_clipping=2.0, capture_trace=True)
    t = report.trace
    g1 = h @ w
    epsilon = 0.3 * g1 / g1.norm()
    g2 = h @ (w + epsilon)
    clipped = 2.0 * g2 / g2.norm()
    torch.testing.assert_close(cat(t.g1), g1)
    torch.testing.assert_close(cat(t.epsilon), epsilon)
    torch.testing.assert_close(cat(t.probe), w + epsilon)
    torch.testing.assert_close(cat(t.g2), g2)
    torch.testing.assert_close(cat(t.restored), w, rtol=0, atol=0)
    torch.testing.assert_close(cat(t.clipped), clipped)
    torch.testing.assert_close(cat(tuple(model.parameters())), w - 0.1 * clipped)
    assert report.epsilon_norm == pytest.approx(0.3, abs=1e-12)
    assert report.restoration_error == 0
    assert report.clipped_gradient_norm == pytest.approx(2.0)
    # Check that clipping g1 would change the tested perturbation location only
    # through normalization (we instead explicitly assert the raw g1 trace).
    assert report.g1_norm > 2 and report.g2_norm > 2


@pytest.mark.parametrize("rho", [0., 0.1])
@pytest.mark.parametrize("nesterov", [False, True])
def test_four_step_momentum_nesterov_recurrence(rho, nesterov):
    model = Quadratic()
    opt = create_optimizer(model.parameters(), lr=0.02, beta=0.7, nesterov=nesterov)
    w = cat(tuple(model.parameters())).detach().clone()
    h = w.new_tensor([[3., 1., 0.], [1., 4., 1.], [0., 1., 2.]])
    buffer = torch.zeros_like(w)
    for _ in range(4):
        g1 = h @ w
        g = h @ (w + rho * g1 / g1.norm()) if rho else g1
        buffer = 0.7 * buffer + g
        w = w - 0.02 * (g + 0.7 * buffer if nesterov else buffer)
        report = train_step(model, opt, toy_batch(), quadratic_loss, rho=rho, gradient_clipping=0)
        torch.testing.assert_close(cat(tuple(model.parameters())), w)
        assert report.forward_passes == (2 if rho else 1)


@pytest.mark.parametrize("exclude_1d", [False, True])
def test_explicit_l2_in_both_gradients_and_no_double_decay(exclude_1d):
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Tanh(), nn.Linear(4, 3)).double()
    batch = StepBatch(torch.randn(5, 4, dtype=torch.float64), torch.tensor([0, 1, 2, 1, 0]))
    opt = create_optimizer(model.parameters(), beta=0, nesterov=False)
    task_only = partial(classification_loss, l2_reg=0)
    regularized = partial(classification_loss, l2_reg=0.2, no_weight_decay_on_bn=exclude_1d)
    # Independent reference gradients at each actual weight location.
    reference = deepcopy(model)
    task_grad = torch.autograd.grad(task_only(reference, batch).optimizer_loss, tuple(reference.parameters()))
    report = train_step(model, opt, batch, regularized, rho=0.05, gradient_clipping=0, capture_trace=True)
    for p, tg, g1 in zip(reference.parameters(), task_grad, report.trace.g1):
        expected = tg + (0.2 * p if not exclude_1d or p.ndim > 1 else 0)
        torch.testing.assert_close(g1, expected)
    with torch.no_grad():
        for p, probe in zip(reference.parameters(), report.trace.probe):
            p.copy_(probe)
    reference.train()
    for m in reference.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False
    loss2 = regularized(reference, batch)
    expected2 = torch.autograd.grad(loss2.optimizer_loss, tuple(reference.parameters()))
    for actual, expected in zip(report.trace.g2, expected2):
        torch.testing.assert_close(actual, expected)
    assert report.optimizer_loss == pytest.approx(report.task_loss + report.l2_penalty)
    assert all(g["weight_decay"] == 0 for g in opt.param_groups)


def test_reject_optimizer_weight_decay_before_mutation():
    model = Quadratic()
    before = deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.2)
    with pytest.raises(ValueError, match="weight_decay"):
        train_step(model, optimizer, toy_batch(), quadratic_loss)
    for name, p in model.state_dict().items():
        assert torch.equal(p, before[name])


def test_rho_zero_equals_independent_sgd_with_clipped_regularized_loss():
    model = nn.Linear(4, 3).double()
    reference = deepcopy(model)
    batch = StepBatch(torch.randn(5, 4, dtype=torch.float64), torch.tensor([0, 1, 2, 1, 0]))
    opt = create_optimizer(model.parameters(), lr=0.03)
    refopt = create_optimizer(reference.parameters(), lr=0.03)
    loss = partial(classification_loss, l2_reg=0.2)
    calls = []
    def counted(m, b):
        calls.append(b)
        return loss(m, b)
    for _ in range(3):
        report = train_step(model, opt, batch, counted, rho=0, gradient_clipping=0.3)
        refopt.zero_grad()
        loss(reference, batch).optimizer_loss.backward()
        norm = global_norm(tuple(p.grad for p in reference.parameters()))
        if norm >= 0.3:
            for p in reference.parameters():
                p.grad = (p.grad / norm) * 0.3
        refopt.step()
        for p, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, expected)
        assert report.forward_passes == 1 and report.epsilon_norm == 0
    assert len(calls) == 3

