"""Correctness gates for pre-update magnitude, SAM-g1 and SAM-g2 masks."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from nm_sam_cifar10.optim.sam import make_sam_optimizer
from nm_sam_cifar10.sparsity import SparsityConfig
from nm_sam_cifar10.sparsity.mask_manager import MaskManager
from nm_sam_cifar10.sparsity.parametrization import make_param_mask_fn
from nm_sam_cifar10.sparsity.perturbation import (
    build_constrained_directions,
    global_direction_norm,
)
from nm_sam_cifar10.trainers.sam_step import sam_step


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def build_sparse(seed=0, score="magnitude"):
    torch.manual_seed(seed)
    model = TinyNet()
    manager = MaskManager(
        model, SparsityConfig(n=2, m=4, enabled=True, score=score), scope="phase1"
    )
    return model, manager, make_param_mask_fn(model)


def optimizer_for(model, lr=0.1):
    return make_sam_optimizer(
        model, lr=lr, momentum=0.0, weight_decay=0.0, rho=0.05
    )


def test_g1_and_g2_capture_match_the_two_actual_backwards():
    torch.manual_seed(4)
    x = torch.randn(6, 8)
    y = torch.randint(0, 4, (6,))
    criterion = nn.CrossEntropyLoss()

    model, manager, mask_fn = build_sparse(seed=9, score="taylor_g2")
    manual = copy.deepcopy(model)
    manual_mask_fn = make_param_mask_fn(manual)

    criterion(manual(x), y).backward()
    expected_g1 = [w.weight_master.grad.detach().clone() for w in manual._nm_wrappers]
    directions = build_constrained_directions(manual.parameters(), manual_mask_fn, False)
    norm = global_direction_norm(directions)
    with torch.no_grad():
        for parameter, direction in directions.items():
            parameter.add_(direction * (0.05 / norm))
    manual.zero_grad(set_to_none=True)
    criterion(manual(x), y).backward()
    expected_g2 = [w.weight_master.grad.detach().clone() for w in manual._nm_wrappers]

    result = sam_step(
        model, optimizer_for(model), criterion, x, y, torch.device("cpu"),
        param_mask_fn=mask_fn, mask_score="taylor_g2", capture_gradients=True,
    )
    assert all(torch.equal(a, b) for a, b in zip(result.g1_snapshots, expected_g1))
    assert all(torch.equal(a, b) for a, b in zip(result.g2_snapshots, expected_g2))
    assert any(not torch.equal(a, b) for a, b in zip(result.g1_snapshots, result.g2_snapshots))


def test_g1_is_not_stale_previous_step_gradient():
    model, _manager, mask_fn = build_sparse(seed=3, score="taylor_g1")
    optimizer = optimizer_for(model)
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(10)
    first = sam_step(
        model, optimizer, criterion, torch.randn(6, 8), torch.randint(0, 4, (6,)),
        torch.device("cpu"), param_mask_fn=mask_fn, mask_score="taylor_g1",
        capture_gradients=True,
    )
    second = sam_step(
        model, optimizer, criterion, torch.randn(6, 8), torch.randint(0, 4, (6,)),
        torch.device("cpu"), param_mask_fn=mask_fn, mask_score="taylor_g1",
        capture_gradients=True,
    )
    assert any(not torch.equal(a, b) for a, b in zip(first.g1_snapshots, second.g1_snapshots))


@pytest.mark.parametrize("score_type", ["magnitude", "taylor_g1", "taylor_g2"])
def test_mask_score_uses_preupdate_center_weight(score_type):
    model, _manager, mask_fn = build_sparse(seed=13, score=score_type)
    result = sam_step(
        model, optimizer_for(model, lr=0.5), nn.CrossEntropyLoss(),
        torch.randn(8, 8), torch.randint(0, 4, (8,)), torch.device("cpu"),
        param_mask_fn=mask_fn, mask_score=score_type, capture_gradients=True,
    )
    for index, score in enumerate(result.mask_scores):
        weight = result.center_weight_snapshots[index]
        if score_type == "magnitude":
            expected = weight.abs()
        elif score_type == "taylor_g1":
            expected = (weight * result.g1_snapshots[index]).abs()
        else:
            expected = (weight * result.g2_snapshots[index]).abs()
        assert torch.equal(score, expected)
    assert any(
        not torch.equal(before, wrapper.weight_master.detach())
        for before, wrapper in zip(result.center_weight_snapshots, model._nm_wrappers)
    )


def test_all_sparse_methods_share_identical_magnitude_initial_mask():
    masks = []
    for score in ("magnitude", "taylor_g1", "taylor_g2"):
        model, _manager, _mask_fn = build_sparse(seed=21, score=score)
        masks.append([wrapper.mask.clone() for wrapper in model._nm_wrappers])
    assert all(
        torch.equal(reference, candidate)
        for reference, candidate in zip(masks[0], masks[1])
    )
    assert all(
        torch.equal(reference, candidate)
        for reference, candidate in zip(masks[0], masks[2])
    )


def test_masked_master_positions_have_score_gradients_and_can_regrow():
    model, manager, _mask_fn = build_sparse(seed=5, score="taylor_g1")
    nn.CrossEntropyLoss()(model(torch.randn(8, 8)), torch.randint(0, 4, (8,))).backward()
    scores = []
    old_masks = [wrapper.mask.clone() for wrapper in manager.wrappers]
    for wrapper in manager.wrappers:
        masked_grad = wrapper.weight_master.grad[wrapper.mask == 0]
        assert masked_grad.numel() and torch.any(masked_grad != 0)
        importance = torch.zeros_like(wrapper.weight_master)
        importance[wrapper.mask == 0] = 2.0
        importance[wrapper.mask == 1] = 1.0
        scores.append(importance)
    manager.recompute_from_scores(scores)
    assert all(
        torch.any((old == 0) & (wrapper.mask == 1))
        for old, wrapper in zip(old_masks, manager.wrappers)
    )


def test_new_mask_is_installed_after_sam_and_used_on_next_step():
    model, manager, mask_fn = build_sparse(seed=8, score="taylor_g2")
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_for(model)
    x = torch.randn(8, 8)
    y = torch.randint(0, 4, (8,))
    before = [wrapper.mask.clone() for wrapper in manager.wrappers]
    result = sam_step(
        model, optimizer, criterion, x, y, torch.device("cpu"),
        param_mask_fn=mask_fn, mask_score="taylor_g2",
    )
    assert all(torch.equal(a, b) for a, b in zip(before, result.pre_masks))
    assert all(torch.equal(a, b) for a, b in zip(before, result.post_masks))
    manager.recompute_from_scores(result.mask_scores)
    installed = [wrapper.mask.clone() for wrapper in manager.wrappers]
    observed = []
    sam_step(
        model, optimizer, criterion, x, y, torch.device("cpu"),
        param_mask_fn=mask_fn,
        input_observer=lambda _name, _tensor: observed.append(
            [wrapper.mask.clone() for wrapper in manager.wrappers]
        ),
    )
    assert all(torch.equal(a, b) for a, b in zip(installed, observed[0]))
    assert all(torch.equal(a, b) for a, b in zip(observed[0], observed[1]))
