"""Frozen Strong SAM protocol and dynamic-mask timing regression tests."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nm_sam_cifar10.models import resnet18_cifar
from nm_sam_cifar10.optim.sam import make_sam_optimizer
from nm_sam_cifar10.sparsity import SparsityConfig
from nm_sam_cifar10.sparsity.mask_manager import MaskManager, plan_sparse_layers
from nm_sam_cifar10.sparsity.parametrization import make_param_mask_fn
from nm_sam_cifar10.strong_sam import load_config
from nm_sam_cifar10.trainers.sam_step import sam_step


PROJECT = Path(__file__).resolve().parents[1]


def test_formal_configs_match_frozen_protocol_field_by_field():
    dense = load_config(PROJECT / "configs" / "strong_dense_sam_seed0.yaml")
    sparse = load_config(PROJECT / "configs" / "strong_2to4_sam_magnitude_seed0.yaml")
    g1 = load_config(PROJECT / "configs" / "strong_2to4_sam_taylor_g1_seed0.yaml")
    g2 = load_config(PROJECT / "configs" / "strong_2to4_sam_taylor_g2_seed0.yaml")
    common = (
        "model", "seed", "epochs", "batch_size", "lr", "momentum",
        "weight_decay", "nesterov", "rho", "adaptive", "scheduler",
        "scheduler_t_max", "warmup_epochs", "precision",
    )
    for candidate in (sparse, g1, g2):
        for name in common:
            assert getattr(dense, name) == getattr(candidate, name), name
    sparse_common = ("n", "m", "sparse_scope", "mask_update_interval")
    for candidate in (g1, g2):
        for name in sparse_common:
            assert getattr(sparse, name) == getattr(candidate, name), name
    assert dense.epochs == 200
    assert dense.batch_size == 128
    assert dense.lr == 0.1 and dense.rho == 0.05
    assert dense.scheduler == "CosineAnnealingLR" and dense.scheduler_t_max == 200
    assert sparse.n == 2 and sparse.m == 4
    assert sparse.sparse_scope == "phase1"
    assert sparse.mask_score == "magnitude"
    assert sparse.mask_update_interval == 20
    assert g1.mask_score == "taylor_g1"
    assert g2.mask_score == "taylor_g2"


def test_phase1_scope_is_16_residual_convs_plus_final_linear():
    decisions = plan_sparse_layers(resnet18_cifar(), "phase1", 2, 4)
    selected = [item for item in decisions if item.sparse]
    selected_convs = [item for item in selected if item.layer_type == "Conv2d"]
    selected_linear = [item for item in selected if item.layer_type == "Linear"]
    assert len(selected_convs) == 16
    assert len(selected_linear) == 1 and selected_linear[0].name == "linear"
    assert not next(item for item in decisions if item.name == "conv1").sparse
    assert all("shortcut.0" not in item.name for item in selected)


class _TinySparseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)

    def forward(self, inputs):
        return self.fc2(torch.relu(self.fc1(inputs)))


def test_step20_mask_update_occurs_only_after_complete_sam_step():
    torch.manual_seed(0)
    model = _TinySparseNet()
    manager = MaskManager(model, SparsityConfig(n=2, m=4, score="magnitude"), scope="phase1")
    mask_fn = make_param_mask_fn(model)
    optimizer = make_sam_optimizer(model, lr=0.01, momentum=0.9, weight_decay=5e-4, rho=0.05)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(8, 8)
    y = torch.randint(0, 4, (8,))
    old_at_19 = new_after_20 = None

    for step in range(1, 22):
        before = [wrapper.mask.clone() for wrapper in manager.wrappers]
        observed = []

        def observer(_name, _tensor):
            observed.append([wrapper.mask.clone() for wrapper in manager.wrappers])

        result = sam_step(
            model, optimizer, criterion, x, y, torch.device("cpu"),
            param_mask_fn=mask_fn, mask_score="magnitude", input_observer=observer,
        )
        assert all(torch.equal(a, b) for a, b in zip(observed[0], observed[1]))
        assert all(torch.equal(a, b) for a, b in zip(before, result.post_masks))
        if step == 19:
            old_at_19 = before
        if step == 20:
            assert all(torch.equal(a, b) for a, b in zip(old_at_19, before))
            # Force a deterministic ranking change only after the complete SAM
            # update, then exercise the production recompute call.
            with torch.no_grad():
                for wrapper in manager.wrappers:
                    flat = wrapper.weight_master.reshape(wrapper.weight_master.shape[0], -1)
                    flat[:, 0::4] = 100.0
                    flat[:, 1::4] = 90.0
                    flat[:, 2::4] = 0.0
                    flat[:, 3::4] = 0.0
            change = manager.recompute("magnitude")
            assert change > 0.0
            new_after_20 = [wrapper.mask.clone() for wrapper in manager.wrappers]
        if step == 21:
            assert all(torch.equal(a, b) for a, b in zip(new_after_20, before))


def test_augmentation_is_sampled_once_then_same_tensor_is_reused():
    calls = 0

    def augment(raw):
        nonlocal calls
        calls += 1
        return raw + torch.rand_like(raw)

    model = nn.Linear(8, 4)
    optimizer = make_sam_optimizer(model, lr=0.1, momentum=0.9, weight_decay=5e-4, rho=0.05)
    raw = torch.zeros(6, 8)
    augmented = augment(raw)
    observed = []
    sam_step(
        model, optimizer, nn.CrossEntropyLoss(), augmented, torch.randint(0, 4, (6,)),
        torch.device("cpu"), input_observer=lambda _name, tensor: observed.append(tensor.clone()),
    )
    assert calls == 1
    assert torch.equal(observed[0], observed[1])


def test_mask_manager_state_round_trip_preserves_generation():
    model = _TinySparseNet()
    manager = MaskManager(model, SparsityConfig(n=2, m=4, score="magnitude"), scope="phase1")
    manager.recompute("magnitude")
    state = manager.state_dict()
    restored_model = _TinySparseNet()
    restored = MaskManager(restored_model, SparsityConfig(n=2, m=4, score="magnitude"), scope="phase1")
    restored.load_state_dict(state)
    assert restored.state_dict() == state
