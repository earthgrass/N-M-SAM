"""ResNet18-CIFAR and configurable sparse-scope regression tests."""
from __future__ import annotations

import torch
import torch.nn as nn

from nm_sam_cifar10.models import resnet18_cifar
from nm_sam_cifar10.sparsity.mask_manager import SPARSE_SCOPES, plan_sparse_layers


def test_resnet18_is_cifar_variant():
    model = resnet18_cifar(num_classes=10)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1)
    assert not any(isinstance(module, nn.MaxPool2d) for module in model.modules())
    assert model.linear.out_features == 10
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_all_required_sparse_scopes_are_supported():
    model = resnet18_cifar()
    for scope in SPARSE_SCOPES:
        decisions = plan_sparse_layers(model, scope, 2, 4)
        assert decisions
    phase1 = {decision.name: decision.sparse for decision in plan_sparse_layers(model, "phase1")}
    assert phase1["conv1"] is False
    assert phase1["linear"] is True
    assert phase1["layer2.0.shortcut.0"] is False


def test_all_supported_includes_first_last_and_downsample():
    model = resnet18_cifar()
    selected = {decision.name for decision in plan_sparse_layers(model, "all_supported") if decision.sparse}
    assert "conv1" in selected
    assert "linear" in selected
    assert "layer2.0.shortcut.0" in selected
