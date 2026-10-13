"""The new baseline must be a one-forward, ordinary-SGD path."""
from dataclasses import fields
import inspect

import torch
import torch.nn as nn

from nm_sam_cifar10.optim.optimizer_factory import build_optimizer
from nm_sam_cifar10.sparsity import SparsityConfig
from nm_sam_cifar10.sparsity.mask_manager import MaskManager
from nm_sam_cifar10.strong_sparse_sgd import (
    StrongSparseSGDConfig,
    sparse_sgd_step,
)


class CountingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return self.linear(inputs)


def test_sparse_sgd_has_no_sam_step_or_adversarial_forward():
    model = CountingNet()
    manager = MaskManager(model, SparsityConfig(n=2, m=4), scope="all_supported")

    class Args:
        lr = 0.1
        momentum = 0.9
        weight_decay = 5e-4
        nesterov = False

    optimizer = build_optimizer(model, "sgd", Args)
    assert type(optimizer) is torch.optim.SGD
    assert not hasattr(optimizer, "first_step")
    assert not hasattr(optimizer, "second_step")
    assert "rho" not in {field.name for field in fields(StrongSparseSGDConfig)}
    source = inspect.getsource(sparse_sgd_step)
    assert "first_step" not in source
    assert "second_step" not in source

    result = sparse_sgd_step(
        model, optimizer, nn.CrossEntropyLoss(),
        torch.randn(6, 8), torch.randint(0, 4, (6,)), torch.device("cpu"),
        manager, step_number=1, mask_update_interval=20,
    )
    assert model.forward_calls == 1
    assert not result.mask_updated
    assert torch.isfinite(torch.tensor(result.train.loss))

