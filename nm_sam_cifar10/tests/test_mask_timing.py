"""Magnitude masks update only after the boundary optimizer step."""
import torch
import torch.nn as nn

from nm_sam_cifar10.sparsity import SparsityConfig
from nm_sam_cifar10.sparsity.mask_manager import MaskManager
from nm_sam_cifar10.strong_sparse_sgd import sparse_sgd_step


def test_mask_update_happens_after_optimizer_step():
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    manager = MaskManager(model, SparsityConfig(n=2, m=4), scope="all_supported")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    events = []
    original_step = optimizer.step
    original_recompute = manager.recompute_from_scores

    def traced_step(*args, **kwargs):
        events.append("optimizer_step")
        return original_step(*args, **kwargs)

    def traced_recompute(scores):
        events.append("mask_recompute")
        return original_recompute(scores)

    optimizer.step = traced_step
    manager.recompute_from_scores = traced_recompute
    result = sparse_sgd_step(
        model, optimizer, nn.CrossEntropyLoss(),
        torch.randn(5, 8), torch.randint(0, 4, (5,)), torch.device("cpu"),
        manager, step_number=20, mask_update_interval=20,
    )
    assert result.mask_updated
    assert result.mask_score_reference == "pre_update"
    assert events == ["optimizer_step", "mask_recompute"]
    assert all(wrapper.generation == 1 for wrapper in model._nm_wrappers)
