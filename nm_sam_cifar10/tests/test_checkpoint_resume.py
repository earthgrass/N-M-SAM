"""A split sparse-SGD trajectory must match uninterrupted execution."""
from pathlib import Path

import torch
import torch.nn as nn

from nm_sam_cifar10.sparsity import SparsityConfig
from nm_sam_cifar10.sparsity.mask_manager import MaskManager
from nm_sam_cifar10.strong_sparse_sgd import (
    _nested_equal,
    load_checkpoint,
    save_checkpoint,
    sparse_sgd_step,
)


def _build():
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    manager = MaskManager(model, SparsityConfig(n=2, m=4), scope="all_supported")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    return model, manager, optimizer, scheduler


def _advance(state, batches, start, end):
    model, manager, optimizer, scheduler = state
    criterion = nn.CrossEntropyLoss()
    for index in range(start, end):
        inputs, targets = batches[index]
        sparse_sgd_step(
            model, optimizer, criterion, inputs, targets, torch.device("cpu"),
            manager, step_number=index + 1, mask_update_interval=2,
        )
        scheduler.step()


def test_checkpoint_resume_matches_uninterrupted(tmp_path: Path):
    generator = torch.Generator().manual_seed(91)
    batches = [
        (torch.randn(6, 8, generator=generator), torch.randint(0, 4, (6,), generator=generator))
        for _ in range(4)
    ]

    continuous = _build()
    _advance(continuous, batches, 0, 4)

    split = _build()
    _advance(split, batches, 0, 2)
    loader_generator = torch.Generator().manual_seed(123)
    checkpoint = tmp_path / "sparse_sgd.pt"
    split_model, split_manager, split_optimizer, split_scheduler = split
    save_checkpoint(
        checkpoint, split_model, split_optimizer, split_scheduler, split_manager,
        epoch=0, global_step=2,
        loader_generator=loader_generator, mask_update_interval=2,
    )

    resumed = _build()
    resumed_model, resumed_manager, resumed_optimizer, resumed_scheduler = resumed
    resumed_generator = torch.Generator()
    payload = load_checkpoint(
        checkpoint, resumed_model, resumed_optimizer, resumed_scheduler,
        resumed_manager, resumed_generator, torch.device("cpu")
    )
    assert payload["global_step"] == 2
    assert payload["next_mask_update_step"] == 4
    assert torch.equal(loader_generator.get_state(), resumed_generator.get_state())
    _advance(resumed, batches, 2, 4)

    for left, right in zip(continuous[0].state_dict().values(), resumed[0].state_dict().values()):
        assert torch.equal(left, right)
    assert _nested_equal(continuous[2].state_dict(), resumed[2].state_dict())
    assert _nested_equal(continuous[3].state_dict(), resumed[3].state_dict())
    assert continuous[1].state_dict() == resumed[1].state_dict()
