"""Both SAM forwards consume the identical already-augmented tensor."""
from __future__ import annotations

import torch
import torch.nn as nn

from nm_sam_cifar10.optim.sam import make_sam_optimizer
from nm_sam_cifar10.trainers.sam_step import sam_step


def test_sam_reuses_same_augmented_batch_tensor():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    optimizer = make_sam_optimizer(model, lr=0.01, momentum=0.0, weight_decay=0.0, rho=0.05)
    inputs = torch.randn(6, 8)
    targets = torch.randint(0, 4, (6,))
    observed = []

    def observer(pass_name, tensor):
        observed.append((pass_name, tensor.data_ptr(), tensor.detach().clone()))

    result = sam_step(
        model, optimizer, nn.CrossEntropyLoss(), inputs, targets,
        torch.device("cpu"), input_observer=observer,
    )
    assert [item[0] for item in observed] == ["first", "second"]
    assert observed[0][1] == observed[1][1]
    assert torch.equal(observed[0][2], observed[1][2])
    assert result.first_input_data_ptr == result.second_input_data_ptr
