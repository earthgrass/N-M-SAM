"""Every complete block in the sparse-SGD path must remain exactly 2:4."""
import torch
import torch.nn as nn

from nm_sam_cifar10.metrics.sanity import check_block_sums
from nm_sam_cifar10.sparsity import SparsityConfig
from nm_sam_cifar10.sparsity.mask_manager import MaskManager
from nm_sam_cifar10.strong_sparse_sgd import sparse_sgd_step


def test_all_complete_blocks_sum_to_two_after_magnitude_update():
    model = nn.Sequential(
        nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 4, bias=False),
    )
    manager = MaskManager(model, SparsityConfig(n=2, m=4), scope="all_supported")
    assert check_block_sums(model, n=2, m=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    result = sparse_sgd_step(
        model, optimizer, nn.CrossEntropyLoss(),
        torch.randn(6, 4, 4, 4), torch.randint(0, 4, (6,)),
        torch.device("cpu"), manager, step_number=20, mask_update_interval=20,
    )
    assert result.mask_updated
    assert result.mask_score_reference == "pre_update"
    assert check_block_sums(model, n=2, m=4)
    for wrapper in model._nm_wrappers:
        flat = wrapper.mask.reshape(wrapper.mask.shape[0], -1)
        complete = (flat.shape[1] // 4) * 4
        sums = flat[:, :complete].reshape(-1, 4).sum(dim=1)
        assert torch.all(sums == 2)
