"""Prove that magnitude scores come from W_t, not W_(t+1)."""
import hashlib

import torch
import torch.nn as nn

from nm_sam_cifar10.sparsity import SparsityConfig
from nm_sam_cifar10.sparsity.mask_manager import MaskManager
from nm_sam_cifar10.strong_sparse_sgd import sparse_sgd_step


def _hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def test_score_uses_pre_update_weight_snapshot():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    manager = MaskManager(model, SparsityConfig(n=2, m=4), scope="all_supported")
    wrapper = manager.wrappers[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.4, momentum=0.0)
    old_weight = wrapper.weight_master.detach().clone()
    captured_scores = []
    original = manager.recompute_from_scores

    def capture(scores):
        captured_scores.extend(score.detach().clone() for score in scores)
        return original(scores)

    manager.recompute_from_scores = capture
    result = sparse_sgd_step(
        model, optimizer, nn.CrossEntropyLoss(),
        torch.randn(16, 8), torch.randint(0, 4, (16,)), torch.device("cpu"),
        manager, step_number=20, mask_update_interval=20,
    )
    new_weight = wrapper.weight_master.detach().clone()
    score = captured_scores[0]

    assert result.mask_score_reference == "pre_update"
    assert not torch.equal(old_weight, new_weight)
    assert torch.equal(score, old_weight.abs())
    assert not torch.equal(score, new_weight.abs())

    # Hash magnitudes because magnitude is the protocol's score domain.
    old_weight_hash = _hash(old_weight.abs())
    score_hash = _hash(score)
    new_weight_hash = _hash(new_weight.abs())
    print(f"old_weight_hash={old_weight_hash}")
    print(f"score_hash={score_hash}")
    print(f"new_weight_hash={new_weight_hash}")
    assert old_weight_hash == score_hash
    assert score_hash != new_weight_hash
