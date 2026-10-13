"""Tests for the SAM-aware mask recomputation."""
from __future__ import annotations

import torch

from nm_sam_cifar10.sparsity import SparsityConfig
from nm_sam_cifar10.sparsity.mask import make_nm_mask


def test_magnitude_score_topn():
    """Manually verified: |W| top-N per block."""
    W = torch.tensor([[1.0, -5.0, 3.0, -4.0]])
    m = make_nm_mask(W, n=2, m=4, score="magnitude")
    # Top-2 by |W| are |-5|=5 (idx 1) and |-4|=4 (idx 3).
    assert int(m[0, 0].item()) == 0
    assert int(m[0, 1].item()) == 1
    assert int(m[0, 2].item()) == 0
    assert int(m[0, 3].item()) == 1


def test_taylor_score_topn():
    """Manually verified: |W * g| top-N per block."""
    W = torch.tensor([[1.0, -5.0, 3.0, -4.0]])
    g = torch.tensor([[2.0, 1.0, 0.5, 100.0]])  # |W*g| = [2, 5, 1.5, 400] -> top-2 at idx 1, 3.
    m = make_nm_mask(W, n=2, m=4, score="sam_taylor", grad=g)
    assert int(m[0, 0].item()) == 0
    assert int(m[0, 1].item()) == 1
    assert int(m[0, 2].item()) == 0
    assert int(m[0, 3].item()) == 1


def test_conv2d_layout_uniform():
    """Conv4D must go through the same canonical mask builder; blocks must be 2:4."""
    W = torch.randn(8, 4, 3, 3)  # inner = 4*3*3 = 36 -> 9 blocks per output
    m = make_nm_mask(W, n=2, m=4, score="magnitude")
    assert m.shape == W.shape
    for o in range(8):
        flat = m[o].reshape(-1)
        for i in range(0, 36, 4):
            assert int(flat[i : i + 4].sum().item()) == 2


def test_recompute_changes_when_grad_changes():
    W = torch.ones(2, 8)
    g1 = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1.0, 0, 0, 0, 0, 0, 0]])
    m1 = make_nm_mask(W, n=2, m=4, score="sam_taylor", grad=g1)
    g2 = torch.tensor([[0, 0, 0, 1.0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1.0, 0, 0, 0]])
    m2 = make_nm_mask(W, n=2, m=4, score="sam_taylor", grad=g2)
    # Different grads should produce different masks.
    assert not torch.equal(m1, m2)


if __name__ == "__main__":
    test_magnitude_score_topn()
    test_taylor_score_topn()
    test_conv2d_layout_uniform()
    test_recompute_changes_when_grad_changes()
    print("All SAM-aware mask tests passed.")
