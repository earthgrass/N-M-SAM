"""Unit tests for the N:M mask construction."""
import torch

from nm_sam_cifar10.sparsity.mask import blockwise_topk_mask, linear_mask, conv2d_mask


def test_linear_mask_per_block():
    w = torch.randn(8, 16)
    m = linear_mask(w, n=2, m=4)
    assert m.shape == w.shape
    # Per block of 4 in the inner dim, exactly 2 ones.
    for r in range(8):
        for c in range(0, 16, 4):
            block = m[r, c : c + 4]
            assert int(block.sum().item()) == 2


def test_linear_mask_tail_dense():
    w = torch.randn(4, 17)  # 17 not divisible by 4 -> tail = 1
    m = linear_mask(w, n=2, m=4)
    assert m.shape == w.shape
    assert int(m[0, 16].item()) == 1
    # First 16 columns: per block of 4, exactly 2 ones.
    for r in range(4):
        for c in range(0, 16, 4):
            assert int(m[r, c : c + 4].sum().item()) == 2


def test_conv2d_mask_per_block():
    w = torch.randn(4, 8, 3, 3)
    m = conv2d_mask(w, n=2, m=4)
    assert m.shape == w.shape
    for o in range(4):
        flat = m[o].reshape(-1)  # 72 = 18 blocks of 4
        for i in range(0, 72, 4):
            assert int(flat[i : i + 4].sum().item()) == 2


def test_blockwise_topk_picks_largest():
    w = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    m = blockwise_topk_mask(w, n=2, m=4)
    # Top-2 by |w|: positions 1 (|-2|=2), 2 (3), 3 (|-4|=4) -> indices 2 and 3.
    assert int(m[0, 0].item()) == 0
    assert int(m[0, 1].item()) == 0
    assert int(m[0, 2].item()) == 1
    assert int(m[0, 3].item()) == 1


if __name__ == "__main__":
    test_linear_mask_per_block()
    test_linear_mask_tail_dense()
    test_conv2d_mask_per_block()
    test_blockwise_topk_picks_largest()
    print("All mask tests passed.")
