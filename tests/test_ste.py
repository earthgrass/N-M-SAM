"""Unit tests for STE parametrization: effective weight is masked, grad flows to master."""
import torch

from nm_sam_cifar10.sparsity import SparsityConfig, attach_nm_sparsity
from nm_sam_cifar10.sparsity.parametrization import NMSparsifiedWeight


def test_ste_effective_weight_is_masked():
    w = torch.randn(3, 8)
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    wrapper = NMSparsifiedWeight(w.clone(), cfg)
    eff = wrapper()
    assert torch.all(eff[wrapper.mask == 0] == 0.0)
    # active positions equal master weight.
    assert torch.allclose(eff[wrapper.mask == 1], wrapper.weight_master[wrapper.mask == 1])


def test_ste_gradient_flows_to_master():
    w = torch.randn(3, 8)
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    wrapper = NMSparsifiedWeight(w.clone(), cfg)
    eff = wrapper()
    eff.sum().backward()
    # Gradient flows to the dense master weight even at masked positions.
    assert wrapper.weight_master.grad is not None
    assert torch.all(wrapper.weight_master.grad != 0)


def test_recompute_mask_changes_positions():
    w = torch.randn(4, 16)
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    wrapper = NMSparsifiedWeight(w.clone(), cfg)
    old_mask = wrapper.mask.clone()
    # Perturb weight so the top-2 per block changes.
    with torch.no_grad():
        wrapper.weight_master.fill_(0.0)
    change = wrapper.recompute_mask(score="magnitude")
    assert change > 0  # mask should have changed
    # After recompute, block-sum is still 2.
    flat = wrapper.mask.reshape(-1, 4)
    sums = flat.sum(dim=-1)
    assert torch.all(sums == 2)


def test_attach_nm_sparsity_simple_cnn():
    import torch.nn as nn
    from nm_sam_cifar10.models import SimpleCNN
    model = SimpleCNN()
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    n, wrappers = attach_nm_sparsity(model, cfg)
    assert n > 0
    # First conv (conv1) is skipped, others should be wrapped.
    names = [w.name for w in wrappers]
    assert "conv1" not in names
    # block-sum sanity.
    for w in wrappers:
        if w.weight.dim() == 4:
            o, ic, kh, kw = w.weight.shape
            flat = w.mask.reshape(o, -1)
        else:
            flat = w.mask
        inner = flat.size(-1)
        num_full = inner // 4
        if num_full == 0:
            continue
        head = flat[..., : num_full * 4].reshape(-1, 4)
        assert torch.all(head.sum(dim=-1) == 2)


if __name__ == "__main__":
    test_ste_effective_weight_is_masked()
    test_ste_gradient_flows_to_master()
    test_recompute_mask_changes_positions()
    test_attach_nm_sparsity_simple_cnn()
    print("All STE tests passed.")
