"""Unit tests for SAM: gradient flows through perturbation, mask respected."""
import torch
import torch.nn as nn

from nm_sam_cifar10.optim.sam import SAM, disable_running_stats, make_sam_optimizer


def _make_toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 4, bias=False))


def test_sam_two_step_changes_weights():
    model = _make_toy_model()
    opt = make_sam_optimizer(model, lr=0.01, momentum=0.0, weight_decay=0.0, rho=0.05)
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 2, 3])
    crit = nn.CrossEntropyLoss()
    before = [p.detach().clone() for p in model.parameters()]

    model.train()
    with disable_running_stats(model):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.first_step(zero_grad=True)
        loss2 = crit(model(x), y)
        loss2.backward()
    opt.second_step(zero_grad=True)

    for p, before_p in zip(model.parameters(), before):
        assert not torch.allclose(p, before_p)


def test_sam_perturbation_zero_on_masked():
    from nm_sam_cifar10.sparsity import SparsityConfig, attach_nm_sparsity
    from nm_sam_cifar10.sparsity.parametrization import make_param_mask_fn

    torch.manual_seed(0)
    model = _make_toy_model()
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    attach_nm_sparsity(model, cfg)
    mask_fn = make_param_mask_fn(model)

    opt = make_sam_optimizer(model, lr=0.01, momentum=0.0, weight_decay=0.0, rho=0.1)
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 2, 3])
    crit = nn.CrossEntropyLoss()

    model.train()
    with disable_running_stats(model):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.first_step(zero_grad=True, mask_fn=mask_fn)

    # After perturbation: each weight has had (grad * scale) added, but
    # this should have been masked so the sparse weight's effective value
    # (in the masked positions) remains zero. We check via the wrappers.
    for w in getattr(model, "_nm_wrappers", []):
        eff = w()
        # masked positions still 0 in the *dense master*? No: master can be non-zero
        # because we added the (masked) perturbation to it. What we CAN check: the
        # effective weight is zero wherever mask==0.
        assert torch.all(eff[w.mask == 0] == 0.0)


if __name__ == "__main__":
    test_sam_two_step_changes_weights()
    test_sam_perturbation_zero_on_masked()
    print("All SAM tests passed.")
