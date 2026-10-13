"""Audit BN running stats behavior during SAM training.

Verifies whether BN running_mean / running_var are actually being updated.
"""
import torch
import torch.nn as nn

from nm_sam_cifar10.models import resnet18_cifar
from nm_sam_cifar10.optim.sam import disable_running_stats, make_sam_optimizer


def main():
    torch.manual_seed(0)
    model = resnet18_cifar(num_classes=10)
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    print(f"# BN modules: {len(bn_modules)}")

    snap_before = [(m.running_mean.clone(), m.running_var.clone(), int(m.num_batches_tracked.item())) for m in bn_modules]
    print(f"BN[0] running_mean[:4] before: {snap_before[0][0][:4].tolist()}")
    print(f"BN[0] running_var[:4]  before: {snap_before[0][1][:4].tolist()}")
    print(f"BN[0] num_batches_tracked before: {snap_before[0][2]}")

    opt = make_sam_optimizer(model, lr=0.1, momentum=0.9, weight_decay=5e-4, rho=0.05)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))

    # Run 5 SAM steps under the CURRENT train.sam_step pattern (both passes disable_running_stats)
    for step in range(5):
        model.train()
        with disable_running_stats(model):
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.first_step(zero_grad=True)
            logits2 = model(x)
            loss2 = crit(logits2, y)
            loss2.backward()
        opt.second_step(zero_grad=True)

    snap_after = [(m.running_mean.clone(), m.running_var.clone(), int(m.num_batches_tracked.item())) for m in bn_modules]
    print()
    print(f"BN[0] running_mean[:4] after:  {snap_after[0][0][:4].tolist()}")
    print(f"BN[0] running_var[:4]  after:  {snap_after[0][1][:4].tolist()}")
    print(f"BN[0] num_batches_tracked after:  {snap_after[0][2]}")

    changed = sum(1 for a, b in zip(snap_before, snap_after) if not torch.allclose(a[0], b[0]) or not torch.allclose(a[1], b[1]))
    print(f"# BN modules with changed running stats after 5 SAM steps: {changed}/{len(bn_modules)}")

    print()
    print("INTERPRETATION:")
    if changed == 0:
        print("  -> BN running stats are FROZEN. evaluate(model.eval()) will use stale stats.")
        print("     This is a P1 bug: causes meaningless test loss + sharpness values.")
    else:
        print("  -> BN running stats updated.")


if __name__ == "__main__":
    main()
