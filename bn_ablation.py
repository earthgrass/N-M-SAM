"""BN handling A/B test.

Compares Dense-SAM trained with the OLD (both passes under
disable_running_stats) vs the NEW (first forward updates, second forward
frozen) BN pattern on the same 4000-subset / 2-epoch configuration. The
goal is to prove that the stale-BN running stats were the cause of the
8e12 test_loss observed in Phase 1.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nm_sam_cifar10.models import resnet18_cifar
from nm_sam_cifar10.optim.sam import (
    disable_running_stats,
    freeze_bn_running_stats,
    make_sam_optimizer,
)


def train_one(method_bn: str, n_epochs=2, n_steps=20):
    """Run a short Dense-SAM training using the chosen BN pattern."""
    torch.manual_seed(0)
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))])
    ds = CIFAR10("./nm_sam_cifar10/data/cifar10", train=True, download=False, transform=tf)
    test = CIFAR10("./nm_sam_cifar10/data/cifar10", train=False, download=False, transform=tf)
    train = Subset(ds, list(range(4000)))
    eval_set = Subset(test, list(range(1000)))
    train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(eval_set, batch_size=128, shuffle=False, num_workers=0)

    model = resnet18_cifar(num_classes=10)
    opt = make_sam_optimizer(model, lr=0.01, momentum=0.9, weight_decay=5e-4, rho=0.05)
    crit = nn.CrossEntropyLoss()

    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    rm_before = [m.running_mean.clone() for m in bn_modules]
    rv_before = [m.running_var.clone() for m in bn_modules]

    step = 0
    train_loss = 0.0; train_correct = 0; train_total = 0
    train_log = []
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            if step >= n_steps:
                break
            if method_bn == "old":
                # OLD bug: both passes under disable_running_stats.
                with disable_running_stats(model):
                    logits = model(x); loss = crit(logits, y)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    opt.first_step(zero_grad=True)
                    logits2 = model(x); loss2 = crit(logits2, y); loss2.backward()
                opt.second_step(zero_grad=True)
            else:
                # NEW pattern: first forward updates BN, second frozen.
                logits = model(x); loss = crit(logits, y)
                opt.zero_grad(set_to_none=True); loss.backward()
                opt.first_step(zero_grad=True)
                with freeze_bn_running_stats(model):
                    logits2 = model(x); loss2 = crit(logits2, y); loss2.backward()
                opt.second_step(zero_grad=True)
            train_loss += loss.item(); train_correct += (logits2.argmax(1) == y).sum().item(); train_total += y.size(0)
            step += 1
            if step % 5 == 0:
                train_log.append({"step": step, "loss": float(loss.item())})
        if step >= n_steps:
            break

    train_loss /= max(step, 1)
    train_acc = train_correct / max(train_total, 1)

    # Evaluate in eval mode (uses BN running stats).
    model.eval()
    test_loss = 0.0; test_correct = 0; test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x); loss = crit(logits, y)
            test_loss += loss.item() * y.size(0); test_correct += (logits.argmax(1) == y).sum().item(); test_total += y.size(0)
    test_loss /= max(test_total, 1)
    test_acc = test_correct / max(test_total, 1)

    # BN running stats drift.
    rm_change = float(sum((a - b).abs().sum() for a, b in zip(rm_before, [m.running_mean for m in bn_modules])))
    rv_change = float(sum((a - b).abs().sum() for a, b in zip(rv_before, [m.running_var for m in bn_modules])))

    return {
        "bn_fix": method_bn,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "bn_running_mean_change": rm_change,
        "bn_running_var_change": rv_change,
        "n_steps": step,
    }


def main():
    rows = []
    for bn in ["old", "new"]:
        print(f"=== BN handling = {bn} ===")
        r = train_one(bn, n_epochs=2, n_steps=20)
        print(json.dumps(r, indent=2))
        rows.append(r)

    os.makedirs("./nm_sam_cifar10/results/phase1_fixed", exist_ok=True)
    out_csv = "./nm_sam_cifar10/results/phase1_fixed/bn_ablation.csv"
    with open(out_csv, "w", newline="") as f:
        cols = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
