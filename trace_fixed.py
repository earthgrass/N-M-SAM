"""Single-batch numerical trace for Dense-SAM, 2:4-SAM, 2:4-SAM-aware.

Fixed input: one FakeData batch. Records:

* L_center, L_adv
* ||g1||, ||epsilon||, ||g2||, |epsilon|/|rho|
* restore_error (after sub_epsilon), update_norm
* max_abs_logit
* NaN / Inf detection
* sparsity-specific: num_sparse_params, active_ratio
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.cifar10 import build_cifar10  # noqa: E402
from metrics.sharpness import compute_sharpness  # noqa: E402
from models import build_model  # noqa: E402
from optim.sam import freeze_bn_running_stats, make_sam_optimizer  # noqa: E402
from sparsity import SparsityConfig, attach_nm_sparsity  # noqa: E402
from sparsity.parametrization import make_param_mask_fn  # noqa: E402
from train import build_transforms  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def trace_dense_sam():
    set_seed(0)
    train_tf, _ = build_transforms()
    train_set = build_cifar10("./nm_sam_cifar10/data/cifar10", train=True,
                              transform=train_tf, use_fake_data=True,
                              fake_train_size=512)
    loader = DataLoader(Subset(train_set, list(range(128))), batch_size=128, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    model = build_model("resnet18_cifar", num_classes=10)
    crit = nn.CrossEntropyLoss()
    opt = make_sam_optimizer(model, lr=0.01, momentum=0.9, weight_decay=5e-4, rho=0.05)

    w0 = [p.detach().clone() for p in model.parameters()]
    model.train()
    # First forward (with BN running stats update).
    logits = model(x); loss1 = crit(logits, y); opt.zero_grad(set_to_none=True); loss1.backward()
    info = opt.first_step(zero_grad=True)
    # Second forward under freeze_bn_running_stats.
    with freeze_bn_running_stats(model):
        logits2 = model(x); loss2 = crit(logits2, y); loss2.backward()
    g2_sq = sum(p.grad.pow(2).sum() for g in opt.param_groups for p in g["params"] if p.grad is not None)
    g2_norm = float(g2_sq.sqrt().item())

    w_before_step = [p.detach().clone() for p in model.parameters()]
    opt.second_step(zero_grad=True)
    w_after_step = [p.detach().clone() for p in model.parameters()]
    restore_err = max((a - b).abs().max().item() for a, b in zip(w_before_step, w_after_step))
    update_norm = float(sum((a - b).pow(2).sum() for a, b in zip(w_after_step, w0)).sqrt().item())

    return {
        "L_center": float(loss1.item()),
        "L_adv": float(loss2.item()),
        "||g1||": info["g_norm"],
        "||epsilon||": info["eps_norm"],
        "epsilon_over_rho": info["eps_norm_over_rho"],
        "||g2||": g2_norm,
        "max_abs_logit_center": float(logits.abs().max().item()),
        "max_abs_logit_adv": float(logits2.abs().max().item()),
        "restore_error": restore_err,
        "update_norm": update_norm,
        "nan": bool(torch.isnan(loss1).item() or torch.isnan(loss2).item()),
        "inf": bool(torch.isinf(loss1).item() or torch.isinf(loss2).item()),
        "active_ratio": 1.0,
        "num_sparse_params": 0,
    }


def trace_sparse(method="nm_sam"):
    set_seed(0)
    train_tf, _ = build_transforms()
    train_set = build_cifar10("./nm_sam_cifar10/data/cifar10", train=True,
                              transform=train_tf, use_fake_data=True,
                              fake_train_size=512)
    loader = DataLoader(Subset(train_set, list(range(128))), batch_size=128, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    model = build_model("resnet18_cifar", num_classes=10)
    cfg = SparsityConfig(n=2, m=4, enabled=True)
    n_wrap, wrappers = attach_nm_sparsity(model, cfg)
    mask_fn = make_param_mask_fn(model)
    crit = nn.CrossEntropyLoss()
    opt = make_sam_optimizer(model, lr=0.01, momentum=0.9, weight_decay=5e-4, rho=0.05)

    score = "magnitude" if method == "nm_sam" else "sam_taylor"
    last_grad = None

    w0 = [p.detach().clone() for p in model.parameters()]
    model.train()
    logits = model(x); loss1 = crit(logits, y); opt.zero_grad(set_to_none=True); loss1.backward()
    info = opt.first_step(zero_grad=True, mask_fn=mask_fn)
    with freeze_bn_running_stats(model):
        logits2 = model(x); loss2 = crit(logits2, y); loss2.backward()
    # Capture g2 if SAM-aware.
    if method == "nm_sam_aware":
        for w in wrappers:
            grad = w.weight_master.grad
            if grad is not None:
                w.last_grad = grad.detach().clone()
    g2_sq = sum(p.grad.pow(2).sum() for g in opt.param_groups for p in g["params"] if p.grad is not None)
    g2_norm = float(g2_sq.sqrt().item())

    w_before_step = [p.detach().clone() for p in model.parameters()]
    opt.second_step(zero_grad=True)
    w_after_step = [p.detach().clone() for p in model.parameters()]
    restore_err = max((a - b).abs().max().item() for a, b in zip(w_before_step, w_after_step))
    update_norm = float(sum((a - b).pow(2).sum() for a, b in zip(w_after_step, w0)).sqrt().item())

    num_sparse = sum(w.mask.numel() for w in wrappers)
    active = sum(w.mask.float().sum().item() for w in wrappers)

    return {
        "L_center": float(loss1.item()),
        "L_adv": float(loss2.item()),
        "||g1||": info["g_norm"],
        "||epsilon||": info["eps_norm"],
        "epsilon_over_rho": info["eps_norm_over_rho"],
        "||g2||": g2_norm,
        "max_abs_logit_center": float(logits.abs().max().item()),
        "max_abs_logit_adv": float(logits2.abs().max().item()),
        "restore_error": restore_err,
        "update_norm": update_norm,
        "nan": bool(torch.isnan(loss1).item() or torch.isnan(loss2).item()),
        "inf": bool(torch.isinf(loss1).item() or torch.isinf(loss2).item()),
        "active_ratio": active / max(num_sparse, 1),
        "num_sparse_params": num_sparse,
    }


def main():
    out = {}
    print("=== Dense-SAM trace ===")
    out["dense_sam"] = trace_dense_sam()
    print(json.dumps(out["dense_sam"], indent=2))
    print("=== 2:4-SAM trace ===")
    out["nm_sam"] = trace_sparse("nm_sam")
    print(json.dumps(out["nm_sam"], indent=2))
    print("=== 2:4-SAM-aware trace ===")
    out["nm_sam_aware"] = trace_sparse("nm_sam_aware")
    print(json.dumps(out["nm_sam_aware"], indent=2))

    os.makedirs("./nm_sam_cifar10/results/phase1_fixed", exist_ok=True)
    with open("./nm_sam_cifar10/results/phase1_fixed/trace_sam_fixed.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
