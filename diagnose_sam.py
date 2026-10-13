"""SAM numerical diagnosis driver.

Runs Dense-SAM on ResNet18 + (real or fake) CIFAR-10 for the first
``--num-steps`` optimization steps and records per-step diagnostics:

    step, center_loss, grad1_norm, epsilon_norm, perturbed_loss, grad2_norm,
    max_abs_weight, max_abs_gradient, max_abs_logit, has_nan, has_inf

After the run we print a summary, write ``sam_diagnosis.csv``, and exit.

This script does NOT touch any global project state; it is a focused
diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
from optim.sam import disable_running_stats, make_sam_optimizer  # noqa: E402
from train import build_transforms  # noqa: E402
from utils.seed import get_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18_cifar")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--data-root", default="./nm_sam_cifar10/data/cifar10")
    p.add_argument("--use-fake-data", action="store_true")
    p.add_argument("--fake-train-size", type=int, default=4000)
    p.add_argument("--subset-train", type=int, default=4000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output-dir", default="./nm_sam_cifar10/results/sam_diagnosis")
    return p.parse_args()


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device(prefer_cuda=True) if args.device == "auto" else torch.device(args.device)

    # Data
    train_tf, _ = build_transforms()
    train_set = build_cifar10(
        args.data_root,
        train=True,
        transform=train_tf,
        use_fake_data=args.use_fake_data,
        fake_train_size=args.fake_train_size,
        fake_test_size=500,
    )
    if not args.use_fake_data and args.subset_train:
        train_set = Subset(train_set, list(range(min(args.subset_train, len(train_set)))))
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # Model + optimizer
    model = build_model(args.model, num_classes=10).to(device)
    opt = make_sam_optimizer(
        model,
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        rho=args.rho,
        nesterov=False,
        adaptive=False,
    )
    crit = nn.CrossEntropyLoss()

    rows = []
    step = 0
    nan_seen = False
    inf_seen = False
    while step < args.num_steps:
        for x, y in loader:
            if step >= args.num_steps:
                break
            x = x.to(device)
            y = y.to(device)
            model.train()
            with disable_running_stats(model):
                logits = model(x)
                loss1 = crit(logits, y)
                opt.zero_grad(set_to_none=True)
                loss1.backward()
                g_norm = opt.first_step(zero_grad=True)
                # Probe diagnostics at step == log_interval boundaries.
                logits2 = model(x)
                loss2 = crit(logits2, y)
                loss2.backward()

            # grad2 norm (computed at w + epsilon)
            grad2_sq = torch.zeros(1, device=device)
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad2_sq = grad2_sq + p.grad.pow(2).sum()
            grad2_norm = float(grad2_sq.sqrt().item())

            # Compute perturbation global norm (cached in state[p]["e"]).
            eps_sq = torch.zeros(1, device=device)
            for group in opt.param_groups:
                for p in group["params"]:
                    if "e" not in opt.state[p]:
                        continue
                    eps_sq = eps_sq + opt.state[p]["e"].pow(2).sum()
            eps_norm = float(eps_sq.sqrt().item())

            max_abs_weight = max(p.detach().abs().max().item() for group in opt.param_groups for p in group["params"] if p.grad is not None)
            max_abs_grad = max(p.grad.detach().abs().max().item() for group in opt.param_groups for p in group["params"] if p.grad is not None)
            with torch.no_grad():
                max_abs_logit = float(logits2.detach().abs().max().item())
                has_nan = bool(torch.isnan(logits2).any().item()) or bool(torch.isnan(loss2))
                has_inf = bool(torch.isinf(logits2).any().item()) or bool(torch.isinf(loss2))

            opt.second_step(zero_grad=True)

            if has_nan:
                nan_seen = True
            if has_inf:
                inf_seen = True

            if step % args.log_interval == 0:
                print(
                    f"[step {step:03d}] L1={loss1.item():.4f} L2={loss2.item():.4f} "
                    f"||g1||={float(g_norm.item()):.4e} ||eps||={eps_norm:.4e} ||g2||={grad2_norm:.4e} "
                    f"max|w|={max_abs_weight:.3f} max|g|={max_abs_grad:.3e} max|logit|={max_abs_logit:.3f}"
                )
            rows.append({
                "step": step,
                "center_loss": float(loss1.item()),
                "perturbed_loss": float(loss2.item()),
                "grad1_norm": float(g_norm.item()),
                "epsilon_norm": eps_norm,
                "grad2_norm": grad2_norm,
                "max_abs_weight": max_abs_weight,
                "max_abs_gradient": max_abs_grad,
                "max_abs_logit": max_abs_logit,
                "has_nan": has_nan,
                "has_inf": has_inf,
            })
            step += 1

    out_csv = os.path.join(args.output_dir, "sam_diagnosis.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summary = {
        "rho": args.rho,
        "lr": args.lr,
        "num_steps": args.num_steps,
        "nan_seen": nan_seen,
        "inf_seen": inf_seen,
        "epsilon_norm_mean": sum(r["epsilon_norm"] for r in rows) / max(len(rows), 1),
        "epsilon_norm_max": max(r["epsilon_norm"] for r in rows),
        "epsilon_norm_min": min(r["epsilon_norm"] for r in rows),
        "grad1_norm_first10_mean": sum(r["grad1_norm"] for r in rows[:10]) / max(min(10, len(rows)), 1),
    }
    with open(os.path.join(args.output_dir, "sam_diagnosis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
