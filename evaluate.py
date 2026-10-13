"""Standalone evaluation script.

Loads a checkpoint (best.pt or last.pt) from a training run directory and
re-evaluates on the full CIFAR-10 test set (or a configurable subset).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nm_sam_cifar10.data.cifar10 import build_cifar10, build_transforms  # noqa: E402
from nm_sam_cifar10.methods import get_method_spec  # noqa: E402
from nm_sam_cifar10.metrics.sharpness import compute_sharpness  # noqa: E402
from nm_sam_cifar10.metrics.task_metrics import evaluate_task  # noqa: E402
from nm_sam_cifar10.models import build_model  # noqa: E402
from nm_sam_cifar10.sparsity import SparsityConfig  # noqa: E402
from nm_sam_cifar10.sparsity.mask_manager import MaskManager  # noqa: E402
from nm_sam_cifar10.sparsity.parametrization import make_param_mask_fn  # noqa: E402
from nm_sam_cifar10.utils.seed import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--data-root", default="./data/cifar10")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--sharpness-subset", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.run_dir) / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        cfg = payload.get("config", payload)

    device = get_device(prefer_cuda=True) if args.device == "auto" else torch.device(args.device)
    spec = get_method_spec(cfg["method"])

    # Build model + sparsity the same way train.py does.
    model = build_model(cfg["model"], num_classes=10).to(device)
    if spec.is_sparse:
        sparsity_cfg = SparsityConfig(n=cfg["n"], m=cfg["m"], enabled=True)
        MaskManager(model, sparsity_cfg, scope=cfg.get("sparse_scope", "phase1"))

    # Load checkpoint.
    ckpt_path = Path(args.run_dir) / args.ckpt
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=False)

    # Data.
    _, test_tf = build_transforms()
    test_set = build_cifar10(args.data_root, train=False, transform=test_tf)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    task = evaluate_task(model, test_loader, criterion, device)
    print(json.dumps({"test_loss": task.loss, "test_acc": task.accuracy, "ckpt": str(ckpt_path)}, indent=2))

    # Sharpness probe.
    train_set = build_cifar10(args.data_root, train=True, transform=test_tf)
    sub = Subset(train_set, list(range(min(args.sharpness_subset, len(train_set)))))
    loader = DataLoader(sub, batch_size=min(args.sharpness_subset, 256), shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    base_loss, adv_loss, sharp, info = compute_sharpness(
        model, x, y, criterion, args.rho, device,
        mask_fn=make_param_mask_fn(model) if spec.is_sparse else None,
    )
    print(json.dumps({
        "sharpness": sharp,
        "worst_case_loss": adv_loss,
        "center_loss": base_loss,
        "info": info,
    }, indent=2))


if __name__ == "__main__":
    main()
