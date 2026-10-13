"""SAM stability grid driver.

Runs Dense-SAM on the 4000-subset / 2-epoch configuration for a 6-cell
grid of (lr, rho) values. The 'stable' flag is determined strictly from
numerical rules:

    * no NaN / Inf in loss or logits
    * loss is finite
    * epsilon_norm / rho is within 1e-3 of 1.0
    * no exploding logits (max_abs_logit < 1e4)
    * optimization trajectory decreased loss at least once

No accuracy-based gating.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
os.chdir(PROJECT_ROOT)

GRID = [(lr, rho) for lr in (0.1, 0.05, 0.01) for rho in (0.05, 0.02)]


def main():
    out_root = "./nm_sam_cifar10/results/phase1_fixed/stability_grid"
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []
    for lr, rho in GRID:
        run_dir = os.path.join(out_root, f"dense_sam_lr{lr}_rho{rho}")
        os.makedirs(run_dir, exist_ok=True)
        cmd = [
            "python", "-m", "nm_sam_cifar10.train",
            "--method", "dense_sam",
            "--model", "resnet18_cifar",
            "--epochs", "2",
            "--batch-size", "128",
            "--lr", str(lr),
            "--rho", str(rho),
            "--subset-train", "4000",
            "--subset-test", "1000",
            "--num-workers", "0",
            "--sharpness-interval", "1",
            "--data-root", "./nm_sam_cifar10/data/cifar10",
            "--output-dir", run_dir,
        ]
        print(f"=== lr={lr} rho={rho} ===")
        t0 = time.time()
        r = subprocess.run(cmd, env={"PYTHONPATH": ".", **os.environ})
        elapsed = time.time() - t0
        # Parse metrics.
        try:
            import pandas as pd
            df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
        except Exception as exc:
            print(f"  failed to read metrics: {exc}")
            continue
        last = df.iloc[-1]
        # Numerical rules for "stable".
        nan = bool(last.get("nan_detected", False))
        inf = bool(last.get("inf_detected", False))
        try:
            eps_ratio = float(last.get("epsilon_norm_over_rho", 0.0))
        except (TypeError, ValueError):
            eps_ratio = 0.0
        try:
            max_logit = float(last.get("max_abs_logit", 0.0))
        except (TypeError, ValueError):
            max_logit = 0.0
        try:
            test_loss = float(last["test_loss"])
        except (KeyError, TypeError, ValueError):
            test_loss = float("nan")
        try:
            train_loss_first = float(df.iloc[0]["train_loss"])
            train_loss_last = float(last["train_loss"])
        except (KeyError, TypeError, ValueError):
            train_loss_first = train_loss_last = float("nan")
        finite_losses = math.isfinite(test_loss) and math.isfinite(train_loss_last)
        ok_eps = abs(eps_ratio - 1.0) < 1e-3 if eps_ratio > 0 else False
        ok_logits = max_logit < 1e4
        loss_decreased = train_loss_last < train_loss_first
        stable = (not nan) and (not inf) and finite_losses and ok_eps and ok_logits and loss_decreased

        summary_rows.append({
            "lr": lr,
            "rho": rho,
            "train_loss_first": train_loss_first,
            "train_loss_last": train_loss_last,
            "test_loss": test_loss,
            "test_acc": float(last["test_acc"]) if "test_acc" in last else float("nan"),
            "sharpness": float(last["sharpness"]) if last.get("sharpness") == last.get("sharpness") else float("nan"),
            "epsilon_norm": float(last.get("epsilon_norm", float("nan"))),
            "epsilon_ratio": eps_ratio,
            "grad1_norm": float(last.get("grad1_norm", float("nan"))),
            "grad2_norm": float(last.get("grad2_norm", float("nan"))),
            "max_abs_logit": max_logit,
            "nan": nan,
            "inf": inf,
            "finite_losses": finite_losses,
            "ok_eps_ratio": ok_eps,
            "ok_logits": ok_logits,
            "loss_decreased": loss_decreased,
            "stable": stable,
            "epoch_time": float(last.get("epoch_time", 0.0)),
            "elapsed_total": elapsed,
        })

    csv_path = "./nm_sam_cifar10/results/phase1_fixed/sam_stability_grid_fixed.csv"
    with open(csv_path, "w", newline="") as f:
        cols = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)
    print()
    print("Stability grid:")
    for r in summary_rows:
        print(f"  lr={r['lr']} rho={r['rho']}  stable={r['stable']}  "
              f"train={r['train_loss_last']:.3f}  test={r['test_loss']:.3f}  "
              f"sharp={r['sharpness']:.3e}  eps/rho={r['epsilon_ratio']:.6f}")
    print(f"\nWritten: {csv_path}")


if __name__ == "__main__":
    main()
