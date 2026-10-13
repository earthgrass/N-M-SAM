"""Plot per-method training curves from ``results/`` and write a summary CSV."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


METHOD_COLORS = {
    "dense_sgd": "tab:blue",
    "dense_sam": "tab:orange",
    "nm_sgd": "tab:green",
    "nm_sam": "tab:red",
    "nm_sam_aware": "tab:purple",
}

METHOD_LABELS = {
    "dense_sgd": "Dense + SGD",
    "dense_sam": "Dense + SAM",
    "nm_sgd": "2:4 + SGD (mag)",
    "nm_sam": "2:4 + SAM (mag)",
    "nm_sam_aware": "2:4 + SAM (SAM-aware)",
}


def collect_runs(results_dir: str) -> Dict[str, List[Dict]]:
    """Find every ``metrics.csv`` whose parent directory name encodes a method+seed."""
    out: Dict[str, List[Dict]] = {}
    for csv_path in sorted(glob.glob(os.path.join(results_dir, "*", "metrics.csv"))):
        run = os.path.basename(os.path.dirname(csv_path))
        # parse run dir like "dense_sgd_seed0"
        method = run.rsplit("_seed", 1)[0]
        out.setdefault(method, []).append({"run": run, "path": csv_path})
    return out


def load_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def avg_over_seeds(runs: List[Dict]) -> pd.DataFrame:
    dfs = [load_metrics(r["path"]) for r in runs]
    if not dfs:
        return pd.DataFrame()
    # Align by epoch.
    base = dfs[0]
    n = min(len(d) for d in dfs)
    base = base.iloc[:n].reset_index(drop=True)
    for i, d in enumerate(dfs):
        if len(d) >= n:
            continue
        dfs[i] = d
    columns = [
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "epoch_time",
        "sharpness",
        "worst_case_loss",
    ]
    out = base.copy()
    for c in columns:
        if c not in base.columns:
            continue
        vals = [d[c].to_numpy()[:n] for d in dfs]
        import numpy as np
        arr = np.stack(vals, axis=0)
        out[c + "_mean"] = arr.mean(axis=0)
        out[c + "_std"] = arr.std(axis=0)
    return out


def plot_metric(
    runs: Dict[str, List[Dict]],
    column: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(7, 4.5))
    for method, lst in runs.items():
        agg = avg_over_seeds(lst)
        if column not in agg.columns:
            continue
        mean_col = column + "_mean" if column + "_mean" in agg.columns else column
        plt.plot(
            agg["epoch"],
            agg[mean_col],
            color=METHOD_COLORS.get(method, "gray"),
            label=METHOD_LABELS.get(method, method),
        )
        if column + "_std" in agg.columns and len(lst) > 1:
            std = agg[column + "_std"].fillna(0)
            plt.fill_between(
                agg["epoch"],
                agg[mean_col] - std,
                agg[mean_col] + std,
                color=METHOD_COLORS.get(method, "gray"),
                alpha=0.2,
            )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_time_per_epoch(runs: Dict[str, List[Dict]], out_path: str) -> None:
    plt.figure(figsize=(7, 4.5))
    methods = []
    means = []
    stds = []
    for method, lst in runs.items():
        agg = avg_over_seeds(lst)
        if "epoch_time" not in agg.columns:
            continue
        methods.append(METHOD_LABELS.get(method, method))
        means.append(float(agg["epoch_time_mean"].mean()))
        stds.append(float(agg["epoch_time_std"].mean()))
    plt.bar(methods, means, yerr=stds, capsize=4, color=[METHOD_COLORS.get(m.split()[0], "gray") for m in methods])
    plt.ylabel("Avg epoch time (s)")
    plt.title("Training time per epoch (lower is better)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_summary(runs: Dict[str, List[Dict]], out_csv: str) -> None:
    rows = []
    for method, lst in sorted(runs.items()):
        per_run = []
        for r in lst:
            df = load_metrics(r["path"])
            if df.empty:
                continue
            last = df.iloc[-1]
            best_acc = float(df["test_acc"].max()) if "test_acc" in df else float("nan")
            sharp_last = (
                float(df["sharpness"].dropna().iloc[-1])
                if "sharpness" in df and df["sharpness"].notna().any()
                else None
            )
            worst_last = (
                float(df["worst_case_loss"].dropna().iloc[-1])
                if "worst_case_loss" in df and df["worst_case_loss"].notna().any()
                else None
            )
            per_run.append({
                "run": r["run"],
                "best_test_acc": best_acc,
                "final_test_acc": float(last["test_acc"]) if "test_acc" in last else float("nan"),
                "final_test_loss": float(last["test_loss"]) if "test_loss" in last else float("nan"),
                "sharpness_last": sharp_last,
                "worst_case_loss_last": worst_last,
                "epoch_time_mean": float(last["epoch_time"]) if "epoch_time" in last else float("nan"),
                "total_time_sec": float(last["cumulative_time"]) if "cumulative_time" in last else float("nan"),
            })
        if not per_run:
            continue
        import numpy as np
        acc_arr = np.array([r["best_test_acc"] for r in per_run])
        sharp_arr = np.array([r["sharpness_last"] for r in per_run if r["sharpness_last"] is not None])
        worst_arr = np.array([r["worst_case_loss_last"] for r in per_run if r["worst_case_loss_last"] is not None])
        time_arr = np.array([r["epoch_time_mean"] for r in per_run])
        total_arr = np.array([r["total_time_sec"] for r in per_run])
        rows.append({
            "method": method,
            "n_seeds": len(per_run),
            "best_test_acc_mean": float(acc_arr.mean()) if len(acc_arr) else float("nan"),
            "best_test_acc_std": float(acc_arr.std()) if len(acc_arr) else float("nan"),
            "sharpness_mean": float(sharp_arr.mean()) if len(sharp_arr) else float("nan"),
            "worst_case_loss_mean": float(worst_arr.mean()) if len(worst_arr) else float("nan"),
            "epoch_time_mean": float(time_arr.mean()) if len(time_arr) else float("nan"),
            "total_time_sec_mean": float(total_arr.mean()) if len(total_arr) else float("nan"),
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        cols = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./nm_sam_cifar10/results")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    runs = collect_runs(args.results_dir)
    if not runs:
        print("No metrics.csv files found in", args.results_dir)
        return

    plot_metric(runs, "train_loss", "Train Loss vs Epoch", "Train Loss",
                os.path.join(out_dir, "train_loss.png"))
    plot_metric(runs, "test_loss", "Test Loss vs Epoch", "Test Loss",
                os.path.join(out_dir, "test_loss.png"))
    plot_metric(runs, "test_acc", "Test Accuracy vs Epoch", "Test Acc",
                os.path.join(out_dir, "test_acc.png"))
    plot_metric(runs, "sharpness", "Sharpness vs Epoch", "Sharpness (F_rho - L)",
                os.path.join(out_dir, "sharpness.png"))
    plot_time_per_epoch(runs, os.path.join(out_dir, "epoch_time.png"))

    write_summary(runs, os.path.join(args.results_dir, "summary.csv"))
    print("Plots saved to:", out_dir)
    print("Summary CSV:", os.path.join(args.results_dir, "summary.csv"))


if __name__ == "__main__":
    main()
