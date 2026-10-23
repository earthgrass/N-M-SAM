from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import numpy as np

from ..config import comparison_protocol
from ..repro import atomic_json, project_path, seed_for, file_hash


def summarize(root, run_paths, output, baseline="sam", split="validation", checkpoint="final"):
    if split not in {"validation", "test"} or checkpoint not in {"final", "best_validation"}:
        raise ValueError("Explicit common split/checkpoint selection required")
    runs, groups, pairs, fairness, seen = [], {}, {}, [], set()
    for relative in run_paths:
        path = project_path(root, relative)
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        c = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if not summary["complete"] or summary["stage"] != "train":
            raise ValueError("Summarize completed dense-training runs only")
        method, seed = c["optimizer"]["method"], c["seed"]
        if (method, seed) in seen:
            raise ValueError("Duplicate method/seed would create pseudoreplication")
        seen.add((method, seed))
        protocol = comparison_protocol(c)
        protocol.pop("seed")
        if runs and protocol != runs[0]["protocol"]:
            raise ValueError("Non-research protocol mismatch; results cannot be pooled")
        selected = path / f"{checkpoint}.pt"
        if not selected.is_file():
            raise ValueError(f"Missing selected checkpoint: {selected}")
        selected_hash = file_hash(selected)
        if split == "test":
            evidence = path / f"evaluation_{checkpoint}_test.json"
            if not evidence.is_file():
                raise ValueError("Explicit frozen-test evaluation required before test aggregation")
            evaluated = json.loads(evidence.read_text(encoding="utf-8"))
            if evaluated["checkpoint_sha256"] != selected_hash:
                raise ValueError("Test evaluation/checkpoint identity mismatch")
            accuracy = evaluated["metrics"]["accuracy"]
        elif checkpoint == "final":
            accuracy = summary["final_validation"]["accuracy"]
        else:
            accuracy = summary["best_validation_accuracy"]
        entry = {"path": str(path.relative_to(root)), "method": method, "seed": seed,
                 "protocol": protocol, "summary": summary,
                 "accuracy": accuracy, "split": split, "checkpoint": checkpoint,
                 "checkpoint_sha256": selected_hash,
                 "validation_accuracy": summary["final_validation"]["accuracy"]}
        prune_reports = []
        for report_path in sorted((path / "pruning").glob("*/pruning.json")):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["parent_sha256"] != selected_hash:
                continue  # never mix pruning from final and best-validation sources
            item = {"protocol": report["protocol"], "hard_loss_increase": report["hard_loss_increase"],
                    "hard_accuracy": report["hard_validation"]["accuracy"],
                    "source_checkpoint": report_path.parent.name}
            recovery = report_path.parent / "recovery" / "summary.json"
            if recovery.exists():
                result = json.loads(recovery.read_text(encoding="utf-8"))
                item["recovered_accuracy"] = result["final_validation"]["accuracy"]
            prune_reports.append(item)
        entry["pruning"] = prune_reports
        runs.append(entry)
        groups.setdefault(method, []).append(entry["accuracy"])
        pairs.setdefault(seed, {})[method] = entry
    if not runs:
        raise ValueError("No runs")
    for seed, methods in pairs.items():
        identities = {(e["summary"]["initialization_hash"], e["summary"]["order_hash"],
                       e["summary"]["dataset_fingerprint"]) for e in methods.values()}
        if len(identities) != 1:
            raise ValueError(f"Initialization/order/data mismatch at seed {seed}")
        fairness.append({"seed": seed, "methods": sorted(methods), "paired_identity_match": True})
    statistics = {method: {"n": len(values), "mean": float(np.mean(values)),
                           "sample_std": float(np.std(values, ddof=1)) if len(values)>1 else None}
                  for method, values in groups.items()}
    paired = {}
    for method in groups:
        if method == baseline:
            continue
        seeds = sorted(s for s, methods in pairs.items() if baseline in methods and method in methods)
        deltas = np.array([pairs[s][method]["accuracy"]-pairs[s][baseline]["accuracy"] for s in seeds])
        interval = None
        if len(deltas) >= 2:
            rng = np.random.default_rng(seed_for(0, "paired-bootstrap", method, baseline))
            means = deltas[rng.integers(0, len(deltas), size=(10000, len(deltas)))].mean(1)
            interval = np.quantile(means, [.025, .975]).tolist()
        paired[method] = {"baseline": baseline, "seeds": seeds, "n": len(seeds),
                          "mean_difference": float(deltas.mean()) if len(deltas) else None,
                          "paired_bootstrap_95_percentile_interval": interval,
                          "unpaired_seeds": sorted(set(pairs)-set(seeds))}
    out = project_path(root, output)
    out.mkdir(parents=True, exist_ok=True)
    result = {"runs": runs, "statistics": statistics, "paired": paired, "fairness": fairness,
              "split": split, "checkpoint": checkpoint,
              "note": "Synthetic runs are engineering evidence only. Small-n bootstrap is exploratory. Training plots use validation even when final test scores are aggregated."}
    atomic_json(out / "summary.json", result)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "method", "seed", "split", "checkpoint", "accuracy"])
        writer.writeheader()
        writer.writerows({key: r[key] for key in writer.fieldnames} for r in runs)
    return result


def plot(root, summary_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = json.loads(project_path(root, summary_path).read_text(encoding="utf-8"))
    out = project_path(root, output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for run in result["runs"]:
        path = project_path(root, run["path"])
        epochs = json.loads((path / "epochs.json").read_text(encoding="utf-8"))
        steps = json.loads((path / "steps.json").read_text(encoding="utf-8"))
        label = f"{run['method']} seed {run['seed']}"
        x = [r["epoch"] for r in epochs]
        axes[0, 0].plot(x, [r["train_loss"] for r in epochs], label=label)
        axes[0, 1].plot(x, [r["validation"]["accuracy"] for r in epochs], label=label)
        axes[1, 0].plot([r["step"] for r in steps], [r["log_volume"] for r in steps], label=label)
        axes[1, 1].plot([r["step"] for r in steps], [r["eps_norm"] for r in steps], label=label)
    for ax, title, xlabel in zip(axes.flat, ("Training loss", "Validation accuracy", "Global log-volume", "Actual perturbation L2"),
                                ("Epoch", "Epoch", "Step", "Step")):
        ax.set(title=title, xlabel=xlabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[0, 0].ticklabel_format(axis="y", useOffset=False)
    axes[0, 1].set_ylim(0, 1)
    if result["runs"][0]["protocol"]["purpose"] == "smoke":
        fig.suptitle("SYNTHETIC SMOKE — engineering validation, not research results", fontsize=11)
    fig.savefig(out / "training_geometry.png", dpi=180)
    fig.savefig(out / "training_geometry.pdf")
    plt.close(fig)
    # Geometry histograms and per-layer migration from method's own final state.
    import torch
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for run in result["runs"]:
        checkpoint = torch.load(project_path(root, run["path"]) / "final.pt", map_location="cpu", weights_only=False)
        logs = checkpoint["stepper"]["geometry"]["logs"]
        if not logs:
            continue  # sphere baselines have no stored anisotropic state
        # Histograms operate on log-radius to remain meaningful at extreme scales.
        values = torch.cat([s.flatten().double() for s in logs]).numpy()
        ax.hist(values, bins=60, histtype="step", label=f"{run['method']} seed {run['seed']}")
    ax.set(xlabel="log(a)", ylabel="Coordinate count", title="Final perturbation geometry")
    ax.legend(fontsize=8)
    fig.savefig(out / "log_radius_histogram.png", dpi=180)
    fig.savefig(out / "log_radius_histogram.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for run in result["runs"]:
        for item in run["pruning"]:
            label = f"{run['method']} s{run['seed']} {item['protocol']['n']}:{item['protocol']['m']}"
            ax.scatter(label, item["hard_loss_increase"])
    ax.set(ylabel="Validation loss increase", title="Immediate hard-pruning damage (no BN calibration)")
    ax.tick_params(axis="x", labelrotation=35)
    fig.savefig(out / "pruning_damage.png", dpi=180)
    fig.savefig(out / "pruning_damage.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    methods = list(result["statistics"])
    means = [result["statistics"][m]["mean"] for m in methods]
    deviations = [result["statistics"][m]["sample_std"] or 0. for m in methods]
    ax.bar(methods, means, yerr=deviations, capsize=4)
    ax.set(ylabel=f"{result['split'].title()} accuracy (mean ± sample SD)", title="Independent seed summary; n shown below")
    ax.set_xticks(range(len(methods)), [f"{m}\nn={result['statistics'][m]['n']}" for m in methods])
    fig.savefig(out / "seed_statistics.png", dpi=180)
    fig.savefig(out / "seed_statistics.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for run in result["runs"]:
        epochs = json.loads((project_path(root, run["path"]) / "epochs.json").read_text(encoding="utf-8"))
        if run["method"] == "vpr":
            names = epochs[-1]["geometry"]["layer_log_volume"]
            for name in names:
                axes[0].plot([e["epoch"] for e in epochs],
                             [e["geometry"]["layer_log_volume"][name] for e in epochs], linewidth=.7, alpha=.6)
        for item in run["pruning"]:
            if "recovered_accuracy" in item:
                label=f"{run['method']} s{run['seed']}"
                axes[1].plot([0, 1], [item["hard_accuracy"], item["recovered_accuracy"]], marker="o", label=label)
    axes[0].set(title="VPR per-tensor volume migration", xlabel="Epoch", ylabel="sum log(a) in tensor")
    axes[1].set(title="Fixed-mask recovery", ylabel="Validation accuracy")
    axes[1].set_xticks([0,1],["Hard-pruned","Recovered"])
    if axes[1].lines: axes[1].legend(fontsize=7)
    fig.savefig(out / "migration_recovery.png", dpi=180)
    fig.savefig(out / "migration_recovery.pdf")
    plt.close(fig)
    return {"output": str(out), "figures": 5}
