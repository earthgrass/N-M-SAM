from __future__ import annotations
import json

from ..repro import project_path, source_hashes
from .summary import summarize

def plot(root, summary_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = json.loads(project_path(root, summary_path).read_text(encoding="utf-8"))
    if result.get("source_hashes") != source_hashes(root):
        raise ValueError("Summary source identity mismatch; regenerate the audited summary")
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
    from ..training.engine import read_checkpoint
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for run in result["runs"]:
        checkpoint, _ = read_checkpoint(project_path(root, run["path"]) / "final.pt", root)
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
    figure_count = 5
    post = result.get("postprocessing", {})
    if post.get("status") == "available":
        stages = [("Hard-pruned accuracy", post["hard"]["accuracy"])]
        if post["recovery"]["status"] == "available":
            stages.append(("Recovered accuracy", post["recovery"]["accuracy"]))
        fig, axes = plt.subplots(1, len(stages), figsize=(6 * len(stages), 4), squeeze=False, constrained_layout=True)
        for ax, (title, stats) in zip(axes.flat, stages):
            methods = list(stats["statistics"])
            means = [stats["statistics"][m]["mean"] for m in methods]
            errors = [stats["statistics"][m]["sample_std"] or 0. for m in methods]
            ax.bar(methods, means, yerr=errors, capsize=4)
            ax.set(title=title, ylabel=f"{result['split'].title()} accuracy: mean ± sample SD", ylim=(0, 1))
            ax.set_xticks(range(len(methods)), [f"{m}\nn={stats['statistics'][m]['n']}" for m in methods])
        fig.savefig(out / "paired_pruning_recovery_means.png", dpi=180)
        fig.savefig(out / "paired_pruning_recovery_means.pdf")
        plt.close(fig)
        comparisons = [(title, method, paired) for title, stats in stages
                       for method, paired in stats["paired"].items() if paired["n"]]
        fig, ax = plt.subplots(figsize=(9, max(3.5, .55 * len(comparisons))), constrained_layout=True)
        labels = []
        for index, (title, method, paired) in enumerate(comparisons):
            stage_label = "Hard" if title.startswith("Hard") else "Recovery"
            labels.append(f"{stage_label}: {method} - {paired['baseline']} (n={paired['n']})")
            ax.scatter(list(paired["differences_by_seed"].values()), [index] * paired["n"], alpha=.6)
            ci = paired["paired_bootstrap_95_percentile_interval"]
            if ci is not None:
                ax.plot(ci, [index, index], color="black", linewidth=2)
            ax.scatter(paired["mean_difference"], index, marker="D", color="black")
        ax.set_yticks(range(len(labels)), labels)
        ax.axvline(0, color="grey", linestyle="--")
        title = "Paired differences; bars = bootstrap percentile 95% interval"
        if result["runs"][0]["protocol"]["purpose"] == "smoke":
            title = "Synthetic smoke: " + title
        ax.set(xlabel="Same-seed accuracy difference", title=title)
        fig.savefig(out / "paired_pruning_recovery_differences.png", dpi=180)
        fig.savefig(out / "paired_pruning_recovery_differences.pdf")
        plt.close(fig)
        figure_count += 2
    return {"output": str(out), "figures": figure_count}
