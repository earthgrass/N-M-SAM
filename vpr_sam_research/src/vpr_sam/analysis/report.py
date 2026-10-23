from __future__ import annotations
import json

from ..repro import project_path, source_hashes, file_hash, atomic_json
from .summary import summarize

def plot(root, summary_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = json.loads(project_path(root, summary_path).read_text(encoding="utf-8"))
    if result.get("source_hashes") != source_hashes(root):
        raise ValueError("Summary source identity mismatch; regenerate the audited summary")
    from ..training.engine import read_checkpoint
    choice, recovery_choice = result["checkpoint"], result["recovery_checkpoint"]
    if choice not in {"final", "best_validation"} or recovery_choice not in {"final", "best_validation"}:
        raise ValueError("Invalid plotting checkpoint selection")
    selected_states, sources = {}, []
    for run in result["runs"]:
        if run["checkpoint"] != choice:
            raise ValueError("Mixed summary/plot checkpoint selections")
        path = project_path(root, run["path"])
        selected, selected_path = read_checkpoint(path / f"{choice}.pt", root)
        if file_hash(selected_path) != run["checkpoint_sha256"]:
            raise ValueError("Selected plotting checkpoint/hash mismatch")
        selected_states[run["path"]] = {key: selected[key] for key in ("epoch", "epoch_rows", "step_rows")}
        selected_states[run["path"]]["stepper"] = {"geometry": selected["stepper"]["geometry"]}
        source = {"method": run["method"], "seed": run["seed"], "checkpoint": choice,
                  "path": str(selected_path.relative_to(root)), "sha256": run["checkpoint_sha256"],
                  "epoch": selected["epoch"], "steps": selected["stepper"]["steps"], "pruning": []}
        for item in run["pruning"]:
            hard_path = path / "pruning" / item["source_checkpoint"] / "pruned.pt"
            read_checkpoint(hard_path, root)
            if item["dense_checkpoint_sha256"] != source["sha256"] or file_hash(hard_path) != item["hard_checkpoint_sha256"]:
                raise ValueError("Hard-pruning plotting checkpoint/hash mismatch")
            post_source = {"hard_path": str(hard_path.relative_to(root)), "hard_sha256": item["hard_checkpoint_sha256"]}
            if "recovered_accuracy" in item:
                if item["recovery_checkpoint"] != recovery_choice:
                    raise ValueError("Mixed recovery plotting checkpoint selections")
                recovery_path = hard_path.parent / "recovery" / f"{recovery_choice}.pt"
                read_checkpoint(recovery_path, root)
                if file_hash(recovery_path) != item["recovery_checkpoint_sha256"]:
                    raise ValueError("Recovery plotting checkpoint/hash mismatch")
                post_source.update(recovery_path=str(recovery_path.relative_to(root)),
                                   recovery_sha256=item["recovery_checkpoint_sha256"], recovery_checkpoint=recovery_choice)
            source["pruning"].append(post_source)
        sources.append(source)
    out = project_path(root, output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for run in result["runs"]:
        path = project_path(root, run["path"])
        selected = selected_states[run["path"]]
        epochs, steps = selected["epoch_rows"], selected["step_rows"]
        label = f"{run['method']} seed {run['seed']} ({choice}, e{selected['epoch']})"
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
        fig.suptitle(f"SYNTHETIC SMOKE — history through {choice}; no research conclusion", fontsize=11)
    else:
        fig.suptitle(f"History through selected {choice} checkpoint", fontsize=11)
    fig.savefig(out / "training_geometry.png", dpi=180)
    fig.savefig(out / "training_geometry.pdf")
    plt.close(fig)
    # All checkpoint-dependent geometry comes from the same selected snapshot.
    import torch
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for run in result["runs"]:
        checkpoint = selected_states[run["path"]]
        logs = checkpoint["stepper"]["geometry"]["logs"]
        if not logs:
            continue  # sphere baselines have no stored anisotropic state
        # Histograms operate on log-radius to remain meaningful at extreme scales.
        values = torch.cat([s.flatten().double() for s in logs]).numpy()
        ax.hist(values, bins=60, histtype="step", label=f"{run['method']} seed {run['seed']} ({choice})")
    ax.set(xlabel="log(a)", ylabel="Coordinate count", title=f"Perturbation geometry: {choice}")
    ax.legend(fontsize=8)
    fig.savefig(out / "log_radius_histogram.png", dpi=180)
    fig.savefig(out / "log_radius_histogram.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for run in result["runs"]:
        for item in run["pruning"]:
            label = f"{run['method']} s{run['seed']} {item['protocol']['n']}:{item['protocol']['m']}"
            ax.scatter(label, item["hard_loss_increase"])
    ax.set(ylabel=f"{result['split'].title()} loss increase", title=f"Hard-pruning damage: {choice} (no BN calibration)")
    ax.tick_params(axis="x", labelrotation=35)
    fig.savefig(out / "pruning_damage.png", dpi=180)
    fig.savefig(out / "pruning_damage.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    methods = list(result["statistics"])
    means = [result["statistics"][m]["mean"] for m in methods]
    deviations = [result["statistics"][m]["sample_std"] or 0. for m in methods]
    ax.bar(methods, means, yerr=deviations, capsize=4)
    ax.set(ylabel=f"{result['split'].title()} accuracy (mean ± sample SD)", title=f"Independent seed summary: {choice}")
    ax.set_xticks(range(len(methods)), [f"{m}\nn={result['statistics'][m]['n']}" for m in methods])
    fig.savefig(out / "seed_statistics.png", dpi=180)
    fig.savefig(out / "seed_statistics.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for run in result["runs"]:
        epochs = selected_states[run["path"]]["epoch_rows"]
        if run["method"] == "vpr":
            names = epochs[-1]["geometry"]["layer_log_volume"]
            for name in names:
                axes[0].plot([e["epoch"] for e in epochs],
                             [e["geometry"]["layer_log_volume"][name] for e in epochs], linewidth=.7, alpha=.6)
        for item in run["pruning"]:
            if "recovered_accuracy" in item:
                label=f"{run['method']} s{run['seed']}"
                axes[1].plot([0, 1], [item["hard_accuracy"], item["recovered_accuracy"]], marker="o", label=label)
    axes[0].set(title=f"VPR migration through {choice}", xlabel="Epoch", ylabel="sum log(a) in tensor")
    axes[1].set(title=f"Recovery: {choice} → {recovery_choice}", ylabel=f"{result['split'].title()} accuracy")
    axes[1].set_xticks([0,1],["Hard-pruned","Recovered"])
    if axes[1].lines: axes[1].legend(fontsize=7)
    fig.savefig(out / "migration_recovery.png", dpi=180)
    fig.savefig(out / "migration_recovery.pdf")
    plt.close(fig)
    figure_count = 5
    post = result.get("postprocessing", {})
    if post.get("status") == "available":
        stages = [(f"Hard-pruned accuracy ({choice})", post["hard"]["accuracy"])]
        if post["recovery"]["status"] == "available":
            stages.append((f"Recovered accuracy ({recovery_choice})", post["recovery"]["accuracy"]))
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
        title = f"Paired differences: {choice} / recovery {recovery_choice}; 95% bootstrap interval"
        if result["runs"][0]["protocol"]["purpose"] == "smoke":
            title = "Synthetic smoke: " + title
        ax.set(xlabel="Same-seed accuracy difference", title=title)
        fig.savefig(out / "paired_pruning_recovery_differences.png", dpi=180)
        fig.savefig(out / "paired_pruning_recovery_differences.pdf")
        plt.close(fig)
        figure_count += 2
    atomic_json(out / "figure_sources.json", {"checkpoint": choice, "recovery_checkpoint": recovery_choice, "runs": sources})
    return {"output": str(out), "figures": figure_count, "checkpoint": choice,
            "recovery_checkpoint": recovery_choice, "source_manifest": str(out / "figure_sources.json")}
