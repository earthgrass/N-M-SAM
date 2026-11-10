"""Build main and supplemental CIFAR-100 method-endpoint tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METHODS = ("magnitude", "wanda", "sparsegpt", "rose", "apbr_sam")
LABELS = {"magnitude": "Magnitude", "wanda": "Wanda", "sparsegpt": "SparseGPT",
          "rose": "ROSE", "apbr_sam": "APBR-SAM"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) >= 2 else None


def result(test_dir: Path, name: str):
    path = test_dir / f"{name}.json"
    if not path.is_file():
        return None
    value = read(path)
    if (value.get("status") != "PASS" or value.get("test_count") != 10_000
            or value.get("selection_performed") is not False
            or value.get("hyperparameter_change_performed") is not False):
        raise RuntimeError(f"Evaluation contract failed: {path}")
    return value


def collect_seed(root: Path, seed: int):
    seed_root = root / f"seed{seed}"
    # The active full-recovery queue writes test_final for every seed. The
    # method-specific directory remains a compatible fallback for an audited
    # alternate execution, but is not used by the active queue.
    test_dir = seed_root / "test_final"
    if not test_dir.is_dir():
        test_dir = seed_root / "test_method_specific"
    if not test_dir.is_dir():
        return None
    dense = result(test_dir, "dense_final"); apbr_dense = result(test_dir, "apbr_final")
    if dense is None or apbr_dense is None:
        return None
    raw = {method: result(test_dir, f"{method}_raw") for method in METHODS}
    apbr_recovery = result(test_dir, "apbr_sam_recovery_final")
    if any(value is None for value in raw.values()) or apbr_recovery is None:
        return None
    endpoints = {
        "dense_sam_final": (dense, None),
        "apbr_sam_dense_final": (apbr_dense, None),
        "magnitude_raw_hard": (raw["magnitude"], dense),
        "wanda_raw_hard": (raw["wanda"], dense),
        "sparsegpt_native_post_pruning": (raw["sparsegpt"], dense),
        "rose_native_post_pruning": (raw["rose"], dense),
        "apbr_sam_raw_hard": (raw["apbr_sam"], apbr_dense),
        "apbr_sam_recovery_20_final": (apbr_recovery, apbr_dense),
    }
    rows = []
    for endpoint, (value, parent) in endpoints.items():
        rows.append({"seed": seed, "endpoint": endpoint, "test_accuracy": value["test_accuracy"],
                     "test_loss": value["test_loss"], "test_count": value["test_count"],
                     "checkpoint_sha256": value["checkpoint_sha256"],
                     "parent_accuracy": None if parent is None else parent["test_accuracy"],
                     "delta_vs_parent_pp": None if parent is None else 100 * (value["test_accuracy"] - parent["test_accuracy"])})
    supplemental = {}
    for method in METHODS:
        value = result(test_dir, f"{method}_recovery_final")
        supplemental[method] = None if value is None else {
            "test_accuracy": value["test_accuracy"], "test_loss": value["test_loss"],
            "test_count": value["test_count"], "checkpoint_sha256": value["checkpoint_sha256"]}
    return {"seed": seed, "rows": rows, "raw": raw, "apbr_recovery": apbr_recovery,
            "supplemental_recovery_20": supplemental}


def build(root: Path):
    seeds = [collect_seed(root, seed) for seed in (0, 1, 2)]
    completed = [item for item in seeds if item is not None]
    rows = [row for item in completed for row in item["rows"]]
    endpoints = sorted({row["endpoint"] for row in rows})
    summary = []
    for endpoint in endpoints:
        selected = [row for row in rows if row["endpoint"] == endpoint]
        mean, std = mean_std([row["test_accuracy"] for row in selected])
        delta_mean, delta_std = mean_std([row["delta_vs_parent_pp"] for row in selected])
        summary.append({"endpoint": endpoint, "seeds": len(selected), "accuracy_mean": mean,
                        "accuracy_sample_std": std, "delta_vs_parent_pp_mean": delta_mean,
                        "delta_vs_parent_pp_sample_std": delta_std})
    paired = []
    for item in completed:
        seed = item["seed"]
        apbr = item["apbr_recovery"]["test_accuracy"]
        for method in ("magnitude", "wanda", "sparsegpt", "rose"):
            paired.append({"seed": seed, "comparison": f"apbr_recovery_minus_{method}_native",
                           "difference_pp": 100 * (apbr - item["raw"][method]["test_accuracy"])})
    paired_summary = []
    for name in sorted({row["comparison"] for row in paired}):
        selected = [row["difference_pp"] for row in paired if row["comparison"] == name]
        mean, std = mean_std(selected)
        paired_summary.append({"comparison": name, "seeds": len(selected),
                               "difference_pp_mean": mean, "difference_pp_sample_std": std})
    supplemental = []
    for seed in (0, 1, 2):
        item = seeds[seed]
        for method in METHODS:
            value = None if item is None else item["supplemental_recovery_20"][method]
            supplemental.append({"seed": seed, "method": method,
                                 "test_accuracy": None if value is None else value["test_accuracy"],
                                 "test_loss": None if value is None else value["test_loss"],
                                 "test_count": None if value is None else value["test_count"],
                                 "checkpoint_sha256": None if value is None else value["checkpoint_sha256"]})
    return {"schema": 1, "comparison_protocol_id": "cifar100_method_specific_endpoint_v1",
            "completed_seeds": [item["seed"] for item in completed], "all_three_complete": len(completed) == 3,
            "main_rows": rows, "main_summary": summary, "paired_differences": paired,
            "paired_summary": paired_summary, "supplemental_recovery_20": supplemental,
            "statistics": {"standard_deviation": "sample", "ddof": 1},
            "notes": ["Main and supplemental endpoints are never ranked together.",
                      "Missing supplemental Recovery-20 values remain null, never zero."]}


def format_percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def write_outputs(value: dict, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite reporting output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "METHOD_ENDPOINT_RESULTS.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    fields = ["seed", "endpoint", "test_accuracy", "test_loss", "test_count", "checkpoint_sha256",
              "parent_accuracy", "delta_vs_parent_pp"]
    with (output / "METHOD_ENDPOINT_MAIN.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(value["main_rows"])
    supplement_fields = ["seed", "method", "test_accuracy", "test_loss", "test_count", "checkpoint_sha256"]
    with (output / "METHOD_ENDPOINT_SUPPLEMENTAL_RECOVERY20.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=supplement_fields); writer.writeheader(); writer.writerows(value["supplemental_recovery_20"])
    lines = ["# CIFAR-100 Method-Specific Endpoint Results", "",
             f"Completed seeds: {value['completed_seeds']}. Three-seed statistics complete: {value['all_three_complete']}.", "",
             "## Main table", "", "| Seed | Endpoint | Accuracy | Loss | Δ vs parent | Checkpoint SHA256 |",
             "|---:|---|---:|---:|---:|---|"]
    for row in value["main_rows"]:
        delta = "—" if row["delta_vs_parent_pp"] is None else f"{row['delta_vs_parent_pp']:+.2f} pp"
        lines.append(f"| {row['seed']} | {row['endpoint']} | {format_percent(row['test_accuracy'])} | {row['test_loss']:.4f} | {delta} | `{row['checkpoint_sha256']}` |")
    lines += ["", "## Supplemental Recovery-20", "", "Only actually completed recovery results are shown; missing entries remain blank.", "",
              "| Seed | Method | Accuracy | Loss |", "|---:|---|---:|---:|"]
    for row in value["supplemental_recovery_20"]:
        loss = "—" if row["test_loss"] is None else f"{row['test_loss']:.4f}"
        lines.append(f"| {row['seed']} | {LABELS[row['method']]} | {format_percent(row['test_accuracy'])} | {loss} |")
    if value["all_three_complete"]:
        lines += ["", "## Three-seed mean ± sample standard deviation", "",
                  "| Endpoint | Seeds | Accuracy mean | Sample std | Mean Δ vs parent |", "|---|---:|---:|---:|---:|"]
        for row in value["main_summary"]:
            acc_std = "—" if row["accuracy_sample_std"] is None else f"{100*row['accuracy_sample_std']:.2f} pp"
            delta = "—" if row["delta_vs_parent_pp_mean"] is None else f"{row['delta_vs_parent_pp_mean']:+.2f} pp"
            lines.append(f"| {row['endpoint']} | {row['seeds']} | {format_percent(row['accuracy_mean'])} | {acc_std} | {delta} |")
    (output / "METHOD_ENDPOINT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.root.resolve()); write_outputs(value, args.output.resolve())
    print(json.dumps({"completed_seeds": value["completed_seeds"], "all_three_complete": value["all_three_complete"]}))


if __name__ == "__main__":
    main()

