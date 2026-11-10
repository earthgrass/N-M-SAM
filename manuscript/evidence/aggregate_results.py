"""Recompute the manuscript's aggregated final-test table from copied evidence."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
historical = json.loads((root / "historical_three_seed_analysis.json").read_text(encoding="utf-8"))
methods = ("Magnitude", "Wanda", "SparseGPT", "ROSE")
summary = {
    item["method"]: {
        "runs": item["seeds"],
        "accuracy_mean": item["accuracy_mean"],
        "accuracy_sample_std": item["accuracy_sample_std"],
    }
    for item in historical["three_seed_summary"] if item["method"] in methods
}
joint = []
for label in ("sam_seed0_final_test.json", "sam_seed1_final_test.json"):
    payload = json.loads((root / label).read_text(encoding="utf-8"))
    record = payload["evaluations"]["original_sparsegpt"]
    if record["status"] != "PASS" or record["test_count"] != 10_000:
        raise RuntimeError(f"Incomplete fixed Final Test: {label}")
    joint.append(record["test_accuracy"])
summary["SAM-Joint + SparseGPT"] = {
    "runs": len(joint),
    "accuracy_mean": statistics.fmean(joint),
    "accuracy_sample_std": statistics.stdev(joint),
    "run_accuracies": joint,
}
(root / "aggregate_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
