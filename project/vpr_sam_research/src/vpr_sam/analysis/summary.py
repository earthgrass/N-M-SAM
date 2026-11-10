"""Two-level fairness checks and seed-paired dense/hard/recovery statistics."""
from __future__ import annotations
import csv
import json

from ..config import comparison_protocol, method_protocol, canonical_pruning
from ..repro import atomic_json, project_path, file_hash, source_hashes
from .artifacts import completed_run, checkpoint_metrics, pruning_artifacts
from .statistics import seed_statistics


def postprocessing_statistics(runs, baseline, requested=None, recovery_checkpoint="final"):
    has_pruning = [bool(run["pruning"]) for run in runs]
    if not any(has_pruning):
        if requested is not None:
            raise ValueError("Requested pruning protocol is missing")
        return {"status": "not_available", "reason": "No selected hard-pruning artifacts"}
    if not all(has_pruning):
        raise ValueError("Missing selected hard-pruning result for one or more method/seed runs")
    first = runs[0]["pruning"][0]
    for run in runs:
        item = run["pruning"][0]
        if item["protocol"] != first["protocol"] or item["selection"] != first["selection"]:
            raise ValueError("Pruning protocol/selection mismatch: compare the same N:M, scope, tail, mask rule and BN policy")
    seed_sets = {}
    for run in runs:
        seed_sets.setdefault(run["method"], set()).add(run["seed"])
    if any(seeds != next(iter(seed_sets.values())) for seeds in seed_sets.values()):
        raise ValueError("Post-pruning comparisons require complete matched seed sets")
    def metric(key):
        observations = {}
        for run in runs:
            observations.setdefault(run["method"], {})[run["seed"]] = run["pruning"][0][key]
        return seed_statistics(observations, baseline)
    result = {"status": "available", "protocol": first["protocol"], "selection": first["selection"],
              "hard": {"accuracy": metric("hard_accuracy"), "loss": metric("hard_loss"),
                       "loss_increase": metric("hard_loss_increase"), "accuracy_drop": metric("accuracy_drop")}}
    recovered = ["recovered_accuracy" in run["pruning"][0] for run in runs]
    if any(recovered) and not all(recovered):
        raise ValueError("Recovery evidence must be complete for every compared method/seed")
    if all(recovered):
        common = first["recovery_protocol"]
        by_seed = {}
        for run in runs:
            item = run["pruning"][0]
            if item["recovery_protocol"] != common or item["recovery_checkpoint"] != recovery_checkpoint:
                raise ValueError("Recovery budget/optimizer/checkpoint selection mismatch")
            by_seed.setdefault(run["seed"], set()).add(item["recovery_order_hash"])
        if any(len(orders) != 1 for orders in by_seed.values()):
            raise ValueError("Recovery data-order mismatch within a paired seed")
        result["recovery"] = {"status": "available", "protocol": common, "checkpoint": recovery_checkpoint,
                              "accuracy": metric("recovered_accuracy"), "loss": metric("recovered_loss")}
    else:
        result["recovery"] = {"status": "not_available"}
    result["difference_definition"] = "method minus baseline on the same seed; lower loss/damage is better"
    return result


def summarize(root, run_paths, output, baseline="sam", split="validation", checkpoint="final",
              pruning_protocol=None, recovery_checkpoint="final"):
    if split not in {"validation", "test"} or checkpoint not in {"final", "best_validation"} or recovery_checkpoint not in {"final", "best_validation"}:
        raise ValueError("Explicit common split/checkpoint selection required")
    if pruning_protocol is not None:
        pruning_protocol = canonical_pruning(pruning_protocol)
    runs, observations, pairs, fairness, seen, method_configs = [], {}, {}, [], set(), {}
    common_environment = None
    for relative in run_paths:
        path = project_path(root, relative)
        summary, c, state, selected = completed_run(root, path, checkpoint)
        method, seed = c["optimizer"]["method"], c["seed"]
        if (method, seed) in seen:
            raise ValueError("Duplicate method/seed would create pseudoreplication")
        seen.add((method, seed))
        if runs and c["device"] != runs[0]["protocol"]["device"]:
            raise ValueError("Mixed CPU/CUDA or different approved CUDA devices cannot be compared")
        if runs and c["threads"] != runs[0]["protocol"]["threads"]:
            raise ValueError("Thread settings differ; comparisons require identical intra-op threads")
        within_method = method_protocol(c)
        if method in method_configs and method_configs[method] != within_method:
            raise ValueError(f"Within-method geometry/protocol mismatch for {method}; do not pool hyperparameter variants as seeds")
        method_configs[method] = within_method
        protocol = comparison_protocol(c)
        protocol.pop("seed")
        if runs and protocol != runs[0]["protocol"]:
            raise ValueError("Non-research protocol mismatch; results cannot be pooled")
        if common_environment is not None and common_environment != state["environment"]:
            raise ValueError("Comparison environment mismatch")
        common_environment = state["environment"]
        metrics = checkpoint_metrics(selected, state, split)
        entry = {"path": str(path.relative_to(root)), "method": method, "seed": seed,
                 "protocol": protocol, "method_protocol": within_method, "summary": summary,
                 "accuracy": metrics["accuracy"], "split": split, "checkpoint": checkpoint,
                 "checkpoint_sha256": file_hash(selected), "validation_accuracy": summary["final_validation"]["accuracy"],
                 "pruning": pruning_artifacts(root, path, state, selected, split, recovery_checkpoint, pruning_protocol)}
        runs.append(entry)
        observations.setdefault(method, {})[seed] = metrics["accuracy"]
        pairs.setdefault(seed, {})[method] = entry
    if not runs:
        raise ValueError("No runs")
    if len(observations) > 1 and baseline not in observations:
        raise ValueError("Requested paired baseline is absent")
    for seed, methods in pairs.items():
        identities = {(e["summary"]["initialization_hash"], e["summary"]["order_hash"],
                       e["summary"]["dataset_fingerprint"]) for e in methods.values()}
        if len(identities) != 1:
            raise ValueError(f"Initialization/order/data mismatch at seed {seed}")
        fairness.append({"seed": seed, "methods": sorted(methods), "paired_identity_match": True})
    dense = seed_statistics(observations, baseline)
    post = postprocessing_statistics(runs, baseline, pruning_protocol, recovery_checkpoint)
    result = {"runs": runs, **dense, "postprocessing": post, "fairness": fairness,
              "method_protocols": method_configs, "environment": common_environment, "source_hashes": source_hashes(root),
              "split": split, "checkpoint": checkpoint, "recovery_checkpoint": recovery_checkpoint,
              "note": "Seed-level paired differences. Synthetic smoke is engineering evidence, not research. Small-n bootstrap is exploratory."}
    out = project_path(root, output)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", result)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "method", "seed", "split", "checkpoint", "accuracy"])
        writer.writeheader()
        writer.writerows({key: r[key] for key in writer.fieldnames} for r in runs)
    with (out / "postprocessing.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "seed", "split", "checkpoint", "recovery_checkpoint", "n", "m", "scope", "tail", "score",
                  "hard_accuracy", "hard_loss", "hard_loss_increase", "accuracy_drop", "recovered_accuracy", "recovered_loss",
                  "hard_checkpoint_sha256", "recovery_checkpoint_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for item in run["pruning"]:
                row = {**item["protocol"], **{k: item[k] for k in fields if k in item},
                       "method": run["method"], "seed": run["seed"], "split": split, "checkpoint": checkpoint}
                writer.writerow(row)
    return result
