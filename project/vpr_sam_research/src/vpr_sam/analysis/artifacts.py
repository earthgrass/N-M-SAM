"""Validate a complete parent -> hard prune -> recovery evidence chain."""
from __future__ import annotations
import copy
import json

from ..config import canonical_pruning, recovery_config
from ..provenance import artifact_reference, verify_evaluation
from ..repro import file_hash
from ..training.engine import read_checkpoint


def completed_run(root, path, checkpoint="final", stage="train"):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    final, final_path = read_checkpoint(path / "final.pt", root)
    if not summary.get("complete") or summary.get("stage") != stage or final["stage"] != stage:
        raise ValueError("Summarize only completed matching-stage runs")
    if config != final["config"] or final["epoch"] != config["training"]["epochs"] or final["next_batch"] != 0:
        raise ValueError("Run config/completion identity mismatch")
    for key, expected in {"method": config["optimizer"]["method"], "seed": config["seed"],
                          "initialization_hash": final["initialization_hash"], "order_hash": final["order_hash"],
                          "dataset_fingerprint": final["manifest"]["fingerprint"],
                          "steps": final["stepper"]["steps"], "parent": final["parent"],
                          "final_validation": final["epoch_rows"][-1]["validation"],
                          "best_validation_accuracy": final["best_validation_accuracy"]}.items():
        if summary.get(key) != expected:
            raise ValueError(f"Run summary/checkpoint mismatch: {key}")
    if summary.get("checkpoint_sha256", {}).get("final") != file_hash(final_path):
        raise ValueError("Final checkpoint/summary hash mismatch")
    selected, selected_path = (final, final_path) if checkpoint == "final" else read_checkpoint(path / f"{checkpoint}.pt", root)
    if selected["stage"] != stage or selected["config"] != config:
        raise ValueError("Selected checkpoint/stage/config mismatch")
    if summary["checkpoint_sha256"].get(checkpoint) != file_hash(selected_path):
        raise ValueError("Selected checkpoint/summary hash mismatch")
    return summary, config, selected, selected_path


def checkpoint_metrics(path, state, split):
    if split == "validation":
        if state["stage"] == "pruned":
            return state["report"]["hard_validation"]
        return state["epoch_rows"][-1]["validation"]
    evidence = path.parent / f"evaluation_{path.stem}_{split}.json"
    if not evidence.is_file():
        raise ValueError("Explicit frozen-test evaluation required for every compared stage")
    result = json.loads(evidence.read_text(encoding="utf-8"))
    verify_evaluation(result, state, path, split)
    return result["metrics"]


def pruning_artifacts(root, path, state, selected_path, split, recovery_checkpoint="final", requested=None):
    items = []
    selected_hash = file_hash(selected_path)
    for report_path in sorted((path / "pruning").glob("*/pruning.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("parent_sha256") != selected_hash:
            continue  # results from another explicitly selected checkpoint are not compared
        protocol = canonical_pruning(report["protocol"])
        if requested is not None and protocol != canonical_pruning(requested):
            continue
        hard, hard_path = read_checkpoint(report_path.parent / "pruned.pt", root)
        if hard["stage"] != "pruned" or hard["report"] != report:
            raise ValueError("Hard-pruning JSON/checkpoint report mismatch")
        if hard["parent"]["sha256"] != selected_hash or report.get("provenance") != artifact_reference(state, selected_path):
            raise ValueError("Hard-pruning parent/provenance mismatch")
        expected_config = copy.deepcopy(state["config"])
        expected_config["pruning"] = protocol
        if hard["config"] != expected_config or hard["initialization_hash"] != state["initialization_hash"]:
            raise ValueError("Hard-pruning config/initialization mismatch")
        if hard["manifest"] != state["manifest"]:
            raise ValueError("Hard-pruning data identity mismatch")
        dense_metrics = checkpoint_metrics(selected_path, state, split)
        hard_metrics = checkpoint_metrics(hard_path, hard, split)
        item = {"protocol": protocol, "selection": report["selection"], "sparsity": report["sparsity"],
                "hard_loss_increase": hard_metrics["loss"] - dense_metrics["loss"],
                "hard_accuracy": hard_metrics["accuracy"], "hard_loss": hard_metrics["loss"],
                "accuracy_drop": dense_metrics["accuracy"] - hard_metrics["accuracy"],
                "source_checkpoint": report_path.parent.name, "split": split,
                "hard_checkpoint_sha256": file_hash(hard_path), "dense_checkpoint_sha256": selected_hash}
        recovery_dir = report_path.parent / "recovery"
        if recovery_dir.exists():
            if not (recovery_dir / "summary.json").is_file():
                raise ValueError("Incomplete recovery run; do not pool partial evidence")
            summary, config, recovered, recovered_path = completed_run(root, recovery_dir, recovery_checkpoint, "recovery")
            if config != recovery_config(hard["config"]):
                raise ValueError("Recovery protocol does not match its approved pruning parent")
            if recovered["parent"]["sha256"] != file_hash(hard_path):
                raise ValueError("Recovery parent identity mismatch")
            if recovered["manifest"] != hard["manifest"] or recovered["initialization_hash"] != hard["initialization_hash"]:
                raise ValueError("Recovery data/initialization identity mismatch")
            metrics = checkpoint_metrics(recovered_path, recovered, split)
            item.update(recovered_accuracy=metrics["accuracy"], recovered_loss=metrics["loss"],
                        recovery_protocol=copy.deepcopy(hard["config"]["recovery"]),
                        recovery_checkpoint=recovery_checkpoint, recovery_order_hash=summary["order_hash"],
                        recovery_checkpoint_sha256=file_hash(recovered_path))
        items.append(item)
    if len(items) > 1:
        raise ValueError("Multiple pruning protocols: select exactly one using --pruning-protocol")
    return items
