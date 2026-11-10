"""Read-only preflight for APBR-parent Magnitude-vs-Geometry ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers, validate_masks

ABLATION_ID = "ablation_apbr_parent_mag_vs_geometry_20261027"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_json(path: Path, value) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def audit_seed(seed: int, root: Path) -> dict:
    parent_path = root / "apbr_sam/final_epoch_200.pt"
    raw_path = root / "methods/apbr_sam/raw_hard.pt"
    recovery_path = root / "methods/apbr_sam/recovery/final_epoch_020.pt"
    dense_eval_path = root / "test_final/apbr_final.json"
    raw_eval_path = root / "test_final/apbr_sam_raw.json"
    recovery_eval_path = root / "test_final/apbr_sam_recovery_final.json"
    formal_magnitude_recovery_path = (
        root / "methods/magnitude/recovery/final_epoch_020.pt"
    )
    required = [
        parent_path,
        raw_path,
        recovery_path,
        dense_eval_path,
        raw_eval_path,
        recovery_eval_path,
        formal_magnitude_recovery_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Seed {seed} missing: {missing}")

    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
    formal_magnitude_recovery = torch.load(
        formal_magnitude_recovery_path, map_location="cpu", weights_only=False
    )
    parent_sha = sha(parent_path)
    raw_sha = sha(raw_path)
    recovery_sha = sha(recovery_path)
    if (
        parent.get("epoch") != 200
        or parent.get("steps") != 78_200
        or parent.get("identity", {}).get("method") != "apbr_sam"
        or parent.get("identity", {}).get("seed") != seed
        or parent.get("test_accessed") is not False
    ):
        raise ValueError(f"Seed {seed} APBR parent identity failed")
    train.validate_checkpoint(parent, parent["identity"])
    geometry = parent.get("stepper", {}).get("geometry", {})
    if (
        geometry.get("method") != "vpr"
        or len(geometry.get("mapping", [])) != len(geometry.get("logs", []))
    ):
        raise ValueError(f"Seed {seed} geometry state mismatch")
    if raw.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError(f"Seed {seed} Geometry Raw parent hash mismatch")
    if recovery.get("parent", {}).get("sha256") != raw_sha:
        raise ValueError(f"Seed {seed} Geometry Recovery parent hash mismatch")
    if recovery.get("epoch") != 20 or recovery.get("steps") != 7_820:
        raise ValueError(f"Seed {seed} Geometry Recovery budget mismatch")

    model = build_cifar100_model()
    model.load_state_dict(raw["model"])
    validate_masks(model, raw["masks"], n=2, m=4, complete=True)
    layers = eligible_layers(model)
    if len(layers) != 19 or len(raw["masks"]) != 19:
        raise ValueError(f"Seed {seed} eligible-layer contract mismatch")
    eligible = deleted = survivor_mismatch = deleted_nonzero = 0
    per_layer = []
    for layer_name, layer in layers.items():
        key = layer_name + ".weight"
        mask = raw["masks"][key].bool()
        dense_weight = parent["model"][key]
        raw_weight = raw["model"][key]
        layer_deleted = int((~mask).sum())
        eligible += mask.numel()
        deleted += layer_deleted
        survivor_mismatch += int((raw_weight[mask] != dense_weight[mask]).sum())
        deleted_nonzero += int((raw_weight[~mask] != 0).sum())
        per_layer.append(
            {
                "layer": layer_name,
                "positions": mask.numel(),
                "deleted": layer_deleted,
            }
        )
    if (
        eligible != 11_157_504
        or deleted != 5_578_752
        or survivor_mismatch
        or deleted_nonzero
    ):
        raise ValueError(f"Seed {seed} Geometry mask/weight audit failed")

    geometry_orders = [row["train_order_sha256"] for row in recovery["history"]]
    magnitude_orders = [
        row["train_order_sha256"] for row in formal_magnitude_recovery["history"]
    ]
    if len(geometry_orders) != 20 or geometry_orders != magnitude_orders:
        raise ValueError(f"Seed {seed} paired recovery order evidence failed")

    dense_eval = json.loads(dense_eval_path.read_text(encoding="utf-8"))
    raw_eval = json.loads(raw_eval_path.read_text(encoding="utf-8"))
    recovery_eval = json.loads(recovery_eval_path.read_text(encoding="utf-8"))
    if (
        dense_eval["checkpoint_sha256"] != parent_sha
        or raw_eval["checkpoint_sha256"] != raw_sha
        or recovery_eval["checkpoint_sha256"] != recovery_sha
        or any(item["test_count"] != 10_000 for item in [dense_eval, raw_eval, recovery_eval])
    ):
        raise ValueError(f"Seed {seed} fixed evaluation binding failed")

    return {
        "seed": seed,
        "formal_root": str(root.resolve()),
        "parent": {
            "path": str(parent_path.resolve()),
            "sha256": parent_sha,
            "model_sha256": parent["model_sha256"],
            "epoch": 200,
            "steps": 78_200,
        },
        "geometry": {
            "state_method": geometry["method"],
            "mapping_count": len(geometry["mapping"]),
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": raw_sha,
            "mask_sha256": raw["mask_sha256"],
            "recovery_path": str(recovery_path.resolve()),
            "recovery_sha256": recovery_sha,
            "raw_accuracy": raw_eval["test_accuracy"],
            "raw_loss": raw_eval["test_loss"],
            "recovery_accuracy": recovery_eval["test_accuracy"],
            "recovery_loss": recovery_eval["test_loss"],
        },
        "dense_accuracy": dense_eval["test_accuracy"],
        "dense_loss": dense_eval["test_loss"],
        "mask_audit": {
            "eligible_layers": 19,
            "eligible_positions": eligible,
            "deleted_positions": deleted,
            "survivor_mismatch": survivor_mismatch,
            "deleted_nonzero": deleted_nonzero,
            "per_layer": per_layer,
        },
        "randomness_control": {
            "matched_train_order_epochs": 20,
            "orders_match_existing_geometry_vs_formal_magnitude": True,
            "loader_seed_rule": "seed*1000003 + epoch*97 + 1",
            "worker_seed_source": "explicit DataLoader generator",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for seed in range(3):
        parser.add_argument(f"--seed{seed}-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(json.dumps(__import__("yaml").safe_load(args.config.read_text())))
    recovery = config["recovery"]
    expected_recovery = {
        "epochs": 20,
        "batch_size": 128,
        "optimizer": "SGD",
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nesterov": False,
        "scheduler": "CosineAnnealingLR_epoch_end",
        "eta_min": 0.0,
        "fixed_mask": True,
        "selection_mode": "fixed_final_epoch",
        "primary_checkpoint": "final_epoch_020.pt",
        "parent": "native_pruned_not_bn_only",
    }
    if recovery != expected_recovery:
        raise ValueError("Frozen Recovery-20 configuration mismatch")
    seeds = [
        audit_seed(seed, getattr(args, f"seed{seed}_root")) for seed in range(3)
    ]
    evidence = {
        "status": "PASS",
        "ablation_id": ABLATION_ID,
        "config": {"path": str(args.config.resolve()), "sha256": sha(args.config)},
        "runner": {"path": str(args.runner.resolve()), "sha256": sha(args.runner)},
        "recovery_config": recovery,
        "seeds": seeds,
        "decision": (
            "Reuse existing Geometry Recovery-20. Run only APBR-parent Magnitude; "
            "the formal deterministic loader binds order and augmentation worker "
            "seeds to (seed, epoch), and all 20 recorded order hashes match."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "preflight_evidence.json", evidence)
    lines = [
        "# APBR Parent Magnitude vs Geometry — Preflight Audit",
        "",
        "Status: **PASS**",
        "",
        "The three fixed APBR-SAM Epoch-200 parents, matching geometry states, "
        "Geometry Raw/Recovery checkpoints, masks, and final evaluations were "
        "verified read-only. Existing Geometry recovery can be reused for a paired "
        "ablation because loader order and augmentation worker seeds are explicit "
        "functions of `(seed, epoch)`; all 20 recorded order hashes match.",
        "",
        "| Seed | APBR parent SHA256 | Geometry mask SHA256 | Raw | Recovery-20 |",
        "|---:|---|---|---:|---:|",
    ]
    for item in seeds:
        lines.append(
            f"| {item['seed']} | `{item['parent']['sha256']}` | "
            f"`{item['geometry']['mask_sha256']}` | "
            f"{100*item['geometry']['raw_accuracy']:.2f}% | "
            f"{100*item['geometry']['recovery_accuracy']:.2f}% |"
        )
    lines += [
        "",
        "Each Geometry mask has 19 layers, 11,157,504 eligible positions, and "
        "5,578,752 deleted positions. Survivors equal their dense-parent values "
        "before recovery; deleted values are exactly zero.",
        "",
        "The new Magnitude branch will use the identical OIHW 2:4 implementation, "
        "Recovery-20 hyperparameters, and final-only Test evaluator in an isolated "
        "ablation directory.",
    ]
    atomic_text(args.output / "PREFLIGHT_AUDIT.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "seeds": 3, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
