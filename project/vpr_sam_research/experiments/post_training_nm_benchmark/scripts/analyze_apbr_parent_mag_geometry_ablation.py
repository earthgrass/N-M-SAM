"""Audit and analyze the three-seed APBR-parent Magnitude-vs-Geometry ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import subprocess
from pathlib import Path
from statistics import mean, stdev

import torch

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
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict:
    return {"mean": mean(values), "sample_std": stdev(values), "n": len(values)}


def audit_seed(seed: int, formal_root: Path, ablation_root: Path) -> tuple[dict, list[dict]]:
    parent_path = formal_root / "apbr_sam/final_epoch_200.pt"
    geometry_raw_path = formal_root / "methods/apbr_sam/raw_hard.pt"
    geometry_recovery_path = formal_root / "methods/apbr_sam/recovery/final_epoch_020.pt"
    magnitude_root = ablation_root / f"seed{seed}/magnitude_apbr_parent"
    magnitude_raw_path = magnitude_root / "raw_hard.pt"
    magnitude_recovery_path = magnitude_root / "recovery/final_epoch_020.pt"
    paths = [parent_path, geometry_raw_path, geometry_recovery_path, magnitude_raw_path, magnitude_recovery_path]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Seed {seed} checkpoint set incomplete")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    geometry_raw = torch.load(geometry_raw_path, map_location="cpu", weights_only=False)
    geometry_recovery = torch.load(geometry_recovery_path, map_location="cpu", weights_only=False)
    magnitude_raw = torch.load(magnitude_raw_path, map_location="cpu", weights_only=False)
    magnitude_recovery = torch.load(magnitude_recovery_path, map_location="cpu", weights_only=False)
    parent_sha = sha(parent_path)
    geometry_raw_sha = sha(geometry_raw_path)
    magnitude_raw_sha = sha(magnitude_raw_path)
    if geometry_raw["parent"]["sha256"] != parent_sha or magnitude_raw["parent"]["sha256"] != parent_sha:
        raise ValueError(f"Seed {seed} common parent binding failed")
    if geometry_recovery["parent"]["sha256"] != geometry_raw_sha or magnitude_recovery["parent"]["sha256"] != magnitude_raw_sha:
        raise ValueError(f"Seed {seed} recovery parent binding failed")
    if magnitude_raw.get("ablation_id") != ABLATION_ID or magnitude_recovery.get("ablation_id") != ABLATION_ID:
        raise ValueError(f"Seed {seed} ablation identity failed")
    if magnitude_recovery.get("epoch") != 20 or magnitude_recovery.get("steps") != 7820:
        raise ValueError(f"Seed {seed} magnitude recovery budget failed")
    if any(value.get("test_accessed") is not False for value in [parent, geometry_raw, geometry_recovery, magnitude_raw, magnitude_recovery]):
        raise ValueError(f"Seed {seed} Test isolation failed")

    geometry_orders = [row["train_order_sha256"] for row in geometry_recovery["history"]]
    magnitude_orders = [row["train_order_sha256"] for row in magnitude_recovery["history"]]
    if geometry_orders != magnitude_orders or len(magnitude_orders) != 20:
        raise ValueError(f"Seed {seed} paired recovery train order mismatch")

    dense_eval = load_json(formal_root / "test_final/apbr_final.json")
    geometry_raw_eval = load_json(formal_root / "test_final/apbr_sam_raw.json")
    geometry_rec_eval = load_json(formal_root / "test_final/apbr_sam_recovery_final.json")
    magnitude_raw_eval = load_json(ablation_root / f"seed{seed}/test_final/magnitude_raw.json")
    magnitude_rec_eval = load_json(ablation_root / f"seed{seed}/test_final/magnitude_recovery_final.json")
    magnitude_status = load_json(magnitude_root / "status.json")
    bindings = [
        (dense_eval, parent_path),
        (geometry_raw_eval, geometry_raw_path),
        (geometry_rec_eval, geometry_recovery_path),
        (magnitude_raw_eval, magnitude_raw_path),
        (magnitude_rec_eval, magnitude_recovery_path),
    ]
    if any(item["checkpoint_sha256"] != sha(path) or item["test_count"] != 10_000 for item, path in bindings):
        raise ValueError(f"Seed {seed} evaluation binding failed")

    logs = {
        name: value
        for (name, _), value in zip(
            parent["stepper"]["geometry"]["mapping"],
            parent["stepper"]["geometry"]["logs"],
        )
    }
    layer_rows = []
    total_positions = total_flips = 0
    magnitude_proxy = geometry_proxy = 0.0
    magnitude_proxy_count = geometry_proxy_count = 0
    for key in sorted(geometry_raw["masks"]):
        geometry_mask = geometry_raw["masks"][key].bool()
        magnitude_mask = magnitude_raw["masks"][key].bool()
        if geometry_mask.shape != magnitude_mask.shape:
            raise ValueError(f"Seed {seed} {key} mask shape mismatch")
        dense_weight = parent["model"][key]
        magnitude_weight = magnitude_raw["model"][key]
        magnitude_recovery_weight = magnitude_recovery["model"][key]
        if torch.any(magnitude_weight[~magnitude_mask] != 0) or torch.any(magnitude_recovery_weight[~magnitude_mask] != 0):
            raise ValueError(f"Seed {seed} {key} deleted weights regrew")
        if torch.any(magnitude_weight[magnitude_mask] != dense_weight[magnitude_mask]):
            raise ValueError(f"Seed {seed} {key} Raw survivor changed")
        if not torch.equal(magnitude_mask, magnitude_recovery["masks"][key].bool()):
            raise ValueError(f"Seed {seed} {key} recovery mask changed")
        flips = int(torch.count_nonzero(geometry_mask != magnitude_mask))
        positions = geometry_mask.numel()
        score = dense_weight.abs().double() * torch.exp(-logs[key].double())
        mag_deleted_score = float(score[~magnitude_mask].sum())
        geo_deleted_score = float(score[~geometry_mask].sum())
        magnitude_proxy += mag_deleted_score
        geometry_proxy += geo_deleted_score
        magnitude_proxy_count += int((~magnitude_mask).sum())
        geometry_proxy_count += int((~geometry_mask).sum())
        total_positions += positions
        total_flips += flips
        layer_rows.append(
            {
                "seed": seed,
                "layer": key.removesuffix(".weight"),
                "positions": positions,
                "magnitude_deleted": int((~magnitude_mask).sum()),
                "geometry_deleted": int((~geometry_mask).sum()),
                "mask_flip_count": flips,
                "mask_flip_rate": flips / positions,
                "magnitude_deleted_geometry_score_sum": mag_deleted_score,
                "geometry_deleted_geometry_score_sum": geo_deleted_score,
            }
        )
    if total_positions != 11_157_504 or magnitude_proxy_count != 5_578_752 or geometry_proxy_count != 5_578_752:
        raise ValueError(f"Seed {seed} global N:M counts failed")
    if geometry_proxy > magnitude_proxy + 1e-9:
        raise ValueError(f"Seed {seed} Geometry did not minimize its own frozen proxy")

    dense_acc, dense_loss = dense_eval["test_accuracy"], dense_eval["test_loss"]
    row = {
        "seed": seed,
        "dense_accuracy": dense_acc,
        "dense_ce": dense_loss,
        "magnitude_raw_accuracy": magnitude_raw_eval["test_accuracy"],
        "magnitude_raw_ce": magnitude_raw_eval["test_loss"],
        "geometry_raw_accuracy": geometry_raw_eval["test_accuracy"],
        "geometry_raw_ce": geometry_raw_eval["test_loss"],
        "magnitude_recovery20_accuracy": magnitude_rec_eval["test_accuracy"],
        "magnitude_recovery20_ce": magnitude_rec_eval["test_loss"],
        "geometry_recovery20_accuracy": geometry_rec_eval["test_accuracy"],
        "geometry_recovery20_ce": geometry_rec_eval["test_loss"],
        "magnitude_raw_drop_pp": 100 * (magnitude_raw_eval["test_accuracy"] - dense_acc),
        "geometry_raw_drop_pp": 100 * (geometry_raw_eval["test_accuracy"] - dense_acc),
        "magnitude_recovery20_drop_pp": 100 * (magnitude_rec_eval["test_accuracy"] - dense_acc),
        "geometry_recovery20_drop_pp": 100 * (geometry_rec_eval["test_accuracy"] - dense_acc),
        "magnitude_raw_ce_increase": magnitude_raw_eval["test_loss"] - dense_loss,
        "geometry_raw_ce_increase": geometry_raw_eval["test_loss"] - dense_loss,
        "magnitude_recovery20_ce_increase": magnitude_rec_eval["test_loss"] - dense_loss,
        "geometry_recovery20_ce_increase": geometry_rec_eval["test_loss"] - dense_loss,
        "geometry_minus_magnitude_raw_pp": 100 * (geometry_raw_eval["test_accuracy"] - magnitude_raw_eval["test_accuracy"]),
        "geometry_minus_magnitude_recovery20_pp": 100 * (geometry_rec_eval["test_accuracy"] - magnitude_rec_eval["test_accuracy"]),
        "geometry_minus_magnitude_raw_ce": geometry_raw_eval["test_loss"] - magnitude_raw_eval["test_loss"],
        "geometry_minus_magnitude_recovery20_ce": geometry_rec_eval["test_loss"] - magnitude_rec_eval["test_loss"],
        "mask_flip_rate": total_flips / total_positions,
        "mask_flip_count": total_flips,
        "magnitude_deleted_geometry_score_mean": magnitude_proxy / magnitude_proxy_count,
        "geometry_deleted_geometry_score_mean": geometry_proxy / geometry_proxy_count,
        "geometry_proxy_cost_ratio": geometry_proxy / magnitude_proxy,
        "recovery_order_hashes_match": True,
        "magnitude_raw_checkpoint_sha256": magnitude_status["raw_hard_sha256"],
        "magnitude_recovery_checkpoint_sha256": magnitude_status["recovery_final_sha256"],
        "magnitude_total_elapsed_seconds": magnitude_status["total_elapsed_seconds"],
        "magnitude_recovery_elapsed_seconds": magnitude_status["recovery_elapsed_seconds"],
        "magnitude_peak_allocated_bytes": magnitude_status["peak_allocated_bytes"],
        "magnitude_peak_reserved_bytes": magnitude_status["peak_reserved_bytes"],
    }
    return row, layer_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    for seed in range(3):
        parser.add_argument(f"--seed{seed}-formal-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.ablation_root.resolve()
    queue_status = load_json(output / "queue_status.json")
    if queue_status.get("status") != "completed" or queue_status.get("completed_seeds") != [0, 1, 2]:
        raise ValueError("Three-seed queue is not complete")
    rows, layer_rows = [], []
    for seed in range(3):
        row, layers = audit_seed(seed, getattr(args, f"seed{seed}_formal_root"), output)
        rows.append(row)
        layer_rows.extend(layers)
    paired = {
        "raw_accuracy_pp": summarize([row["geometry_minus_magnitude_raw_pp"] for row in rows]),
        "recovery20_accuracy_pp": summarize([row["geometry_minus_magnitude_recovery20_pp"] for row in rows]),
        "raw_ce": summarize([row["geometry_minus_magnitude_raw_ce"] for row in rows]),
        "recovery20_ce": summarize([row["geometry_minus_magnitude_recovery20_ce"] for row in rows]),
        "mask_flip_rate": summarize([row["mask_flip_rate"] for row in rows]),
        "geometry_proxy_cost_ratio": summarize([row["geometry_proxy_cost_ratio"] for row in rows]),
    }
    gpu_line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    environment = {
        "execution_gpu": gpu_line,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "formal_geometry_hardware": {
            "seed0": "NVIDIA GeForce RTX 4080 SUPER",
            "seed1": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "seed2": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        },
        "new_magnitude_hardware": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    }
    write_csv(output / "metrics_per_seed.csv", rows)
    paired_rows = [
        {
            "seed": row["seed"],
            "geometry_minus_magnitude_raw_pp": row["geometry_minus_magnitude_raw_pp"],
            "geometry_minus_magnitude_recovery20_pp": row["geometry_minus_magnitude_recovery20_pp"],
            "geometry_minus_magnitude_raw_ce": row["geometry_minus_magnitude_raw_ce"],
            "geometry_minus_magnitude_recovery20_ce": row["geometry_minus_magnitude_recovery20_ce"],
        }
        for row in rows
    ]
    write_csv(output / "paired_differences.csv", paired_rows)
    write_csv(output / "per_layer_mask_audit.csv", layer_rows)
    evidence = {
        "status": "PASS",
        "ablation_id": ABLATION_ID,
        "seeds_complete": [0, 1, 2],
        "metrics_per_seed": rows,
        "paired_summary": paired,
        "environment": environment,
        "randomness_control": {
            "train_order_hashes_match_all_20_epochs_all_seeds": True,
            "augmentation_worker_seed_is_explicit_function_of_seed_and_epoch": True,
            "limitation": "No serialized per-sample crop/flip trace; equivalence follows from the deterministic worker seeding implementation and matched software path.",
        },
        "interpretation_guardrail": "n=3 paired seeds is not a statistical-significance proof.",
    }
    atomic_json(output / "ablation_evidence.json", evidence)

    raw_mean = paired["raw_accuracy_pp"]["mean"]
    rec_mean = paired["recovery20_accuracy_pp"]["mean"]
    if raw_mean > 0 and rec_mean > 0:
        case = "A"
    elif raw_mean > 0 and abs(rec_mean) < 0.25:
        case = "B"
    elif raw_mean < 0 and abs(rec_mean) < 0.25:
        case = "C"
    else:
        case = "D"
    lines = [
        "# APBR-SAM Same-Parent Ablation: Magnitude vs Geometry",
        "",
        f"Audit status: **PASS**. Interpretation case: **{case}**.",
        "",
        "Both branches use the same APBR-SAM Epoch-200 parent per seed, identical "
        "eligible layers and OIHW 2:4 grouping, and the same deterministic Recovery-20 schedule.",
        "",
        "## Results",
        "",
        "| Seed | Dense | Mag Raw | Geo Raw | Δ Raw (Geo−Mag) | Mag Rec-20 | Geo Rec-20 | Δ Rec (Geo−Mag) | Flip rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {100*row['dense_accuracy']:.2f}% | "
            f"{100*row['magnitude_raw_accuracy']:.2f}% | {100*row['geometry_raw_accuracy']:.2f}% | "
            f"{row['geometry_minus_magnitude_raw_pp']:+.2f} pp | "
            f"{100*row['magnitude_recovery20_accuracy']:.2f}% | {100*row['geometry_recovery20_accuracy']:.2f}% | "
            f"{row['geometry_minus_magnitude_recovery20_pp']:+.2f} pp | {100*row['mask_flip_rate']:.2f}% |"
        )
    lines += [
        "",
        f"Raw paired difference: **{raw_mean:+.2f} ± {paired['raw_accuracy_pp']['sample_std']:.2f} pp** (sample SD).",
        f"Recovery-20 paired difference: **{rec_mean:+.2f} ± {paired['recovery20_accuracy_pp']['sample_std']:.2f} pp** (sample SD).",
        f"Raw CE difference (Geometry−Magnitude): **{paired['raw_ce']['mean']:+.4f} ± {paired['raw_ce']['sample_std']:.4f}**.",
        f"Recovery CE difference: **{paired['recovery20_ce']['mean']:+.4f} ± {paired['recovery20_ce']['sample_std']:.4f}**.",
        f"Mask flip rate: **{100*paired['mask_flip_rate']['mean']:.2f}% ± {100*paired['mask_flip_rate']['sample_std']:.2f} pp**.",
        "",
        "## Compute cost",
        "",
        "| Seed | Magnitude total | Recovery-20 | Peak allocated | Peak reserved |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['magnitude_total_elapsed_seconds']:.2f} s | "
            f"{row['magnitude_recovery_elapsed_seconds']:.2f} s | "
            f"{row['magnitude_peak_allocated_bytes']/2**30:.3f} GiB | "
            f"{row['magnitude_peak_reserved_bytes']/2**30:.3f} GiB |"
        )
    lines += [
        "",
        "## Mechanism and limits",
        "",
        "The frozen geometry proxy cost is reported as the ratio of summed deleted "
        "`|w| exp(-s)` under Geometry versus Magnitude. This is a deterministic "
        "mechanism diagnostic and partly tautological because Geometry directly "
        "optimizes that ranking; it is not independent evidence of test accuracy.",
        "",
        "All 20 recovery train-order hashes match per seed. Crop/flip equivalence "
        "follows from the shared deterministic DataLoader generator and worker-seed "
        "implementation, although individual augmentation decisions were not serialized.",
        "",
        "Seed 0 Geometry Recovery was originally executed on an RTX 4080 SUPER, "
        "whereas the new Magnitude branch used the RTX PRO 6000 Blackwell. Seeds 1 "
        "and 2 used Blackwell for both branches. The frozen software/data schedule "
        "is matched, but Seed 0's near-zero Recovery difference should not be treated "
        "as hardware-independent numerical equivalence.",
        "",
        "Three paired seeds quantify repeatability but do not establish statistical significance.",
    ]
    atomic_text(output / "ABLATION_REPORT.md", "\n".join(lines) + "\n")

    artifacts = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "RUN_MANIFEST.json")
    manifest = {
        "ablation_id": ABLATION_ID,
        "status": "PASS",
        "files": [{"path": str(path.relative_to(output)), "sha256": sha(path), "bytes": path.stat().st_size} for path in artifacts],
    }
    atomic_json(output / "RUN_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", "case": case, "paired_summary": paired}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
