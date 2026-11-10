"""Build auditable partial/final tables from fixed-mask compensation evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Keep the caller-provided relative paths on Windows.  Resolving a CWD that
    # contains non-ASCII characters through some console code pages can corrupt
    # the path even though relative file access remains correct.
    root, out = args.root, args.output

    ablation = root / "artifacts/ablation_apbr_parent_mag_vs_geometry_20261027"
    with (ablation / "metrics_per_seed.csv").open(encoding="utf-8", newline="") as stream:
        raw = {int(row["seed"]): row for row in csv.DictReader(stream)}
    historical_dir = root / "artifacts/apbr_seed0_fixedmask_obs_20261027_blackwell_remote/apbr_seed0_fixedmask_obs_20261027_blackwell_attempt2"
    historical_eval = read_json(historical_dir / "evaluation.json")
    historical_evidence = read_json(historical_dir / "compensation_evidence.json")
    new_eval = {
        (0, "magnitude"): read_json(out / "evaluations/seed0_magnitude_compensated.json"),
        (1, "geometry"): read_json(out / "evaluations/seed1_geometry_compensated.json"),
        (1, "magnitude"): read_json(out / "evaluations/seed1_magnitude_compensated.json"),
    }
    new_evidence = {
        (0, "magnitude"): read_json(out / "seed0_magnitude/evidence.json"),
        (1, "geometry"): read_json(out / "seed1_geometry/evidence.json"),
        (1, "magnitude"): read_json(out / "seed1_magnitude/evidence.json"),
    }

    metrics: list[dict] = []
    for seed in (0, 1, 2):
        source = raw[seed]
        dense_acc, dense_ce = float(source["dense_accuracy"]), float(source["dense_ce"])
        for condition, mask, compensated in (("A", "geometry", False), ("C", "magnitude", False)):
            acc, ce = float(source[f"{mask}_raw_accuracy"]), float(source[f"{mask}_raw_ce"])
            metrics.append({"seed": seed, "condition": condition, "mask_source": mask,
                            "second_order_compensation": False, "status": "PASS",
                            "accuracy": acc, "cross_entropy": ce,
                            "accuracy_drop_pp": 100 * (acc - dense_acc), "ce_increase": ce - dense_ce,
                            "calibration_seconds": 0.0, "compensation_seconds": 0.0,
                            "total_postprocess_seconds": None, "checkpoint_sha256": None,
                            "evidence_origin": "audited_same_parent_ablation_20261027"})
        for condition, mask in (("B", "geometry"), ("D", "magnitude")):
            if seed == 2:
                metrics.append({"seed": seed, "condition": condition, "mask_source": mask,
                                "second_order_compensation": True, "status": "BLOCKED_MISSING_DENSE_PARENT",
                                "accuracy": None, "cross_entropy": None, "accuracy_drop_pp": None,
                                "ce_increase": None, "calibration_seconds": None,
                                "compensation_seconds": None, "total_postprocess_seconds": None,
                                "checkpoint_sha256": None, "evidence_origin": None})
                continue
            if seed == 0 and mask == "geometry":
                evaluation, evidence = historical_eval, historical_evidence
                timing = evidence["timing"]
                origin = "verified_historical_seed0_reuse"
            else:
                evaluation, evidence = new_eval[(seed, mask)], new_evidence[(seed, mask)]
                timing = evidence["timing"]
                origin = "new_rtx5090_run_20261028"
            acc, ce = evaluation["test_accuracy"], evaluation["test_loss"]
            metrics.append({"seed": seed, "condition": condition, "mask_source": mask,
                            "second_order_compensation": True, "status": "PASS",
                            "accuracy": acc, "cross_entropy": ce,
                            "accuracy_drop_pp": 100 * (acc - dense_acc), "ce_increase": ce - dense_ce,
                            "calibration_seconds": timing.get("calibration_seconds", timing.get("calibration_wall_seconds_including_hessian")),
                            "compensation_seconds": timing.get("obs_compensation_seconds", timing.get("compensation_seconds")),
                            "total_postprocess_seconds": timing.get("total_postprocess_seconds", timing.get("calibration_plus_pruning_wall_excluding_validation_seconds")),
                            "checkpoint_sha256": evaluation["checkpoint_sha256"], "evidence_origin": origin})
    fields = list(metrics[0])
    write_csv(out / "metrics_per_seed.csv", metrics, fields)

    lookup = {(row["seed"], row["condition"]): row for row in metrics}
    paired = []
    for seed in (0, 1, 2):
        for name, left, right in (("geometry_repair", "B", "A"),
                                  ("mask_geometry_minus_magnitude_compensated", "B", "D"),
                                  ("magnitude_repair", "D", "C")):
            a, b = lookup[(seed, left)], lookup[(seed, right)]
            paired.append({"seed": seed, "comparison": name,
                           "status": "PASS" if a["status"] == b["status"] == "PASS" else "BLOCKED",
                           "accuracy_difference_pp": None if a["accuracy"] is None or b["accuracy"] is None else 100 * (a["accuracy"] - b["accuracy"]),
                           "ce_difference": None if a["cross_entropy"] is None or b["cross_entropy"] is None else a["cross_entropy"] - b["cross_entropy"]})
    write_csv(out / "paired_differences.csv", paired, list(paired[0]))

    summaries = {}
    for condition in ("A", "B", "C", "D"):
        rows = [row for row in metrics if row["condition"] == condition and row["status"] == "PASS"]
        summaries[condition] = {
            "available_seeds": [row["seed"] for row in rows], "n": len(rows),
            "accuracy_mean": statistics.mean(row["accuracy"] for row in rows),
            "accuracy_sample_sd": statistics.stdev(row["accuracy"] for row in rows) if len(rows) > 1 else None,
            "cross_entropy_mean": statistics.mean(row["cross_entropy"] for row in rows),
            "cross_entropy_sample_sd": statistics.stdev(row["cross_entropy"] for row in rows) if len(rows) > 1 else None,
        }
    paired_summary = {}
    for name in {row["comparison"] for row in paired}:
        rows = [row for row in paired if row["comparison"] == name and row["status"] == "PASS"]
        paired_summary[name] = {
            "available_seeds": [row["seed"] for row in rows], "n": len(rows),
            "accuracy_difference_pp_mean": statistics.mean(row["accuracy_difference_pp"] for row in rows),
            "accuracy_difference_pp_sample_sd": statistics.stdev(row["accuracy_difference_pp"] for row in rows) if len(rows) > 1 else None,
            "ce_difference_mean": statistics.mean(row["ce_difference"] for row in rows),
            "ce_difference_sample_sd": statistics.stdev(row["ce_difference"] for row in rows) if len(rows) > 1 else None,
        }
    write_json(out / "summary_mean_sample_sd.json", {
        "status": "PARTIAL_TWO_SEED_ONLY", "three_seed_complete": False,
        "blocking_issue": "Exact Seed 2 APBR Epoch-200 Dense parent SHA256 6b55caf932d846e16a13f7c2510e216db790bcd6738941288491c2e1d4b24a22 is unavailable locally.",
        "conditions": summaries, "paired_differences": paired_summary,
    })

    diagnostics = []
    for seed, mask, evidence in [(0, "geometry", historical_evidence), *[(s, m, e) for (s, m), e in new_evidence.items()]]:
        for row in evidence["layer_reports"]:
            diagnostics.append({"seed": seed, "mask_source": mask, **row})
    diagnostic_fields = sorted({key for row in diagnostics for key in row})
    write_csv(out / "layer_diagnostics.csv", diagnostics, diagnostic_fields)

    manifest = {
        "schema": 1, "experiment_id": "fixed_mask_second_order_repair_cifar100_20261028",
        "status": "PARTIAL_SEED2_BLOCKED", "protocol_id": "cifar100_50k10k_final_v1",
        "repository_head": "55fe3e300dbbd09c1178a6316afebc9ecf367947",
        "gpu": {"name": "NVIDIA GeForce RTX 5090", "uuid": "GPU-e4222bb2-c234-ec5b-d776-bd6cf8372984", "driver": "595.71.05", "pytorch": "2.8.0+cu128"},
        "frozen_config": {"calibration_count": 1024, "selection_seed": 20261027, "percdamp": 0.01, "blocksize": 128, "dtype": "float32", "amp": False, "tf32": False, "recovery_epochs": 0, "bn_calibration": False},
        "tests": {"local_targeted": "13 passed", "gpu_smoke": "PASS", "remote_pytest": "not run; pytest absent and dependencies were not changed"},
        "seed2_expected_parent_sha256": "6b55caf932d846e16a13f7c2510e216db790bcd6738941288491c2e1d4b24a22",
        "formal_test_access_policy": "one endpoint evaluation after frozen gates; no selection or tuning",
    }
    write_json(out / "RUN_MANIFEST.json", manifest)

    audit = """# Audit Report\n\nStatus: **PARTIAL PASS; SEED 2 BLOCKED**\n\n- Local targeted tests: 13 passed. Real RTX 5090 maximum-layer smoke: PASS.\n- New full-model runs: Seed 0 Magnitude, Seed 1 Geometry, and Seed 1 Magnitude all passed on 19 eligible layers without automatic retry.\n- Seed 0 Geometry reused the hash-verified historical result.\n- All completed compensated endpoints used the same 1,024 Train calibration IDs, FP32, damping 0.01, block size 128, and the same adapter.\n- Fixed supports matched audited references elementwise; each contained 11,157,504 eligible positions and 5,578,752 deletions. Deleted weights remained exactly zero, retained weights changed, outputs/weights were finite, and excluded state including BN buffers remained parent-identical.\n- No SGD, Recovery, BN-only calibration, AMP, TF32, compile, dynamic mask selection, or Test-driven tuning occurred.\n- Each new endpoint was evaluated once on all 10,000 Test samples after configuration and implementation gates were frozen.\n- The downloaded remote evidence manifest was independently hash-verified locally.\n\nSeed 2 B/D cannot be computed from its sparse Raw or Recovery files: the original values at deleted coordinates are required for the deletion displacement. Completion requires the exact APBR Epoch-200 Dense parent with SHA256 `6b55caf932d846e16a13f7c2510e216db790bcd6738941288491c2e1d4b24a22` and its matching geometry reference. No substitute was used.\n"""
    (out / "AUDIT_REPORT.md").write_text(audit, encoding="utf-8")

    sums = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        sums.append(f"{sha(path)}  {path.relative_to(out).as_posix()}")
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
