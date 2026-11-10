"""Verify copied Geo-OBS evidence and compute same-parent comparisons."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(directory: Path) -> dict:
    checked = 0
    for name in ("SHA256SUMS.txt", "SHA256SUMS_SUPPLEMENT.txt"):
        manifest = directory / name
        if not manifest.exists():
            if name == "SHA256SUMS_SUPPLEMENT.txt" and directory.name.startswith("geo_obs_fixed_l2"):
                continue
            raise FileNotFoundError(manifest)
        for line in manifest.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            target = directory / relative
            if not target.exists() or sha(target) != expected:
                raise ValueError(f"SHA256 mismatch: {target}")
            checked += 1
    return {"verified_files": checked, "manifest_sha256": sha(directory / "SHA256SUMS.txt")}


def read_results(directory: Path) -> dict:
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    frozen = json.loads((directory / "frozen_pretest.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETE" or frozen["status"] != "ALL_CHECKPOINTS_FROZEN":
        raise ValueError(f"Incomplete output: {directory}")
    results = {}
    for row in csv.DictReader((directory / "results.csv").open(newline="", encoding="utf-8")):
        method = row["method"]
        if int(row["test_samples"]) != 10000 or int(row["test_evaluations"]) != 1:
            raise ValueError("Invalid fixed Test evaluation contract")
        record = frozen["methods"][method]
        if row["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise ValueError("Test evaluated checkpoint differs from frozen checkpoint")
        checkpoint = directory / "methods" / method / "checkpoint.pt"
        if sha(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        results[method] = {"accuracy_percent": float(row["accuracy_percent"]),
                           "ce": float(row["ce"]),
                           "checkpoint_sha256": row["checkpoint_sha256"],
                           "mask_sha256": row["mask_sha256"],
                           "pruning_compensation_seconds": float(row["pruning_compensation_seconds"])}
    return results


def mask_disagreement(directory: Path, other_method: str) -> dict:
    baseline = torch.load(directory / "methods/dynamic_group_obs/masks.pt", map_location="cpu", weights_only=False)
    other = torch.load(directory / "methods" / other_method / "masks.pt", map_location="cpu", weights_only=False)
    changed = total = 0
    for name, mask in baseline.items():
        a = mask.reshape(mask.shape[0], -1, 4)
        b = other[name].reshape(mask.shape[0], -1, 4)
        changed += int((a != b).any(-1).sum())
        total += a.shape[0] * a.shape[1]
    return {"changed_groups": changed, "total_groups": total,
            "changed_percent": 100 * changed / total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--copy-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ("geo_obs_fixed_l2_seed0_v1", "geo_dynamic_fixed_l2_seed0_v1", "geo_dynamic_vpr_seed0_v1")
    verification = {name: verify_manifest(args.copy_root / name) for name in names}
    static = read_results(args.copy_root / names[0])
    l2 = read_results(args.copy_root / names[1])
    vpr = read_results(args.copy_root / names[2])
    l2_baseline = json.loads((args.copy_root.parent / "apbr_fixed_l2_seed0_20261102/remote_final_copy/local_verification.json").read_text())
    l2_ref = {x["endpoint"]: x for x in l2_baseline["endpoints"]}
    if not all(l2_ref[x]["verified"] for x in ("dense", "original_sparsegpt_native")):
        raise ValueError("Historical L2 reference not verified")
    original_root = args.copy_root.parent / "cifar100_50k10k_final_v1/seed0_remote_copy/test_final"
    vpr_dense = json.loads((original_root / "apbr_final.json").read_text())
    sam_native = json.loads((original_root / "sparsegpt_raw.json").read_text())
    vpr_native_path = args.copy_root.parent / "apbr_radius_guided_sparsegpt_seed012/evidence/original_controls/seed0/test_final.json"
    vpr_native = json.loads(vpr_native_path.read_text())
    reference = {
        "apbr_fixed_l2_dense": l2_ref["dense"],
        "apbr_fixed_l2_native_sparsegpt": l2_ref["original_sparsegpt_native"],
        "apbr_vpr_dense": vpr_dense,
        "apbr_vpr_native_sparsegpt": vpr_native,
        "sam_native_sparsegpt": sam_native,
    }
    agreement = {parent: {m: mask_disagreement(args.copy_root / folder, m)
                          for m in ("geo_select_std_comp", "geo_full", "shuffled_geo_select", "shuffled_geo_full")}
                 for parent, folder in (("fixed_l2", names[1]), ("vpr", names[2]))}
    comparison = {
        "l2_geo_select_minus_dynamic_group_pp": l2["geo_select_std_comp"]["accuracy_percent"] - l2["dynamic_group_obs"]["accuracy_percent"],
        "l2_geo_select_minus_shuffled_pp": l2["geo_select_std_comp"]["accuracy_percent"] - l2["shuffled_geo_select"]["accuracy_percent"],
        "vpr_geo_select_minus_dynamic_group_pp": vpr["geo_select_std_comp"]["accuracy_percent"] - vpr["dynamic_group_obs"]["accuracy_percent"],
        "vpr_geo_select_minus_shuffled_pp": vpr["geo_select_std_comp"]["accuracy_percent"] - vpr["shuffled_geo_select"]["accuracy_percent"],
        "l2_dynamic_group_minus_native_sparsegpt_pp": l2["dynamic_group_obs"]["accuracy_percent"] - l2_ref["original_sparsegpt_native"]["accuracy_percent"],
        "vpr_dynamic_group_minus_native_sparsegpt_pp": vpr["dynamic_group_obs"]["accuracy_percent"] - 100 * vpr_native["test_accuracy"],
        "l2_geo_select_minus_different_parent_sam_native_pp": l2["geo_select_std_comp"]["accuracy_percent"] - 100 * sam_native["test_accuracy"],
        "vpr_geo_select_minus_different_parent_sam_native_pp": vpr["geo_select_std_comp"]["accuracy_percent"] - 100 * sam_native["test_accuracy"],
        "vpr_geo_full_minus_different_parent_sam_native_pp": vpr["geo_full"]["accuracy_percent"] - 100 * sam_native["test_accuracy"],
    }
    result = {"status": "PASS", "seed": 0, "verified_transfer": verification,
              "references": reference, "static_l2": static, "dynamic_l2": l2,
              "dynamic_vpr": vpr, "mask_disagreement": agreement,
              "paired_differences": comparison,
              "interpretation": "Single-seed exploratory comparison; cross-parent values are not geometry-only causal estimates."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "verified_files": verification,
                      "paired_differences": comparison}, indent=2))


if __name__ == "__main__":
    main()
