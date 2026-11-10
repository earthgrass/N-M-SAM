"""Read-only numerical audit of the completed HC-SAM Seed 0 artifacts.

This script never imports a dataset loader and never executes model inference.
It reads frozen JSON/checkpoints and writes derived diagnostics beside itself.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "artifacts" / "hq_apbr_seed0_full_20261105_remote_copy"
RUN = ROOT / "seed0_full_override_v1"
EVAL = RUN / "evaluation"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else None


def spearman(x, y):
    return pearson(average_ranks(np.asarray(x).reshape(-1)), average_ranks(np.asarray(y).reshape(-1)))


def canonical_magnitude_mask(weight: torch.Tensor) -> torch.Tensor:
    matrix = weight.reshape(weight.shape[0], -1)
    end = matrix.shape[1] // 4 * 4
    keep = torch.ones_like(matrix, dtype=torch.bool)
    quartets = matrix[:, :end].abs().reshape(matrix.shape[0], -1, 4)
    removed = quartets.argsort(dim=-1, stable=True)[..., :2]
    block_keep = torch.ones_like(quartets, dtype=torch.bool)
    block_keep.scatter_(-1, removed, False)
    keep[:, :end] = block_keep.reshape(matrix.shape[0], end)
    return keep.reshape_as(weight)


def combine_stats(items, count_key):
    total = sum(row[count_key] for row in items)
    mean = sum(row[count_key] * row["mean"] for row in items) / total
    second = sum(row[count_key] * (row["std"] ** 2 + row["mean"] ** 2) for row in items) / total
    return {
        "min": min(row["min"] for row in items),
        "max": max(row["max"] for row in items),
        "mean": mean,
        "std": math.sqrt(max(0.0, second - mean * mean)),
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    transfer = read_json(ROOT / "LOCAL_SHA256_VERIFICATION.json")
    local_audit = read_json(ROOT / "LOCAL_FINAL_AUDIT.json")
    assert transfer["status"] == "LOCAL_TRANSFER_SHA256_PASS"
    assert local_audit["status"] == "COMPLETED_SEED0_LOCAL_AUDIT_PASS"

    config = read_json(RUN / "FINAL_TRAINING_CONFIG.json")
    freeze = read_json(EVAL / "all_endpoints_frozen.json")
    history = read_json(RUN / "training" / "history.json")
    report = read_json(RUN / "SEED0_EVIDENCE_REPORT.json")
    comparison = read_json(ROOT / "seed0_full_override_v1_historical_comparison" / "comparison.json")
    layer_logs = {}
    for line in (EVAL / "pruning_layers.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        layer_logs[row["layer"] + ".weight"] = row
    assert len(layer_logs) == 19

    parent = torch.load(RUN / "training" / "final_epoch_199.pt", map_location="cpu", weights_only=False)
    dense_ck = torch.load(EVAL / "dense.pt", map_location="cpu", weights_only=False)
    static_ck = torch.load(EVAL / "final_q_mask_only.pt", map_location="cpu", weights_only=False)
    fixed_ck = torch.load(EVAL / "final_q_fixed_obs.pt", map_location="cpu", weights_only=False)
    native_ck = torch.load(EVAL / "sparsegpt_native.pt", map_location="cpu", weights_only=False)
    eligible = sorted(static_ck["masks"])
    assert eligible == sorted(fixed_ck["masks"]) == sorted(native_ck["masks"])
    assert all(torch.equal(static_ck["masks"][k], fixed_ck["masks"][k]) for k in eligible)

    def bn_state(model):
        return {k: v for k, v in model.items() if ("bn" in k or ".shortcut.1." in k)}

    calibration_hashes = {row["calibration"]["calibration_ids_sha256"] for row in layer_logs.values()}
    assert len(calibration_hashes) == 1
    calibration_hash = next(iter(calibration_hashes))
    mapping_payload = "\n".join(f"{k}:{tuple(dense_ck['model'][k].shape)}" for k in eligible).encode()
    mapping_hash = hashlib.sha256(mapping_payload).hexdigest()
    endpoint_rows = []
    endpoint_checkpoints = {
        "dense": dense_ck,
        "final_q_mask_only": static_ck,
        "final_q_fixed_obs": fixed_ck,
        "sparsegpt_native": native_ck,
    }
    for name, checkpoint in endpoint_checkpoints.items():
        metrics = read_json(EVAL / f"{name}_evaluation.json")
        endpoint_rows.append({
            "endpoint": name,
            "accuracy_percent": metrics["accuracy_percent"],
            "ce": metrics["ce"],
            "dense_parent_sha256": checkpoint["parent_sha256"],
            "checkpoint_sha256": freeze["endpoints"][name]["checkpoint_sha256"],
            "mask_sha256": checkpoint["mask_sha256"],
            "calibration_ids_sha256": calibration_hash,
            "dataset_archive_sha256": freeze["dataset"]["archive_sha256"],
            "config_sha256": checkpoint["config_sha256"],
            "percdamp": config["percdamp"],
            "blocksize": config["blocksize"],
            "compute_dtype": config["hessian_dtype"],
            "checkpoint_epoch_index": checkpoint["epoch_index"],
            "completed_epochs": checkpoint["completed_epochs"],
            "eligible_layer_mapping_sha256": mapping_hash,
            "bn_state_sha256": tensor_hash(bn_state(checkpoint["model"])),
            "recovery_epochs": checkpoint["recovery_epochs"],
        })
    write_csv(HERE / "mask_compensation_endpoint_comparison.csv", endpoint_rows)

    # Per-Epoch Q summary, drift, CE, and layer-energy concentration.
    evolution = []
    energy_by_layer = {key: [] for key in eligible}
    for epoch, hist in enumerate(history):
        cal = read_json(RUN / "training" / f"calibration_epoch_{epoch:03d}.json")
        q_items = []
        for key in eligible:
            stats = cal["q_diagonal_statistics"][key]
            shape = cal["layers"][key]["weight_shape"]
            groups = cal["layers"][key]["groups"]
            features = math.prod(shape[1:])
            q_items.append({**stats, "unique_count": groups * features, "coordinate_count": math.prod(shape)})
            energy_by_layer[key].append(hist["perturbation_energy_distribution"][key])
        unique = combine_stats(q_items, "unique_count")
        coordinate = combine_stats(q_items, "coordinate_count")
        eligible_energy = {k: hist["perturbation_energy_distribution"][k] for k in eligible}
        total_eligible_energy = sum(eligible_energy.values())
        ranked = sorted(eligible_energy.items(), key=lambda item: item[1], reverse=True)
        normalized = [value / total_eligible_energy for _, value in ranked]
        evolution.append({
            "epoch": epoch,
            "train_ce": hist["train_loss"],
            "adversarial_train_ce": hist["adversarial_train_loss"],
            "geometry_drift_rms": hist["log_geometry_drift"]["rms"],
            "geometry_drift_max_abs": hist["log_geometry_drift"]["max_abs"],
            "q_unique_min": unique["min"], "q_unique_max": unique["max"],
            "q_unique_mean": unique["mean"], "q_unique_std": unique["std"],
            "q_coordinate_weighted_mean": coordinate["mean"],
            "q_coordinate_weighted_std": coordinate["std"],
            "eligible_energy_share": total_eligible_energy,
            "noneligible_energy_share": hist["perturbation_energy_distribution"]["noneligible"],
            "top1_eligible_layer": ranked[0][0],
            "top1_share_of_total_energy": ranked[0][1],
            "top1_share_within_eligible": normalized[0],
            "top3_share_of_total_energy": sum(v for _, v in ranked[:3]),
            "top3_share_within_eligible": sum(normalized[:3]),
            "eligible_energy_hhi": sum(v * v for v in normalized),
        })
    write_csv(HERE / "geometry_evolution_summary.csv", evolution)

    snapshot_rows = []
    for epoch in (0, 9, 49, 99, 149, 199):
        ck_path = RUN / "training" / ("final_epoch_199.pt" if epoch == 199 else f"epoch_{epoch:03d}.pt")
        ck = parent if epoch == 199 else torch.load(ck_path, map_location="cpu", weights_only=False)
        tensors = [ck["stepper"]["geometry"][key].double().reshape(-1) for key in eligible]
        count = sum(t.numel() for t in tensors)
        total = sum(float(t.sum()) for t in tensors)
        square = sum(float(t.square().sum()) for t in tensors)
        snapshot_rows.append({
            "epoch_index_after_refresh": epoch,
            "coordinates": count,
            "s_mean": total / count,
            "s_std_population": math.sqrt(max(0.0, square / count - (total / count) ** 2)),
            "s_min": min(float(t.min()) for t in tensors),
            "s_max": max(float(t.max()) for t in tensors),
            "checkpoint_sha256": file_sha(ck_path),
        })
        if epoch != 199:
            del ck
    write_csv(HERE / "geometry_checkpoint_snapshots.csv", snapshot_rows)

    per_layer = []
    final_q = parent["q_diags"]
    final_s = parent["stepper"]["geometry"]
    total_mask_diff = 0
    total_coords = 0
    total_groups = 0
    identical_groups = 0
    magnitude_native_diff = 0
    magnitude_native_identical_groups = 0
    static_magnitude_diff = 0
    static_magnitude_identical_groups = 0
    top_q_deleted = top_q_count = bottom_q_deleted = bottom_q_count = 0
    deleted_abs_sum = deleted_abs_count = kept_abs_sum = kept_abs_count = 0
    for key in eligible:
        dense_w = dense_ck["model"][key].double()
        static_mask = static_ck["masks"][key]
        fixed_mask = fixed_ck["masks"][key]
        native_mask = native_ck["masks"][key]
        magnitude_mask = canonical_magnitude_mask(dense_ck["model"][key])
        assert torch.equal(static_mask, fixed_mask)
        matrix_native = native_mask.reshape(native_mask.shape[0], -1)
        matrix_static = static_mask.reshape(static_mask.shape[0], -1)
        matrix_mag = magnitude_mask.reshape(magnitude_mask.shape[0], -1)
        diff = int((matrix_static != matrix_native).sum())
        mag_diff = int((matrix_mag != matrix_native).sum())
        groups = matrix_native.shape[0] * (matrix_native.shape[1] // 4)
        same_groups = int((matrix_static.reshape(-1, 4) == matrix_native.reshape(-1, 4)).all(1).sum())
        mag_same_groups = int((matrix_mag.reshape(-1, 4) == matrix_native.reshape(-1, 4)).all(1).sum())
        total_mask_diff += diff
        magnitude_native_diff += mag_diff
        magnitude_native_identical_groups += mag_same_groups
        static_magnitude_diff += int((matrix_static != matrix_mag).sum())
        static_magnitude_identical_groups += int((matrix_static.reshape(-1, 4) == matrix_mag.reshape(-1, 4)).all(1).sum())
        total_coords += matrix_native.numel()
        total_groups += groups
        identical_groups += same_groups

        q = final_q[key].double().reshape(-1)
        s_matrix = final_s[key].double().reshape(final_s[key].shape[0], -1)
        s_feature = s_matrix.mean(0)
        q_delete_rate = (~matrix_native).double().mean(0).numpy()
        q_np = q.numpy()
        s_np = s_feature.numpy()
        low = np.quantile(q_np, 0.1)
        high = np.quantile(q_np, 0.9)
        q_broadcast = q.reshape(1, -1).expand_as(matrix_native)
        deleted = ~matrix_native
        abs_w = dense_w.reshape_as(matrix_native).abs()
        high_mask = q >= high
        low_mask = q <= low
        top_q_deleted += int(deleted[:, high_mask].sum())
        top_q_count += int(deleted[:, high_mask].numel())
        bottom_q_deleted += int(deleted[:, low_mask].sum())
        bottom_q_count += int(deleted[:, low_mask].numel())
        deleted_abs_sum += float(abs_w[deleted].sum())
        deleted_abs_count += int(deleted.sum())
        kept_abs_sum += float(abs_w[~deleted].sum())
        kept_abs_count += int((~deleted).sum())
        fixed_delta = fixed_ck["model"][key].double() - dense_w
        native_delta = native_ck["model"][key].double() - dense_w
        log = layer_logs[key]
        diag = parent["calibration_report"]["layers"][key]["groups_diagnostics"][0]
        energy = np.asarray(energy_by_layer[key], dtype=np.float64)
        per_layer.append({
            "layer": key,
            "shape": "x".join(map(str, dense_w.shape)),
            "coordinates": matrix_native.numel(),
            "deletions": int(deleted.sum()),
            "static_native_mask_difference_count": diff,
            "static_native_coordinate_agreement": 1 - diff / matrix_native.numel(),
            "static_native_identical_group_rate": same_groups / groups,
            "magnitude_native_coordinate_agreement": 1 - mag_diff / matrix_native.numel(),
            "magnitude_native_identical_group_rate": mag_same_groups / groups,
            "static_magnitude_coordinate_agreement": float((matrix_static == matrix_mag).double().mean()),
            "mask_only_reconstruction_mse": log["layer_reconstruction_mse"]["final_q_mask_only"],
            "fixed_obs_reconstruction_mse": log["layer_reconstruction_mse"]["final_q_fixed_obs"],
            "native_reconstruction_mse": log["layer_reconstruction_mse"]["sparsegpt_native"],
            "fixed_obs_survivor_delta_rms": float(fixed_delta[fixed_mask].square().mean().sqrt()),
            "native_survivor_delta_rms": float(native_delta[native_mask].square().mean().sqrt()),
            "fixed_obs_survivor_changed_count": log["fixed_obs"]["survivor_changed_count"],
            "fixed_obs_compensation_seconds": log["fixed_obs"]["compensation_seconds"],
            "native_compensation_seconds": log["sparsegpt_native"]["compensation_seconds"],
            "dead_input_features": log["dead_input_features"],
            "hq_absolute_damping": log["hq_absolute_damping"],
            "gram_damped_diagonal_ratio_proxy": diag["damped_diagonal_ratio_proxy"],
            "spectral_condition_number": "NOT AVAILABLE",
            "final_q_min": float(q.min()), "final_q_max": float(q.max()),
            "final_q_mean": float(q.mean()), "final_q_std": float(q.std(unbiased=False)),
            "final_s_min": float(s_matrix.min()), "final_s_max": float(s_matrix.max()),
            "final_s_mean": float(s_matrix.mean()), "final_s_std": float(s_matrix.std(unbiased=False)),
            "s_output_row_variation_max": float((s_matrix - s_feature).abs().max()),
            "spearman_logq_vs_final_s": spearman(np.log(q_np), s_np),
            "spearman_q_vs_native_delete_rate": spearman(q_np, q_delete_rate),
            "native_delete_rate_top_q_decile": float(deleted[:, q >= high].double().mean()),
            "native_delete_rate_bottom_q_decile": float(deleted[:, q <= low].double().mean()),
            "q_mean_deleted": float(q_broadcast[deleted].mean()),
            "q_mean_kept": float(q_broadcast[~deleted].mean()),
            "dense_abs_weight_mean_deleted": float(abs_w[deleted].mean()),
            "dense_abs_weight_mean_kept": float(abs_w[~deleted].mean()),
            "perturbation_energy_mean_over_epochs": float(energy.mean()),
            "perturbation_energy_max_over_epochs": float(energy.max()),
            "perturbation_energy_final_epoch": float(energy[-1]),
        })
    write_csv(HERE / "per_layer_diagnostics.csv", per_layer)

    drift = np.asarray([row["geometry_drift_rms"] for row in evolution])
    ce = np.asarray([row["train_ce"] for row in evolution])
    adv_ce = np.asarray([row["adversarial_train_ce"] for row in evolution])
    q_vs_s = [row["spearman_logq_vs_final_s"] for row in per_layer]
    native_mse = sorted(per_layer, key=lambda row: row["native_reconstruction_mse"], reverse=True)
    summary = {
        "status": "READ_ONLY_ROOT_CAUSE_AUDIT_COMPLETE",
        "test_repeated": False,
        "dataset_opened": False,
        "training_started": False,
        "transfer": transfer,
        "formal_local_audit_status": local_audit["status"],
        "parent_sha256": report["final_checkpoint_sha256"],
        "config_sha256": freeze["config_sha256"],
        "calibration_ids_sha256": calibration_hash,
        "eligible_layer_mapping_sha256": mapping_hash,
        "mask_identity": {
            "mask_only_equals_fixed_obs": True,
            "mask_only_hash": static_ck["mask_sha256"],
            "fixed_obs_hash": fixed_ck["mask_sha256"],
            "native_hash": native_ck["mask_sha256"],
            "static_native_coordinate_agreement": 1 - total_mask_diff / total_coords,
            "static_native_identical_group_rate": identical_groups / total_groups,
            "static_native_difference_count": total_mask_diff,
            "magnitude_native_coordinate_agreement": 1 - magnitude_native_diff / total_coords,
            "magnitude_native_identical_group_rate": magnitude_native_identical_groups / total_groups,
            "static_magnitude_coordinate_agreement": 1 - static_magnitude_diff / total_coords,
            "static_magnitude_identical_group_rate": static_magnitude_identical_groups / total_groups,
            "static_native_keep_support_jaccard": (1 - total_mask_diff / total_coords) / (1 + total_mask_diff / total_coords),
        },
        "bn_state_hashes": {row["endpoint"]: row["bn_state_sha256"] for row in endpoint_rows},
        "geometry_history_availability": {
            "per_epoch_q_summary": "AVAILABLE",
            "per_epoch_geometry_drift": "AVAILABLE",
            "full_s_tensors": "AVAILABLE_AT_EPOCHS_0_9_49_99_149_199_ONLY",
            "ordinary_sam_direction_cosine": "NOT AVAILABLE",
            "per_coordinate_perturbation_energy": "NOT AVAILABLE",
            "per_layer_perturbation_energy": "AVAILABLE_EACH_EPOCH",
        },
        "geometry_drift_train_ce_correlation": {
            "pearson": pearson(drift, ce), "spearman": spearman(drift, ce),
            "adversarial_ce_pearson": pearson(drift, adv_ce),
            "interpretation": "association only; same-epoch post-training refresh versus epoch CE",
        },
        "final_q_history_s_relationship": {
            "per_layer_spearman_logq_vs_s_median": float(np.median(q_vs_s)),
            "per_layer_spearman_logq_vs_s_min": float(np.min(q_vs_s)),
            "per_layer_spearman_logq_vs_s_max": float(np.max(q_vs_s)),
        },
        "native_mask_score_diagnostics": {
            "top_within_layer_q_decile_delete_rate": top_q_deleted / top_q_count,
            "bottom_within_layer_q_decile_delete_rate": bottom_q_deleted / bottom_q_count,
            "dense_abs_weight_mean_deleted": deleted_abs_sum / deleted_abs_count,
            "dense_abs_weight_mean_kept": kept_abs_sum / kept_abs_count,
            "interpretation": "Associations with the final native mask; native SparseGPT used evolving compensated weights, so these are not a causal score decomposition.",
        },
        "reconstruction_summary": {
            "sum_layer_mse_mask_only": sum(row["mask_only_reconstruction_mse"] for row in per_layer),
            "sum_layer_mse_fixed_obs": sum(row["fixed_obs_reconstruction_mse"] for row in per_layer),
            "sum_layer_mse_native": sum(row["native_reconstruction_mse"] for row in per_layer),
            "native_lower_than_fixed_obs_layer_count": sum(row["native_reconstruction_mse"] < row["fixed_obs_reconstruction_mse"] for row in per_layer),
            "layer_count": len(per_layer),
        },
        "compensability_alignment_A": {
            "status": "NOT AVAILABLE",
            "reason": "Only per-layer perturbation energy was logged; the requested deleted-coordinate energy fraction requires per-coordinate energy.",
            "random_2_4_expectation": 0.5,
        },
        "energy_concentration": {
            "mean_top1_share_of_total": float(np.mean([row["top1_share_of_total_energy"] for row in evolution])),
            "mean_top3_share_of_total": float(np.mean([row["top3_share_of_total_energy"] for row in evolution])),
            "final_top1_layer": evolution[-1]["top1_eligible_layer"],
            "final_top1_share_of_total": evolution[-1]["top1_share_of_total_energy"],
            "final_top3_share_of_total": evolution[-1]["top3_share_of_total_energy"],
        },
        "native_reconstruction_error_top5": [
            {"layer": row["layer"], "mse": row["native_reconstruction_mse"]} for row in native_mse[:5]
        ],
        "historical_reference": comparison,
        "classification": {
            "verified_implementation_issue": [],
            "observed_mechanism_behavior": [
                "Static Final-Q support and native SparseGPT support differ.",
                "Native SparseGPT has lower reconstruction MSE than static-mask fixed OBS on the frozen HC parent.",
                "Perturbation energy is concentrated in a small number of late layers by the final epoch.",
            ],
            "supported_hypothesis": [
                "The static Final-Q mask is less compatible with the frozen OBS reconstruction path than the native dynamic mask on this parent.",
                "HC-SAM Seed 0 did not improve native SparseGPT endpoint accuracy over the audited SAM/APBR references.",
            ],
            "inconclusive": [
                "Whether poor coordinate-level energy/mask alignment caused the negative native endpoint.",
                "Whether concentration in late layers causally reduced accuracy.",
                "Whether the result is stable across seeds.",
            ],
            "not_available": [
                "Per-coordinate cumulative perturbation energy and alignment A.",
                "Cosine between HC perturbations and ordinary SAM perturbations.",
                "Exact spectral Gram condition numbers.",
                "Same-protocol historical SAM/APBR per-layer reconstruction traces.",
            ],
        },
    }
    (HERE / "audit_evidence.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    claim_rows = [
        {"id": "C1", "claim": "Transfer and local evidence are complete.", "classification": "VERIFIED IMPLEMENTATION STATE", "evidence": "351/351 files; transfer and original manifests PASS", "boundary": "Not an efficacy claim"},
        {"id": "C2", "claim": "All endpoints share parent, Epoch, calibration IDs, config, mapping, and BN state.", "classification": "VERIFIED IMPLEMENTATION STATE", "evidence": "Endpoint checkpoint metadata and tensor hashes", "boundary": "Pruning algorithms differ"},
        {"id": "C3", "claim": "Mask-only and Fixed OBS use identical support.", "classification": "VERIFIED IMPLEMENTATION STATE", "evidence": "Bitwise Mask equality and identical SHA256", "boundary": "Only these two branches"},
        {"id": "C4", "claim": "The 26.66 pp Mask-only to Fixed OBS gain is a same-support compensation gain.", "classification": "SUPPORTED HYPOTHESIS", "evidence": "Same parent/support/BN/config; fixed OBS changes survivors", "boundary": "One parent and Seed"},
        {"id": "C5", "claim": "Native GPT's 1.87 pp advantage is purely compensation.", "classification": "REJECTED CLAIM", "evidence": "Native support differs; 79.23% coordinate agreement", "boundary": "Selection and dynamic path confounded"},
        {"id": "C6", "claim": "Native GPT has lower layer reconstruction MSE.", "classification": "OBSERVED MECHANISM BEHAVIOR", "evidence": "Lower than Fixed OBS in 19/19 layers", "boundary": "Layer sums are not end-to-end MSE"},
        {"id": "C7", "claim": "Historical s tracks Q ranking.", "classification": "VERIFIED MECHANISM BEHAVIOR", "evidence": "Per-layer Spearman median 0.99948", "boundary": "Expected from EMA; not usefulness"},
        {"id": "C8", "claim": "Perturbation energy concentrates in late layers.", "classification": "OBSERVED MECHANISM BEHAVIOR", "evidence": "Final top-one 36.88%; top-three 72.38%", "boundary": "Coordinate concentration unavailable"},
        {"id": "C9", "claim": "Energy/Mask alignment A is worse than random 0.5.", "classification": "NOT AVAILABLE", "evidence": "Per-coordinate energy was not logged", "boundary": "No A value computed"},
        {"id": "C10", "claim": "High Q coordinates are more often deleted by Native GPT.", "classification": "OBSERVED ASSOCIATION", "evidence": "Top Q decile 55.17%; bottom 44.93% deletion", "boundary": "Post-hoc; evolving weights"},
        {"id": "C11", "claim": "Dense magnitude relates to Native support.", "classification": "OBSERVED ASSOCIATION", "evidence": "Mean abs weight deleted 0.00409; retained 0.00852", "boundary": "Not a causal score decomposition"},
        {"id": "C12", "claim": "Geometry drift causes CE to fall.", "classification": "INCONCLUSIVE", "evidence": "High correlation with common Epoch trend", "boundary": "No causal intervention"},
        {"id": "C13", "claim": "A software defect explains the result.", "classification": "NOT SUPPORTED", "evidence": "All identity and numerical audits pass", "boundary": "No evidenced defect"},
        {"id": "C14", "claim": "HC-SAM improves Native GPT over references.", "classification": "NOT SUPPORTED", "evidence": "HC 76.59; SAM 77.47/77.51; APBR 76.99", "boundary": "Reference-only, one Seed"},
        {"id": "C15", "claim": "Expand unchanged HC-SAM immediately.", "classification": "NOT RECOMMENDED", "evidence": "No positive task signal; alignment A unavailable", "boundary": "A bounded redesign remains possible"},
    ]
    write_csv(HERE / "claim_to_evidence.csv", claim_rows)

    manifest_lines = []
    for path in sorted(HERE.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            manifest_lines.append(f"{file_sha(path)}  {path.name}")
    (HERE / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
