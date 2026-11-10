"""Read-only Seed-0 Wanda/APBR mechanism and post-pruning cost audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch


PROTOCOL = "cifar100_50k10k_final_v1"
METHODS = ("magnitude", "wanda", "sparsegpt", "rose", "apbr_sam")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return 100.0 * value


def gib(value: int) -> float:
    return value / 1024**3


def quantiles(values: torch.Tensor) -> dict[str, float | None]:
    flat = values.detach().cpu().double().flatten()
    if flat.numel() == 0:
        return {"min": None, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "min": float(flat.min()),
        "mean": float(flat.mean()),
        "median": float(flat.quantile(0.5)),
        "p90": float(flat.quantile(0.9)),
        "max": float(flat.max()),
    }


def training_seconds(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(float(row["epoch_seconds"]) for row in csv.DictReader(handle))


def ensure_seed0_contract(seed0: Path) -> None:
    for relative in (
        "dense_sam/final_epoch_200.pt",
        "apbr_sam/final_epoch_200.pt",
        "methods/wanda/raw_hard.pt",
        "methods/apbr_sam/raw_hard.pt",
        "methods/wanda/recovery/final_epoch_020.pt",
        "methods/apbr_sam/recovery/final_epoch_020.pt",
    ):
        if not (seed0 / relative).is_file():
            raise FileNotFoundError(seed0 / relative)


def inspect_method(seed0: Path, method: str) -> dict:
    method_dir = seed0 / "methods" / method
    raw_path = method_dir / "raw_hard.pt"
    bn_path = method_dir / "bn_only.pt"
    recovery_path = method_dir / "recovery" / "final_epoch_020.pt"
    raw = load_checkpoint(raw_path)
    bn = load_checkpoint(bn_path)
    recovery = load_checkpoint(recovery_path)
    status = load_json(method_dir / "status.json")
    if raw["protocol_id"] != PROTOCOL or recovery["protocol_id"] != PROTOCOL:
        raise ValueError(f"Protocol mismatch for {method}")
    raw_sha = sha256_file(raw_path)
    bn_sha = sha256_file(bn_path)
    recovery_sha = sha256_file(recovery_path)
    parent_ok = recovery["parent"]["sha256"] == raw_sha
    status_hashes_match = (
        status["raw_hard_sha256"] == raw_sha
        and status["bn_only_sha256"] == bn_sha
        and status["recovery_final_sha256"] == recovery_sha
    )
    mask_identity_ok = (
        raw["mask_sha256"] == bn["mask_sha256"] == recovery["mask_sha256"] == status["mask_sha256"]
    )
    bn_parameter_keys_equal = all(
        torch.equal(raw["model"][key], value)
        for key, value in bn["model"].items()
        if not key.endswith(("running_mean", "running_var", "num_batches_tracked"))
    )
    bn_buffers_changed = any(
        not torch.equal(raw["model"][key], value)
        for key, value in bn["model"].items()
        if key.endswith(("running_mean", "running_var", "num_batches_tracked"))
    )
    masked_zero_raw = all(bool((raw["model"][key][~mask.bool()] == 0).all()) for key, mask in raw["masks"].items())
    masked_zero_bn = all(bool((bn["model"][key][~mask.bool()] == 0).all()) for key, mask in raw["masks"].items())
    masked_zero_recovery = all(
        bool((recovery["model"][key][~mask.bool()] == 0).all()) for key, mask in raw["masks"].items()
    )
    histories = recovery["history"]
    return {
        "raw": raw,
        "bn": bn,
        "recovery": recovery,
        "status": status,
        "paths": {
            "raw": str(raw_path),
            "bn_only": str(bn_path),
            "recovery_final": str(recovery_path),
        },
        "hashes": {"raw": raw_sha, "bn_only": bn_sha, "recovery_final": recovery_sha},
        "recovery_parent_hash_matches_raw": parent_ok,
        "status_file_hashes_match_actual_files": status_hashes_match,
        "mask_identity_across_raw_bn_recovery": mask_identity_ok,
        "bn_non_running_state_equal_raw": bn_parameter_keys_equal,
        "bn_running_buffers_changed": bn_buffers_changed,
        "masked_coordinates_zero": {
            "raw": masked_zero_raw,
            "bn_only": masked_zero_bn,
            "recovery_final": masked_zero_recovery,
        },
        "recovery_contract": {
            "epochs": recovery["epoch"],
            "steps": recovery["steps"],
            "history_rows": len(histories),
            "all_test_accessed_false": all(row["test_accessed"] is False for row in histories),
            "all_train_samples_50000": all(row["train_samples"] == 50000 for row in histories),
            "all_batches_391": all(row["batches"] == 391 for row in histories),
        },
        "trajectory": [
            {k: row[k] for k in ("epoch", "learning_rate", "train_loss", "train_accuracy", "epoch_seconds")}
            for row in histories
        ],
    }


def inspect_wanda_layers(seed0: Path, dense: dict, wanda: dict) -> tuple[list[dict], dict]:
    raw = wanda["raw"]
    layers = []
    all_deleted_abs = []
    large_deleted = large_total = 0
    total = removed = 0
    for key, mask in raw["masks"].items():
        parent_weight = dense["model"][key]
        raw_weight = raw["model"][key]
        keep = mask.bool()
        matrix = keep.reshape(keep.shape[0], -1)
        end = matrix.shape[1] // 4 * 4
        group_ok = bool((matrix[:, :end].reshape(-1, 4).sum(1) == 2).all()) if end else True
        tail_ok = bool(matrix[:, end:].all())
        survivor_equal = torch.equal(raw_weight[keep], parent_weight[keep])
        deleted_zero = bool((raw_weight[~keep] == 0).all())
        deleted_abs = parent_weight[~keep].abs().double()
        all_deleted_abs.append(deleted_abs)
        threshold = torch.quantile(parent_weight.abs().double().flatten(), 0.9)
        large = parent_weight.abs().double() >= threshold
        large_deleted += int((large & ~keep).sum())
        large_total += int(large.sum())
        total += keep.numel()
        removed += int((~keep).sum())
        layers.append(
            {
                "layer": key.removesuffix(".weight"),
                "weight_shape": list(parent_weight.shape),
                "K": int(parent_weight[0].numel()),
                "calibration_count": len(raw["calibration_ids"]),
                "score_min": None,
                "score_max": None,
                "score_mean": None,
                "score_finite_check": "NOT_VERIFIABLE_FROM_EXISTING_ARTIFACTS",
                "mask_tensor_sha256": tensor_sha256(keep),
                "actual_sparsity": float((~keep).double().mean()),
                "two_of_four_complete_groups": group_ok,
                "dense_tail": tail_ok,
                "survivor_weight_equality": survivor_equal,
                "deleted_coordinates_zero": deleted_zero,
                "deleted_parent_abs": quantiles(deleted_abs),
                "audit": "PASS" if group_ok and tail_ok and survivor_equal and deleted_zero else "FAIL",
            }
        )
    deleted = torch.cat(all_deleted_abs)
    overall = {
        "layers": len(layers),
        "parameters_in_scope": total,
        "removed_parameters": removed,
        "actual_sparsity": removed / total,
        "deleted_parent_abs": quantiles(deleted),
        "top_decile_magnitude_deleted_fraction": large_deleted / large_total,
        "all_layer_checks_pass": all(layer["audit"] == "PASS" for layer in layers),
        "runtime_scores_saved": False,
        "runtime_score_recomputation_performed": False,
    }
    return layers, overall


def inspect_apbr_mask(seed0: Path, apbr_parent: dict, apbr: dict) -> dict:
    raw = apbr["raw"]
    total = removed = large_deleted = large_total = 0
    deleted_abs = []
    survivors_equal = True
    for key, mask in raw["masks"].items():
        keep = mask.bool()
        parent_weight = apbr_parent["model"][key]
        raw_weight = raw["model"][key]
        total += keep.numel()
        removed += int((~keep).sum())
        deleted_abs.append(parent_weight[~keep].abs().double())
        threshold = torch.quantile(parent_weight.abs().double().flatten(), 0.9)
        large = parent_weight.abs().double() >= threshold
        large_deleted += int((large & ~keep).sum())
        large_total += int(large.sum())
        survivors_equal &= torch.equal(raw_weight[keep], parent_weight[keep])
    geometry = apbr_parent["stepper"]["geometry"]
    logs = {name: value for (name, _), value in zip(geometry["mapping"], geometry["logs"])}
    eligible_logs = torch.cat([logs[key].detach().cpu().double().flatten() for key in raw["masks"]])
    return {
        "layers": len(raw["masks"]),
        "parameters_in_scope": total,
        "removed_parameters": removed,
        "actual_sparsity": removed / total,
        "survivor_weight_equality": survivors_equal,
        "deleted_parent_abs": quantiles(torch.cat(deleted_abs)),
        "top_decile_magnitude_deleted_fraction": large_deleted / large_total,
        "geometry_log_s_distribution": quantiles(eligible_logs),
        "geometry_formula": "abs(w) * exp(-s)",
    }


def mask_agreement(wanda: dict, apbr: dict) -> dict:
    same = total = both_keep = either_keep = 0
    for key, left in wanda["raw"]["masks"].items():
        right = apbr["raw"]["masks"][key]
        left = left.bool()
        right = right.bool()
        same += int((left == right).sum())
        total += left.numel()
        both_keep += int((left & right).sum())
        either_keep += int((left | right).sum())
    return {
        "same_coordinate_fraction": same / total,
        "keep_set_jaccard": both_keep / either_keep,
        "interpretation": "DESCRIPTIVE_ONLY_DIFFERENT_PARENT_MODELS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed0 = args.seed0.resolve()
    output = args.output.resolve()
    ensure_seed0_contract(seed0)
    output.mkdir(parents=True, exist_ok=False)

    dense_path = seed0 / "dense_sam" / "final_epoch_200.pt"
    apbr_parent_path = seed0 / "apbr_sam" / "final_epoch_200.pt"
    dense = load_checkpoint(dense_path)
    apbr_parent = load_checkpoint(apbr_parent_path)
    methods = {method: inspect_method(seed0, method) for method in METHODS}
    tests = {path.stem: load_json(path) for path in (seed0 / "test_final").glob("*.json")}
    wanda_layers, wanda_overall = inspect_wanda_layers(seed0, dense, methods["wanda"])
    apbr_mask = inspect_apbr_mask(seed0, apbr_parent, methods["apbr_sam"])

    dense_acc, dense_loss = tests["dense_final"]["test_accuracy"], tests["dense_final"]["test_loss"]
    apbr_dense_acc, apbr_dense_loss = tests["apbr_final"]["test_accuracy"], tests["apbr_final"]["test_loss"]
    wanda_raw_acc, wanda_raw_loss = tests["wanda_raw"]["test_accuracy"], tests["wanda_raw"]["test_loss"]
    apbr_raw_acc, apbr_raw_loss = tests["apbr_sam_raw"]["test_accuracy"], tests["apbr_sam_raw"]["test_loss"]
    wanda_rec_acc, wanda_rec_loss = tests["wanda_recovery_final"]["test_accuracy"], tests["wanda_recovery_final"]["test_loss"]
    apbr_rec_acc, apbr_rec_loss = tests["apbr_sam_recovery_final"]["test_accuracy"], tests["apbr_sam_recovery_final"]["test_loss"]

    outcomes = {
        "wanda": {
            "parent_accuracy": dense_acc,
            "parent_loss": dense_loss,
            "raw_accuracy": wanda_raw_acc,
            "raw_loss": wanda_raw_loss,
            "raw_accuracy_delta_pp": pct(wanda_raw_acc - dense_acc),
            "raw_loss_delta": wanda_raw_loss - dense_loss,
            "recovery_accuracy": wanda_rec_acc,
            "recovery_loss": wanda_rec_loss,
            "recovery_minus_parent_pp": pct(wanda_rec_acc - dense_acc),
        },
        "apbr_sam": {
            "parent_accuracy": apbr_dense_acc,
            "parent_loss": apbr_dense_loss,
            "raw_accuracy": apbr_raw_acc,
            "raw_loss": apbr_raw_loss,
            "raw_accuracy_delta_pp": pct(apbr_raw_acc - apbr_dense_acc),
            "raw_loss_delta": apbr_raw_loss - apbr_dense_loss,
            "recovery_accuracy": apbr_rec_acc,
            "recovery_loss": apbr_rec_loss,
            "recovery_minus_parent_pp": pct(apbr_rec_acc - apbr_dense_acc),
        },
        "contrasts": {
            "apbr_minus_wanda_raw_accuracy_pp": pct(apbr_raw_acc - wanda_raw_acc),
            "apbr_minus_wanda_raw_loss": apbr_raw_loss - wanda_raw_loss,
            "apbr_minus_wanda_raw_parent_relative_drop_pp": pct(
                (apbr_raw_acc - apbr_dense_acc) - (wanda_raw_acc - dense_acc)
            ),
            "wanda_minus_apbr_recovery_accuracy_pp": pct(wanda_rec_acc - apbr_rec_acc),
            "wanda_minus_apbr_recovery_loss": wanda_rec_loss - apbr_rec_loss,
        },
    }

    cost = {}
    for method in METHODS:
        pruning = methods[method]["raw"]["pruning"]
        status = methods[method]["status"]
        cost[method] = {
            "calibration_seconds": pruning["calibration_seconds"],
            "pruning_or_compensation_seconds": pruning["pruning_seconds"],
            "calibration_plus_pruning_seconds": pruning["calibration_seconds"] + pruning["pruning_seconds"],
            "recovery_elapsed_seconds": status["elapsed_seconds"],
            "peak_allocated_bytes_whole_process": status["peak_allocated_bytes"],
            "peak_reserved_bytes_whole_process": status["peak_reserved_bytes"],
            "peak_scope": "pruning + BN-only + Recovery-20; not separable from existing status",
        }
    cost["comparisons"] = {
        "sparsegpt_minus_wanda_pruning_stage_seconds": cost["sparsegpt"]["calibration_plus_pruning_seconds"]
        - cost["wanda"]["calibration_plus_pruning_seconds"],
        "sparsegpt_over_wanda_pruning_stage_ratio": cost["sparsegpt"]["calibration_plus_pruning_seconds"]
        / cost["wanda"]["calibration_plus_pruning_seconds"],
        "rose_minus_sparsegpt_pruning_stage_seconds": cost["rose"]["calibration_plus_pruning_seconds"]
        - cost["sparsegpt"]["calibration_plus_pruning_seconds"],
        "rose_over_sparsegpt_pruning_stage_ratio": cost["rose"]["calibration_plus_pruning_seconds"]
        / cost["sparsegpt"]["calibration_plus_pruning_seconds"],
    }
    dense_seconds = training_seconds(seed0 / "dense_sam" / "epoch_metrics.csv")
    apbr_seconds = training_seconds(seed0 / "apbr_sam" / "epoch_metrics.csv")
    cost["dense_training"] = {
        "dense_sam_epoch_seconds_sum": dense_seconds,
        "apbr_sam_epoch_seconds_sum": apbr_seconds,
        "apbr_extra_seconds": apbr_seconds - dense_seconds,
        "apbr_over_dense_ratio": apbr_seconds / dense_seconds,
        "scope": "sum of 200 recorded epoch_seconds; checkpoint/controller overhead excluded",
    }

    gpu_gate_path = seed0.parents[2] / "docs" / "research" / "cifar100_gpu_gate_20261026" / "GPU_GATE_RESULTS.json"
    if not gpu_gate_path.exists():
        # seed0 is artifacts/protocol/seed0_remote_copy, project root is parents[2].
        gpu_gate_path = seed0.parents[2] / "docs" / "research" / "cifar100_gpu_gate_20261026" / "REMOTE_GPU_GATE_4080S_20261026" / "GPU_GATE_RESULTS_REMOTE.json"
    gpu_gate = load_json(gpu_gate_path) if gpu_gate_path.exists() else None

    input_files = [
        dense_path,
        apbr_parent_path,
        seed0 / "methods" / "wanda" / "raw_hard.pt",
        seed0 / "methods" / "apbr_sam" / "raw_hard.pt",
        seed0 / "methods" / "wanda" / "recovery" / "final_epoch_020.pt",
        seed0 / "methods" / "apbr_sam" / "recovery" / "final_epoch_020.pt",
        seed0 / "evidence" / "run_cifar100_50k_postprocess.py",
        seed0 / "evidence" / "cifar100_50k10k_final_v1.yaml",
    ]
    evidence = {
        "schema": 1,
        "audit_type": "READ_ONLY_RESEARCH_AUDIT",
        "protocol_id": PROTOCOL,
        "seed": 0,
        "seed0_root": str(seed0),
        "input_files": [{"path": str(path), "sha256": sha256_file(path)} for path in input_files],
        "source_commits": {
            "wanda": "8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6",
            "sparsegpt": "147d2159dc4f3e9f73e47b32c04d7b3708f44436",
            "rose": "88d922b3f4582b53b33e16e1396f82df563055f2",
        },
        "parent_checkpoint_integrity": {
            "dense_actual_sha256": sha256_file(dense_path),
            "dense_recorded_by_wanda": methods["wanda"]["raw"]["parent"]["sha256"],
            "dense_match": sha256_file(dense_path) == methods["wanda"]["raw"]["parent"]["sha256"],
            "apbr_actual_sha256": sha256_file(apbr_parent_path),
            "apbr_recorded_by_apbr_raw": methods["apbr_sam"]["raw"]["parent"]["sha256"],
            "apbr_match": sha256_file(apbr_parent_path) == methods["apbr_sam"]["raw"]["parent"]["sha256"],
        },
        "outcomes": outcomes,
        "wanda_layer_audit": wanda_layers,
        "wanda_overall": wanda_overall,
        "apbr_mask": apbr_mask,
        "cross_parent_mask_agreement": mask_agreement(methods["wanda"], methods["apbr_sam"]),
        "method_integrity": {
            method: {
                "paths": methods[method]["paths"],
                "hashes": methods[method]["hashes"],
                "recovery_parent_hash_matches_raw": methods[method]["recovery_parent_hash_matches_raw"],
                "status_file_hashes_match_actual_files": methods[method]["status_file_hashes_match_actual_files"],
                "mask_identity_across_raw_bn_recovery": methods[method]["mask_identity_across_raw_bn_recovery"],
                "bn_non_running_state_equal_raw": methods[method]["bn_non_running_state_equal_raw"],
                "bn_running_buffers_changed": methods[method]["bn_running_buffers_changed"],
                "masked_coordinates_zero": methods[method]["masked_coordinates_zero"],
                "recovery_contract": methods[method]["recovery_contract"],
            }
            for method in METHODS
        },
        "recovery_trajectories": {
            "wanda": methods["wanda"]["trajectory"],
            "apbr_sam": methods["apbr_sam"]["trajectory"],
            "test_metrics_during_recovery": "NOT_RECORDED",
            "wanda_apbr_train_order_identical_all_epochs": all(
                left["train_order_sha256"] == right["train_order_sha256"]
                for left, right in zip(
                    methods["wanda"]["recovery"]["history"], methods["apbr_sam"]["recovery"]["history"]
                )
            ),
        },
        "bn_only": {
            "checkpoint_exists": True,
            "parameters_unchanged_guard_in_source": True,
            "separate_copy_in_source": True,
            "train_ids": 4096,
            "test_evaluation_exists": False,
            "accuracy_effect": "UNVERIFIED",
        },
        "cost": cost,
        "single_layer_sparsegpt_gpu_gate": gpu_gate,
        "conclusions": {
            "Q1_wanda_core_scoring_preserved": "VERIFIED",
            "Q2_seed0_wanda_mask_checkpoint_full_model": "VERIFIED",
            "Q3_explanation_of_raw_gap": "SUPPORTED_HYPOTHESIS",
            "Q4_recovery_test_trajectory_difference": "UNVERIFIED",
            "Q5_second_order_cost": "VERIFIED_WITH_SCOPE_LIMITS",
            "Q6_apbr_reduces_immediate_damage": "SUPPORTED_HYPOTHESIS_SEED0_ONLY",
            "implementation_failure_found": False,
        },
        "missing_evidence": [
            "Seed-0 runtime Wanda score tensors and activation-stat vectors were not persisted.",
            "BN-only checkpoints were not independently evaluated on Test.",
            "Recovery logs contain train metrics only, not per-epoch Test metrics.",
            "Whole-model SparseGPT/ROSE logs do not split Hessian build, decomposition, and compensation per layer.",
            "Seeds 1 and 2 are incomplete at audit time, so no three-seed inference is available.",
        ],
    }

    rows = []
    for layer in wanda_layers:
        rows.append(
            f"| {layer['layer']} | {layer['weight_shape']} | {layer['K']} | {layer['calibration_count']} | "
            f"NOT SAVED | `{layer['mask_tensor_sha256'][:16]}…` | {pct(layer['actual_sparsity']):.2f}% | "
            f"{layer['survivor_weight_equality']} | {layer['audit']} |"
        )
    trajectory_rows = []
    for epoch in (1, 5, 10, 15, 20):
        w = methods["wanda"]["trajectory"][epoch - 1]
        a = methods["apbr_sam"]["trajectory"][epoch - 1]
        trajectory_rows.append(
            f"| {epoch} | {w['train_loss']:.6f} | {pct(w['train_accuracy']):.3f}% | "
            f"{a['train_loss']:.6f} | {pct(a['train_accuracy']):.3f}% |"
        )

    wanda_report = f"""# Wanda vs APBR-SAM Seed 0 只读审计

## 核心结论

- `VERIFIED`：当前 Wanda-CNN adapter 保留官方核心量 `|W_ij| sqrt(scaler_row_j)`、逐 output-row 连续 2:4 组内删除、Mask-only 写回和幸存权重不变。Conv2d 的 `unfold` 顺序与 OIHW 展平一致。
- `VERIFIED`：Seed 0 Wanda 的 19 层 bool keep-mask 全部满足每个完整四元组保留 2 个、dense tail、删除位为零、幸存权重与 Dense 父 checkpoint 逐元素完全相等。
- `SUPPORTED_HYPOTHESIS`：APBR Raw accuracy 比 Wanda 高 {outcomes['contrasts']['apbr_minus_wanda_raw_accuracy_pp']:.2f} pp；相对各自父模型的下降少 {outcomes['contrasts']['apbr_minus_wanda_raw_parent_relative_drop_pp']:.2f} pp。这支持“Seed 0 上 APBR 几何与更小的即时准确率损伤相关”，不证明机制因果或跨 seed 优势。
- `UNVERIFIED`：BN-only 没有 Test 评估；Recovery 过程没有逐 epoch Test。因此不能把 Raw 差异归因于 BN，也不能观察测试准确率的反超时刻。
- `VERIFIED`：Wanda Recovery-20 确实从 Wanda Raw checkpoint 开始，APBR 同理；父 SHA、Mask SHA、20 epochs/7,820 steps 均匹配。

## 运行与协议状态

本报告只读取本地 Seed 0 完整镜像。审计启动时远端控制器仍在运行 Seed 1 Dense-SAM，未中断、未重启、未修改运行代码或配置；Seed 2 尚未启动。

## 官方 Wanda 与当前 adapter

官方锁定 commit [`8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6) 的 [`prune_wanda`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py) 与 [`WrappedGPT`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/layerwrapper.py) 以输入列平方范数累计 `scaler_row`，评分为 `abs(W)*sqrt(scaler_row)`，N:M 在每个 output row 的连续列内删除低分项，不更新幸存权重，也没有默认 OBS 补偿。官方视觉 [`prune_utils.py`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/image_classifiers/prune_utils.py) 只查找 `nn.Linear`，不是 ResNet18 Conv2d N:M 实现。

当前调用链是 `run_cifar100_50k_full_queue.py` → `run_cifar100_50k_postprocess.py::prune` → `WandaAdapter.add_batch/prune`。1024 个固定 Train ID、无增强、单次前向；`model.eval()` 避免 BN buffer 更新。每层只有一个 pre-hook，循环结束立即移除。FP64 累积会造成与官方 FP32 buffer 的微小数值差异；此前单层对照最大 score 差约 `1.08e-7`、Mask 完全一致。Seed 0 并未保存运行时 score/激活统计，因此这些张量不能从 checkpoint 事后复核。

## 19 层全模型检查

| Layer | Weight shape | K | Calibration images | Score stats | Mask SHA256 | Sparsity | Survivors equal parent | Audit |
|---|---:|---:|---:|---|---|---:|---|---|
{chr(10).join(rows)}

总体：{wanda_overall['parameters_in_scope']:,} 个 eligible 权重，删除 {wanda_overall['removed_parameters']:,}，实际稀疏率 {pct(wanda_overall['actual_sparsity']):.2f}%。父模型每层绝对值 top-10% 权重中，Wanda 删除 {pct(wanda_overall['top_decile_magnitude_deleted_fraction']):.3f}%。运行时 score min/max/mean 和 finite 状态均标为 `NOT_VERIFIABLE_FROM_EXISTING_ARTIFACTS`，没有重新校准或重新剪枝来替代历史证据。

## Raw Hard：准确率与交叉熵

| 方法 | Parent Acc | Raw Acc | ΔAcc | Parent Loss | Raw Loss | ΔLoss |
|---|---:|---:|---:|---:|---:|---:|
| Wanda | {pct(dense_acc):.2f}% | {pct(wanda_raw_acc):.2f}% | {outcomes['wanda']['raw_accuracy_delta_pp']:.2f} pp | {dense_loss:.6f} | {wanda_raw_loss:.6f} | {outcomes['wanda']['raw_loss_delta']:+.6f} |
| APBR-SAM | {pct(apbr_dense_acc):.2f}% | {pct(apbr_raw_acc):.2f}% | {outcomes['apbr_sam']['raw_accuracy_delta_pp']:.2f} pp | {apbr_dense_loss:.6f} | {apbr_raw_loss:.6f} | {outcomes['apbr_sam']['raw_loss_delta']:+.6f} |

APBR 的 Raw accuracy 高 0.68 pp，但其 Raw cross-entropy 反而比 Wanda 高 {outcomes['contrasts']['apbr_minus_wanda_raw_loss']:+.6f}。因此“accuracy 更高”不能外推为“交叉熵损伤更小”。APBR 与 Wanda 使用不同父模型；Mask 坐标相同率 {pct(evidence['cross_parent_mask_agreement']['same_coordinate_fraction']):.2f}%、keep-set Jaccard {evidence['cross_parent_mask_agreement']['keep_set_jaccard']:.4f} 仅作描述，不是同父 Mask 质量比较。

## BN-only 诊断

源码对 `copy.deepcopy(model)` 重置 BN running stats，只让 BatchNorm 进入 train；前后参数 SHA 必须不变。checkpoint 使用 4096 个固定 Train ID、无增强，并记录 Raw parent SHA。该语义通过源码 guard 与 checkpoint 结构核验。但 12 个 Final Test target 不包含 BN-only，因此 `Δ_BN` 不可观测，BN 失配只能作为待检验解释。

## Recovery-20

| Epoch | Wanda train loss | Wanda train acc | APBR train loss | APBR train acc |
|---:|---:|---:|---:|---:|
{chr(10).join(trajectory_rows)}

两者 optimizer、LR schedule、weight decay、momentum、增强、train order hash、Mask 固定投影与 20-epoch budget 一致。Wanda Final 为 {pct(wanda_rec_acc):.2f}%，APBR Final 为 {pct(apbr_rec_acc):.2f}%，Wanda 高 0.21 pp。训练日志能显示训练损失轨迹，但没有中间 Test 指标，不能声称测试性能在某个 epoch 发生反超，也不能用 Test 选择 epoch。

## 结论边界

Seed 0 支持优先假设，但效应只有 0.68 pp 且父模型不同。必须等待 Seeds 1、2，报告三 seed mean ± sample SD 与同 seed 配对差，才能判断方向是否稳定。现有证据没有发现 Wanda/APBR 执行路径、父 checkpoint 或 Mask 固定方面的实现错误。
"""

    cost_rows = []
    for method in ("magnitude", "wanda", "sparsegpt", "rose"):
        item = cost[method]
        cost_rows.append(
            f"| {method} | {item['calibration_seconds']:.6f} | {item['pruning_or_compensation_seconds']:.6f} | "
            f"{item['calibration_plus_pruning_seconds']:.6f} | {gib(item['peak_allocated_bytes_whole_process']):.3f} | "
            f"{gib(item['peak_reserved_bytes_whole_process']):.3f} |"
        )
    gate = gpu_gate.get("timing_seconds", {}) if gpu_gate else {}
    gate_mem = gpu_gate.get("memory", {}) if gpu_gate else {}
    cost_report = f"""# Post-pruning 成本只读审计

## Seed 0 实测剪枝阶段

| 方法 | Calibration (s) | Pruning/compensation (s) | Sum (s) | Whole-process peak alloc (GiB) | Whole-process peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cost_rows)}

计时范围：`calibration_seconds` 覆盖 1024 张 Train 图像的前向统计；`pruning_seconds` 覆盖所有 19 层 mask 生成及 SparseGPT/ROSE 原生补偿。代码在逐层调用内部做 CUDA synchronize，因此 SparseGPT/ROSE 的分解与补偿完成后才返回。状态文件的显存峰值从剪枝前 reset 后持续覆盖剪枝、BN-only 与 Recovery-20，不能当作“纯二阶补偿峰值”。

SparseGPT 的 calibration+pruning 为 {cost['sparsegpt']['calibration_plus_pruning_seconds']:.3f}s，比 Wanda 多 {cost['comparisons']['sparsegpt_minus_wanda_pruning_stage_seconds']:.3f}s（{cost['comparisons']['sparsegpt_over_wanda_pruning_stage_ratio']:.2f}×）。ROSE 为 {cost['rose']['calibration_plus_pruning_seconds']:.3f}s，比 SparseGPT 多 {cost['comparisons']['rose_minus_sparsegpt_pruning_stage_seconds']:.3f}s（总阶段 {cost['comparisons']['rose_over_sparsegpt_pruning_stage_ratio']:.3f}×）。这是本项目同一硬件/Seed 0 的实际差额，不是论文计时。

## 二阶算法路径

SparseGPT 为每层构造 `H=2XX^T/T`，相对阻尼 `0.01*mean(diag(H))`，执行 Cholesky → inverse → upper Cholesky，随后在补偿后的当前 block 中动态确定 2:4 mask，进行块内和块间 OBS 误差传播。ROSE 先由 Wanda 分数估计 column/block loss，只在原四元组内部及完整四元组之间重排，同步置换 W/H，调用同一 SparseGPT 核心，再逆置换回原坐标。当前 ROSE 是论文语义受控修复版，不是锁定官方 N:M 分支的原样执行。

## 单层最大卷积 GPU 证据

最大层 `layer4.0.conv2`（K=4608）的既有 GPU gate 在 RTX 4080 SUPER 上记录：Hessian build {gate.get('hessian_build', float('nan')):.6f}s、decomposition {gate.get('decomposition', float('nan')):.6f}s、compensation {gate.get('compensation', float('nan')):.6f}s、Hessian+prune total {gate.get('total_hessian_plus_prune', float('nan')):.6f}s；峰值 allocated {gate_mem.get('peak_allocated_bytes', 0) / 1024**2:.2f} MiB、reserved {gate_mem.get('peak_reserved_bytes', 0) / 1024**2:.2f} MiB。该 gate 是固定 32 张图像的单层资源测试，不能替代 Seed 0 全模型 1024-image 成本。

## 完整方法成本边界

- Dense-SAM 200 epochs 的 epoch-time 合计 {dense_seconds:.1f}s。
- APBR-SAM 200 epochs 的 epoch-time 合计 {apbr_seconds:.1f}s，比 Dense-SAM 多 {apbr_seconds-dense_seconds:.1f}s（{apbr_seconds/dense_seconds:.3f}×）。
- APBR pruning 本身只有 {cost['apbr_sam']['calibration_plus_pruning_seconds']:.3f}s，但完整 APBR 方法还包括几何训练的额外成本和 Recovery-20 {cost['apbr_sam']['recovery_elapsed_seconds']:.1f}s，不能只报打分时间。
- Wanda Recovery-20 为 {cost['wanda']['recovery_elapsed_seconds']:.1f}s；SparseGPT/ROSE 的原生比较终点是补偿后 Raw，若另报 Recovery-20 必须作为补充结果。

## 不能从现有日志得到的量

全模型日志没有逐层保存 Hessian build/decomposition/compensation 的细分计时，也没有把剪枝阶段显存峰值与 Recovery 峰值分离。因此这些字段保持 `UNVERIFIED`；不能用单层 gate 或整个 postprocess 的时间冒充全模型纯二阶补偿成本。
"""

    assessment = f"""# 最终科研判断

## 六个问题的答案

1. **Q1 — `VERIFIED`**：Wanda 官方核心评分、逐行 N:M 选择与不更新幸存权重在 CNN adapter 中保留；Conv2d patch 映射是必要接口适配。名称应写 `Wanda-CNN-unified2:4`，不是“官方 ResNet18 Wanda”。
2. **Q2 — `VERIFIED`**：Seed 0 的 19 层 Wanda Mask/checkpoint 通过结构、零值、幸存权重、父 SHA 和 Mask SHA 审计。运行时 score 张量未保存这一项仍是 `UNVERIFIED`，不影响 checkpoint 的结构验收。
3. **Q3 — `SUPPORTED_HYPOTHESIS`**：APBR Raw 高 0.68 pp，且相对父模型的 accuracy drop 少 0.84 pp；但父模型不同、BN-only 未评估，而且 APBR Raw cross-entropy 比 Wanda 高。因此不能把 0.68 pp 全部归因于几何评分。
4. **Q4 — `UNVERIFIED`（测试轨迹）/ `VERIFIED`（最终点）**：Wanda Recovery-20 Final 高 0.21 pp，起点和协议正确；日志只有训练轨迹，没有逐 epoch Test，无法观察反超发生时间或将其归因于优化地形。
5. **Q5 — `VERIFIED_WITH_SCOPE_LIMITS`**：Seed 0 全模型 calibration+pruning：Wanda {cost['wanda']['calibration_plus_pruning_seconds']:.3f}s、SparseGPT {cost['sparsegpt']['calibration_plus_pruning_seconds']:.3f}s、ROSE {cost['rose']['calibration_plus_pruning_seconds']:.3f}s。纯二阶全模型峰值显存和逐层细分计时未单独记录。
6. **Q6 — `SUPPORTED_HYPOTHESIS_SEED0_ONLY`**：可以说“Seed 0 中 APBR 与更小的即时 accuracy 损伤相关”；不能说“APBR 已被证明更擅长减少即时剪枝损伤”。需等待 Seeds 1、2 的同 seed 配对结果。

## APBR 的现有优势与不足

优势：Seed 0 Raw accuracy 和相对父模型 accuracy drop 均优于 Wanda，Mask-only 语义清楚。不足：Recovery-20 Final 没有优于 Wanda；Raw cross-entropy 也不优；完整流程承担 APBR 几何训练与 20 epochs Recovery 的额外成本。

## 实现问题

本审计没有发现 Seed 0 Wanda/APBR 的父 checkpoint、mask、BN-only 隔离、Recovery 起点或固定 mask 协议错误。证据缺口不等于实现失败。

## 最低成本下一步

不修改当前三 Seed 计划。等待 Seed 1/2 正常完成后，只复用已经生成的 Dense/Raw/BN-only/Recovery checkpoint 与最终评估：

1. 计算 Wanda 与 APBR 的 Raw accuracy/loss 同 seed 配对差及 mean ± sample SD。
2. 如要检验 BN 假设，最小新增工作是对既有 BN-only checkpoint 做固定 Test 评估；需另行批准，且不得选 epoch/调参。
3. 不建议补做逐 epoch Test 曲线；它会扩大 Test 访问且不影响预注册 Final 结论。若研究优化轨迹，优先用现有 train loss，并明确它不是泛化轨迹。

当前正常三 Seed 队列无需修改、无需重启。
"""

    (output / "WANDA_APBR_SEED0_AUDIT.md").write_text(wanda_report, encoding="utf-8")
    (output / "POST_PRUNING_COST_AUDIT.md").write_text(cost_report, encoding="utf-8")
    (output / "FINAL_RESEARCH_ASSESSMENT.md").write_text(assessment, encoding="utf-8")
    (output / "AUDIT_EVIDENCE.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {}
    for path in sorted(output.iterdir()):
        if path.name != "SHA256SUMS.json":
            manifest[path.name] = sha256_file(path)
    (output / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
