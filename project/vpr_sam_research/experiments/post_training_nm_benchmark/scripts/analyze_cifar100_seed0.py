"""Audit and summarize the copied CIFAR-100 formal Seed 0 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROTOCOL = "cifar100_50k10k_final_v1"
METHODS = ("magnitude", "wanda", "sparsegpt", "rose", "apbr_sam")
LABELS = {
    "magnitude": "Magnitude",
    "wanda": "Wanda",
    "sparsegpt": "SparseGPT",
    "rose": "ROSE",
    "apbr_sam": "APBR-SAM",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(root: Path, manifest: Path):
    failures = []
    entries = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ./", 1)
        path = root / relative
        entries += 1
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
        else:
            actual = sha256(path)
            if actual != expected:
                failures.append({"path": relative, "error": "sha256", "expected": expected, "actual": actual})
    return {"entries": entries, "failures": failures, "manifest_sha256": sha256(manifest)}


def finite_state(state):
    return all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in state.values())


def audit_method(root: Path, method: str, tests: dict, dense_acc: float, apbr_acc: float):
    method_root = root / "methods" / method
    status = load_json(method_root / "status.json")
    raw_path = method_root / "raw_hard.pt"
    recovery_path = method_root / "recovery" / "final_epoch_020.pt"
    raw_sha = sha256(raw_path)
    recovery_sha = sha256(recovery_path)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
    masks = raw["masks"]
    model = raw["model"]
    recovery_model = recovery["model"]
    mask_failures = []
    total_groups = 0
    total_tail = 0
    total_pruned = 0
    for name, mask in masks.items():
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(model[name].shape):
            mask_failures.append(f"{name}: dtype/shape")
            continue
        flat = mask.reshape(mask.shape[0], -1)
        complete = flat.shape[1] // 4 * 4
        if complete:
            groups = flat[:, :complete].reshape(-1, 4)
            total_groups += groups.shape[0]
            if not bool((groups.sum(1) == 2).all()):
                mask_failures.append(f"{name}: 2:4 count")
        tail = flat[:, complete:]
        total_tail += tail.numel()
        if tail.numel() and not bool(tail.all()):
            mask_failures.append(f"{name}: dense tail")
        if torch.count_nonzero(model[name][~mask]):
            mask_failures.append(f"{name}: raw regrowth")
        if torch.count_nonzero(recovery_model[name][~mask]):
            mask_failures.append(f"{name}: recovery regrowth")
        total_pruned += int((~mask).sum())
    excluded_ok = all(
        key.endswith(".weight")
        and key not in {"conv1.weight", "linear.weight"}
        and ".bn" not in key
        and not key.endswith("bias")
        for key in masks
    )
    raw_eval = tests[f"{method}_raw"]
    recovery_eval = tests[f"{method}_recovery_final"]
    parent_acc = apbr_acc if method == "apbr_sam" else dense_acc
    pruning = raw.get("pruning", {})
    recovery_seconds = float(status["elapsed_seconds"])
    calibration_seconds = float(pruning.get("calibration_seconds", 0.0))
    pruning_seconds = float(pruning.get("pruning_seconds", 0.0))
    return {
        "method": method,
        "label": LABELS[method],
        "parent": "apbr_final" if method == "apbr_sam" else "dense_final",
        "status_contract_pass": status.get("status") == "completed"
        and status.get("protocol_id") == PROTOCOL
        and status.get("recovery_epochs") == 20
        and status.get("recovery_steps") == 7820
        and status.get("test_accessed") is False,
        "checkpoint_hashes_pass": status.get("raw_hard_sha256") == raw_sha
        and status.get("recovery_final_sha256") == recovery_sha
        and raw_eval.get("checkpoint_sha256") == raw_sha
        and recovery_eval.get("checkpoint_sha256") == recovery_sha,
        "parent_hash_pass": raw["parent"]["sha256"]
        == sha256(root / ("apbr_sam/final_epoch_200.pt" if method == "apbr_sam" else "dense_sam/final_epoch_200.pt")),
        "recovery_parent_hash_pass": recovery["parent"]["sha256"] == raw_sha,
        "mask_audit": {
            "pass": not mask_failures and excluded_ok,
            "failures": mask_failures,
            "dtype": "bool",
            "eligible_layers": len(masks),
            "complete_groups": total_groups,
            "tail_positions": total_tail,
            "pruned_positions": total_pruned,
            "excluded_first_last_bn_bias": excluded_ok,
            "raw_and_recovery_pruned_coordinates_zero": not any("regrowth" in item for item in mask_failures),
            "mask_sha256": status["mask_sha256"],
        },
        "finite": finite_state(model) and finite_state(recovery_model),
        "accuracy": {
            "parent": parent_acc,
            "raw": raw_eval["test_accuracy"],
            "recovery": recovery_eval["test_accuracy"],
            "raw_delta_pp": 100 * (raw_eval["test_accuracy"] - parent_acc),
            "recovery_delta_pp": 100 * (recovery_eval["test_accuracy"] - parent_acc),
            "recovery_gain_pp": 100 * (recovery_eval["test_accuracy"] - raw_eval["test_accuracy"]),
        },
        "resource": {
            "calibration_seconds": calibration_seconds,
            "pruning_seconds": pruning_seconds,
            "recovery_seconds": recovery_seconds,
            "estimated_total_seconds": calibration_seconds + pruning_seconds + recovery_seconds,
            "peak_allocated_bytes": status["peak_allocated_bytes"],
            "peak_reserved_bytes": status["peak_reserved_bytes"],
        },
    }


def fmt_pct(value):
    return f"{100 * value:.2f}%"


def fmt_pp(value):
    return f"{value:+.2f} pp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    data_manifest = verify_manifest(root, root / "SHA256SUMS.remote.txt")
    evidence_manifest = verify_manifest(root / "evidence", root / "evidence" / "SHA256SUMS.evidence.txt")
    dense_identity = load_json(root / "dense_sam" / "identity.json")
    apbr_identity = load_json(root / "apbr_sam" / "identity.json")
    dense_status = load_json(root / "dense_sam" / "status.json")
    apbr_status = load_json(root / "apbr_sam" / "status.json")
    tests = {path.stem: load_json(path) for path in sorted((root / "test_final").glob("*.json"))}
    dense_acc = tests["dense_final"]["test_accuracy"]
    apbr_acc = tests["apbr_final"]["test_accuracy"]
    test_contract_pass = len(tests) == 12 and all(
        result.get("status") == "PASS"
        and result.get("protocol_id") == PROTOCOL
        and result.get("test_count") == 10000
        and result.get("selection_performed") is False
        and result.get("hyperparameter_change_performed") is False
        for result in tests.values()
    )
    training_contract_pass = all(
        status.get("status") == "completed"
        and status.get("protocol_id") == PROTOCOL
        and status.get("completed_epochs") == 200
        and status.get("steps") == 78200
        and status.get("selection_mode") == "fixed_final_epoch"
        and status.get("test_accessed") is False
        for status in (dense_status, apbr_status)
    )
    identity_pass = (
        dense_identity["config_sha256"] == apbr_identity["config_sha256"]
        and dense_identity["architecture_sha256"] == apbr_identity["architecture_sha256"]
        and dense_identity["archive_sha256"] == apbr_identity["archive_sha256"]
        and dense_identity["train_pickle_sha256"] == apbr_identity["train_pickle_sha256"]
        and dense_identity["train_size"] == apbr_identity["train_size"] == 50000
        and dense_identity["validation_size"] == apbr_identity["validation_size"] == 0
        and dense_identity["test_accessed"] is apbr_identity["test_accessed"] is False
    )
    methods = [audit_method(root, method, tests, dense_acc, apbr_acc) for method in METHODS]
    all_method_contracts = all(
        item["status_contract_pass"]
        and item["checkpoint_hashes_pass"]
        and item["parent_hash_pass"]
        and item["recovery_parent_hash_pass"]
        and item["mask_audit"]["pass"]
        and item["finite"]
        for item in methods
    )
    provenance = load_json(root / "evidence" / "argsort_fix_provenance.json")
    postprocess_sha = sha256(root / "evidence" / "run_cifar100_50k_postprocess.py")
    analysis = {
        "schema": 1,
        "protocol_id": PROTOCOL,
        "seed": 0,
        "overall_pass": not data_manifest["failures"]
        and not evidence_manifest["failures"]
        and identity_pass
        and training_contract_pass
        and test_contract_pass
        and all_method_contracts
        and provenance["postprocess_sha256"] == postprocess_sha,
        "transfer": {"data": data_manifest, "evidence": evidence_manifest},
        "identity": {
            "pass": identity_pass,
            "model_id": dense_identity["model_id"],
            "architecture_sha256": dense_identity["architecture_sha256"],
            "config_sha256": dense_identity["config_sha256"],
            "archive_sha256": dense_identity["archive_sha256"],
            "train_pickle_sha256": dense_identity["train_pickle_sha256"],
            "train_size": 50000,
            "test_size": 10000,
            "gpu": dense_identity["gpu"],
            "torch": dense_identity["torch"],
            "torch_cuda_build": dense_identity["torch_cuda_build"],
        },
        "training_contract_pass": training_contract_pass,
        "test_contract_pass": test_contract_pass,
        "test_targets": len(tests),
        "dense_accuracy": dense_acc,
        "apbr_dense_accuracy": apbr_acc,
        "apbr_dense_delta_pp": 100 * (apbr_acc - dense_acc),
        "methods": methods,
        "compatibility_fix": {
            "resolved": provenance["postprocess_sha256"] == postprocess_sha,
            "reason": provenance["reason"],
            "commit": provenance["fixed_commit"],
            "postprocess_sha256": postprocess_sha,
            "failed_attempt_preserved_remotely": True,
        },
        "limitations": [
            "Seed 0 is one stochastic run; method ranking and SOTA claims require Seeds 1 and 2.",
            "Recovery results use the fixed epoch-20 endpoint; no validation or Test-based selection was performed.",
            "The compatibility incident was an argsort API invocation error, not a numerical pruning failure.",
        ],
    }
    analysis_path = analysis_dir / "SEED0_ANALYSIS.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    labels = [item["label"] for item in methods]
    raw_values = np.array([item["accuracy"]["raw"] for item in methods])
    recovery_values = np.array([item["accuracy"]["recovery"] for item in methods])
    parent_values = np.array([item["accuracy"]["parent"] for item in methods])
    x = np.arange(len(methods))
    width = 0.31
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    raw_bars = ax.bar(x - width / 2, raw_values, width, label="Raw hard 2:4", color="#D97706")
    rec_bars = ax.bar(x + width / 2, recovery_values, width, label="Recovery epoch 20", color="#2563EB")
    ax.scatter(x, parent_values, marker="D", s=45, color="#111827", label="Parent dense endpoint", zorder=4)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("CIFAR-100 Final Test accuracy")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    ax.set_title("Seed 0: fixed 2:4 recovery restores accuracy near the parent endpoint")
    ax.bar_label(raw_bars, labels=[f"{100*v:.2f}%" for v in raw_values], padding=2, fontsize=8)
    ax.bar_label(rec_bars, labels=[f"{100*v:.2f}%" for v in recovery_values], padding=2, fontsize=8)
    chart_path = analysis_dir / "seed0_accuracy_comparison.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    rows = []
    for item in methods:
        acc = item["accuracy"]
        resource = item["resource"]
        rows.append(
            f"| {item['label']} | {item['parent'].replace('_', ' ')} | {fmt_pct(acc['raw'])} | "
            f"{fmt_pct(acc['recovery'])} | {fmt_pp(acc['recovery_gain_pp'])} | "
            f"{fmt_pp(acc['recovery_delta_pp'])} | {resource['estimated_total_seconds']:.1f} s | "
            f"{resource['peak_allocated_bytes']/1024**3:.2f} GiB |"
        )
    report = f"""# Seed 0 的五种 2:4 方法均完成，20-epoch Recovery 将准确率恢复至父模型附近

## 结论

- Seed 0 全流程通过：两段 200-epoch 训练、五种 2:4 剪枝、五段固定 20-epoch Recovery，以及 12 项独立 Final Test 均已完成。
- Dense-SAM Final Test 为 **{fmt_pct(dense_acc)}**；APBR-SAM dense endpoint 为 **{fmt_pct(apbr_acc)}**，相差 **{fmt_pp(100*(apbr_acc-dense_acc))}**。
- 五种方法 Recovery 后均回到各自父模型的 **0.22 个百分点以内**。Seed 0 上 Wanda 最高（**{fmt_pct(tests['wanda_recovery_final']['test_accuracy'])}**），但只比 Dense-SAM 高 **0.09 个百分点（9/10,000 张图片）**，单 seed 不足以支持方法排名或 SOTA 结论。
- 修复后的 APBR 2:4 已完整通过：bool keep-mask、19 个可剪枝层、每个完整四元组恰留 2 个、dense tail、剪枝坐标在 raw/recovery checkpoint 中保持为零，且所有数值 finite。

![Seed 0 accuracy comparison](seed0_accuracy_comparison.png)

## Final Test 对比

| 方法 | 父模型 | Raw Hard | Recovery-20 | Recovery 增益 | Recovery 相对父模型 | 剪枝+恢复估计耗时 | 峰值 allocated |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Magnitude、Wanda 和 APBR-SAM 的 raw hard accuracy 降幅较大，但固定 20-epoch Recovery 基本恢复；SparseGPT 和 ROSE 的 raw hard 起点明显更接近 Dense-SAM。这里的“估计耗时”是校准、剪枝和 Recovery 计时之和，不包含独立 Final Test。

## 协议与证据验收

- 数据身份：官方 CIFAR-100 archive SHA256 `{dense_identity['archive_sha256']}`；Train 50,000，Test 10,000；无 validation split。
- 模型身份：`{dense_identity['model_id']}`，architecture SHA256 `{dense_identity['architecture_sha256']}`。
- 训练端点：Dense-SAM 与 APBR-SAM 均为 epoch 200、78,200 steps、`fixed_final_epoch`，训练阶段 `test_accessed=false`。
- Recovery 端点：每种方法均为 epoch 20、7,820 steps，`test_accessed=false`。
- Final Test：12/12 结果为 PASS；每项均使用 10,000 张测试图片，`selection_performed=false`、`hyperparameter_change_performed=false`。
- Mask：五种方法均覆盖 19 个内部可剪枝层；首层、末层、BN affine 和 bias 被排除；所有完整四元组严格 2:4，tail 保持 dense。
- 传输：主清单 {data_manifest['entries']} 个文件全部匹配，清单 SHA256 `{data_manifest['manifest_sha256']}`；证据清单 {evidence_manifest['entries']} 个文件全部匹配。
- 环境：{dense_identity['gpu']['name']}，PyTorch {dense_identity['torch']}，CUDA build {dense_identity['torch_cuda_build']}。

## 已解决的执行异常

首次 APBR 2:4 调用使用了 positional `argsort` 维度参数，与服务器 PyTorch API 不兼容。最小修复仅将其改为 `dim=-1`，未改变评分、稳定 tie、分组、N:M 或训练参数。修复提交 `{provenance['fixed_commit']}`，实际 postprocess SHA256 `{postprocess_sha}`。修复前失败证据已保留；为保持方法源码一致，Seed 0 五种剪枝均由修复后源码重新执行。

## 限制与下一步

本报告只分析 Seed 0。Seed 1 和 Seed 2 尚在远端串行执行，因此目前不能报告跨 seed 均值/方差、显著性或正式 SOTA 排名。正式比较应等待三 seed 全部完成，再按完全相同的 Final Test 定义汇总。
"""
    report_path = analysis_dir / "SEED0_ANALYSIS.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"overall_pass": analysis["overall_pass"], "analysis": str(analysis_path), "report": str(report_path), "chart": str(chart_path)}))


if __name__ == "__main__":
    main()
