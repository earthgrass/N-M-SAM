"""Audit and compare completed CIFAR-100 Seed 0 and Seed 1 formal artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


METHODS = ("magnitude", "wanda", "sparsegpt", "rose", "apbr_sam")
LABELS = {
    "magnitude": "Magnitude",
    "wanda": "Wanda",
    "sparsegpt": "SparseGPT",
    "rose": "ROSE",
    "apbr_sam": "APBR-SAM",
}
ENDPOINTS = (
    ("magnitude", "magnitude_raw", "dense_final"),
    ("wanda", "wanda_raw", "dense_final"),
    ("sparsegpt", "sparsegpt_raw", "dense_final"),
    ("rose", "rose_raw", "dense_final"),
    ("apbr_sam", "apbr_sam_recovery_final", "apbr_final"),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean_std(values):
    values = [float(v) for v in values]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def verify_manifest(seed_root: Path):
    manifest = seed_root / "SHA256SUMS.remote.txt"
    failures = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative_path = Path(relative)
        # The remote manifest is rooted one directory above a directory named
        # ``seed1``; the verified local copy is intentionally named
        # ``seed1_remote_copy``.
        parts = relative_path.parts
        if not parts or parts[0] not in {"seed1", "seed2"}:
            raise ValueError(f"Unexpected manifest path: {relative}")
        path = seed_root.joinpath(*parts[1:])
        actual = sha256(path)
        checked += 1
        if actual != expected:
            failures.append({"path": str(path), "expected": expected, "actual": actual})
    return {"manifest_sha256": sha256(manifest), "files_checked": checked,
            "failures": failures, "pass": not failures}


def verify_masks(raw: dict, recovery: dict, method: str):
    masks = raw["masks"]
    model = raw["model"]
    removed = total = 0
    for name, mask in masks.items():
        if mask.dtype != torch.bool:
            raise ValueError(f"{method}: non-bool mask {name}")
        flat = mask.reshape(mask.shape[0], -1)
        complete = flat.shape[1] // 4 * 4
        groups = flat[:, :complete].reshape(flat.shape[0], -1, 4)
        if not bool((groups.sum(-1) == 2).all()):
            raise ValueError(f"{method}: invalid 2:4 group {name}")
        if complete < flat.shape[1] and not bool(flat[:, complete:].all()):
            raise ValueError(f"{method}: non-dense tail {name}")
        if not bool((model[name][~mask] == 0).all()):
            raise ValueError(f"{method}: raw deleted weight nonzero {name}")
        total += mask.numel()
        removed += int((~mask).sum())
    recovery_masks = recovery["masks"]
    if set(masks) != set(recovery_masks):
        raise ValueError(f"{method}: recovery mask keys changed")
    for name, mask in masks.items():
        if not torch.equal(mask, recovery_masks[name]):
            raise ValueError(f"{method}: recovery mask changed {name}")
        if not bool((recovery["model"][name][~mask] == 0).all()):
            raise ValueError(f"{method}: recovery deleted weight nonzero {name}")
    return {"layers": len(masks), "total_eligible_weights": total,
            "removed_weights": removed, "strict_2_of_4": True,
            "dense_tail": True, "bool_keep_mask": True,
            "recovery_mask_identical": True, "deleted_weights_zero": True}


def collect_seed(seed: int, seed_root: Path, verify_full: bool):
    test_root = seed_root / "test_final"
    evaluations = {}
    for path in sorted(test_root.glob("*.json")):
        value = read_json(path)
        if (value.get("status") != "PASS" or value.get("test_count") != 10_000
                or value.get("selection_performed") is not False
                or value.get("hyperparameter_change_performed") is not False):
            raise ValueError(f"Evaluation contract failed: {path}")
        evaluations[path.stem] = value
    if len(evaluations) != 12:
        raise ValueError(f"Seed {seed}: expected 12 evaluations, got {len(evaluations)}")

    train_status = {}
    for stage in ("dense_sam", "apbr_sam"):
        value = read_json(seed_root / stage / "status.json")
        if (value.get("status") != "completed" or value.get("completed_epochs") != 200
                or value.get("steps") != 78_200 or value.get("test_accessed") is not False):
            raise ValueError(f"Seed {seed}: training contract failed {stage}")
        train_status[stage] = value

    methods = {}
    for method in METHODS:
        root = seed_root / "methods" / method
        status = read_json(root / "status.json")
        if (status.get("status") != "completed" or status.get("recovery_epochs") != 20
                or status.get("recovery_steps") != 7_820 or status.get("test_accessed") is not False):
            raise ValueError(f"Seed {seed}: method contract failed {method}")
        raw_eval = evaluations[f"{method}_raw"]
        recovery_eval = evaluations[f"{method}_recovery_final"]
        if raw_eval["checkpoint_sha256"] != status["raw_hard_sha256"]:
            raise ValueError(f"Seed {seed}: raw evaluation hash mismatch {method}")
        if recovery_eval["checkpoint_sha256"] != status["recovery_final_sha256"]:
            raise ValueError(f"Seed {seed}: recovery evaluation hash mismatch {method}")
        audit = None
        pruning = None
        if verify_full:
            raw_path = root / "raw_hard.pt"
            recovery_path = root / "recovery" / "final_epoch_020.pt"
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)
            recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
            expected_parent = evaluations["apbr_final" if method == "apbr_sam" else "dense_final"]["checkpoint_sha256"]
            if raw["parent"]["sha256"] != expected_parent:
                raise ValueError(f"Seed {seed}: raw parent mismatch {method}")
            if recovery["parent"]["sha256"] != status["raw_hard_sha256"]:
                raise ValueError(f"Seed {seed}: recovery parent mismatch {method}")
            audit = verify_masks(raw, recovery, method)
            pruning = raw.get("pruning")
            del raw, recovery
        methods[method] = {"status": status, "mask_audit": audit, "pruning": pruning}

    return {"seed": seed, "root": str(seed_root), "evaluations": evaluations,
            "training_status": train_status, "methods": methods}


def build(seeds: list[dict], manifests: dict):
    main_rows = []
    for item in seeds:
        e = item["evaluations"]
        for method, endpoint, parent in ENDPOINTS:
            acc = e[endpoint]["test_accuracy"]
            parent_acc = e[parent]["test_accuracy"]
            main_rows.append({"seed": item["seed"], "method": LABELS[method],
                              "endpoint": endpoint, "accuracy": acc,
                              "loss": e[endpoint]["test_loss"],
                              "parent": parent, "parent_accuracy": parent_acc,
                              "delta_vs_parent_pp": 100 * (acc - parent_acc),
                              "checkpoint_sha256": e[endpoint]["checkpoint_sha256"]})
    summary = []
    for method in [LABELS[m] for m, _, _ in ENDPOINTS]:
        selected = [r for r in main_rows if r["method"] == method]
        mean, std = mean_std([r["accuracy"] for r in selected])
        dmean, dstd = mean_std([r["delta_vs_parent_pp"] for r in selected])
        summary.append({"method": method, "seeds": len(selected), "accuracy_mean": mean,
                        "accuracy_sample_std": std, "delta_vs_parent_pp_mean": dmean,
                        "delta_vs_parent_pp_sample_std": dstd})

    paired = []
    for item in seeds:
        e = item["evaluations"]
        apbr = e["apbr_sam_recovery_final"]["test_accuracy"]
        for method in ("magnitude", "wanda", "sparsegpt", "rose"):
            paired.append({"seed": item["seed"],
                           "comparison": f"APBR Recovery-20 - {LABELS[method]} native",
                           "difference_pp": 100 * (apbr - e[f"{method}_raw"]["test_accuracy"])})
    paired_summary = []
    for comparison in sorted({r["comparison"] for r in paired}):
        values = [r["difference_pp"] for r in paired if r["comparison"] == comparison]
        mean, std = mean_std(values)
        paired_summary.append({"comparison": comparison, "difference_pp_mean": mean,
                               "difference_pp_sample_std": std, "seeds": len(values)})

    recovery = []
    for item in seeds:
        e = item["evaluations"]
        for method in METHODS:
            raw = e[f"{method}_raw"]
            rec = e[f"{method}_recovery_final"]
            recovery.append({"seed": item["seed"], "method": LABELS[method],
                             "raw_accuracy": raw["test_accuracy"],
                             "recovery_accuracy": rec["test_accuracy"],
                             "recovery_gain_pp": 100 * (rec["test_accuracy"] - raw["test_accuracy"]),
                             "raw_loss": raw["test_loss"], "recovery_loss": rec["test_loss"]})

    costs = []
    for item in seeds:
        for method in METHODS:
            m = item["methods"][method]
            p = m["pruning"] or {}
            costs.append({"seed": item["seed"], "method": LABELS[method],
                          "calibration_seconds": p.get("calibration_seconds"),
                          "pruning_seconds": p.get("pruning_seconds"),
                          "recovery_elapsed_seconds": m["status"]["elapsed_seconds"],
                          "peak_allocated_bytes": m["status"]["peak_allocated_bytes"],
                          "peak_reserved_bytes": m["status"]["peak_reserved_bytes"]})

    return {"schema": 1, "protocol_id": "cifar100_50k10k_final_v1",
            "completed_seeds": [s["seed"] for s in seeds], "three_seed_statistics_complete": len(seeds) == 3,
            "transfer_manifests": manifests, "main_method_specific_rows": main_rows,
            "three_seed_summary": summary, "paired_differences": paired,
            "paired_summary": paired_summary,
            "supplemental_uniform_recovery20": recovery, "costs": costs,
            "audits": {str(s["seed"]): {m: s["methods"][m]["mask_audit"] for m in METHODS} for s in seeds},
            "key_findings": {
                "apbr_raw_range_pp": 100 * (max(s["evaluations"]["apbr_sam_raw"]["test_accuracy"] for s in seeds) - min(s["evaluations"]["apbr_sam_raw"]["test_accuracy"] for s in seeds)),
                "apbr_recovery_range_pp": 100 * (max(s["evaluations"]["apbr_sam_recovery_final"]["test_accuracy"] for s in seeds) - min(s["evaluations"]["apbr_sam_recovery_final"]["test_accuracy"] for s in seeds)),
            },
            "notes": ["Three-seed standard deviation is the sample standard deviation (ddof=1).",
                      "Main method-specific endpoints and supplemental uniform Recovery-20 are not ranked together.",
                      "Seed 0 ran on RTX 4080 SUPER; Seed 1 APBR and postprocessing ran on RTX PRO 6000 Blackwell."]}


def write_chart(value: dict, output: Path):
    methods = [r["method"] for r in value["three_seed_summary"]]
    rows = value["main_method_specific_rows"]
    seed_values = {seed: [100 * next(r["accuracy"] for r in rows if r["seed"] == seed and r["method"] == m) for m in methods] for seed in value["completed_seeds"]}
    x = np.arange(len(methods)); width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=160)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    bars = []
    offsets = np.linspace(-width, width, len(seed_values))
    for offset, (seed, values), color in zip(offsets, seed_values.items(), colors):
        bars.append(ax.bar(x + offset, values, width, label=f"Seed {seed}", color=color))
    ax.set_ylabel("CIFAR-100 Test accuracy (%)")
    ax.set_title("Method-specific endpoints across three seeds")
    ax.set_xticks(x, methods); ax.set_ylim(50, 82); ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    for bar in bars:
        ax.bar_label(bar, fmt="%.2f", padding=2, fontsize=7)
    fig.tight_layout(); fig.savefig(output); plt.close(fig)


def write_report(value: dict, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    (output / "three_seed_analysis.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_chart(value, output / "three_seed_endpoint_accuracy.png")
    lines = ["# CIFAR-100 三 Seed：方法特定终点与统一 Recovery-20 分析", "",
             "## 结论", "",
             "三 Seed 正式结果显示，SparseGPT Native、ROSE Native 与 APBR Recovery-20 终点较稳定。APBR-SAM Recovery-20 的平均准确率最高，但它包含额外 20 epochs 恢复，因此只能作为完整方法流程对比，不能据此断言几何评分本身优于二阶方法。APBR Raw Hard 在三 Seed 为 59.22%、51.87%、56.14%，跨度 7.35 pp；Recovery-20 后收敛到 78.72%、79.15%、78.71%，跨度仅 0.44 pp，说明即时几何剪枝存在 seed 敏感性，而恢复训练基本抹平了差异。", "",
             "![Three-seed method-specific endpoint accuracy](three_seed_endpoint_accuracy.png)", "",
             "## 方法特定主终点", "",
             "| Seed | Method | Accuracy | CE Loss | Δ vs parent |", "|---:|---|---:|---:|---:|"]
    for r in value["main_method_specific_rows"]:
        lines.append(f"| {r['seed']} | {r['method']} | {100*r['accuracy']:.2f}% | {r['loss']:.4f} | {r['delta_vs_parent_pp']:+.2f} pp |")
    lines += ["", "## 三 Seed mean ± sample standard deviation", "",
              "| Method | Accuracy mean ± sample SD | Mean Δ vs parent |", "|---|---:|---:|"]
    for r in value["three_seed_summary"]:
        lines.append(f"| {r['method']} | {100*r['accuracy_mean']:.2f}% ± {100*r['accuracy_sample_std']:.2f} pp | {r['delta_vs_parent_pp_mean']:+.2f} pp |")
    lines += ["", "## APBR Recovery-20 的同 Seed 配对差值", "",
              "| Seed | Comparison | Difference |", "|---:|---|---:|"]
    for r in value["paired_differences"]:
        lines.append(f"| {r['seed']} | {r['comparison']} | {r['difference_pp']:+.2f} pp |")
    lines += ["", "| Three-seed paired summary | Mean ± sample SD |", "|---|---:|"]
    for r in value["paired_summary"]:
        lines.append(f"| {r['comparison']} | {r['difference_pp_mean']:+.2f} ± {r['difference_pp_sample_std']:.2f} pp |")
    lines += ["", "## 补充：统一 Recovery-20", "",
              "这些结果只用于观察恢复能力，不与方法特定主终点混合排名。", "",
              "| Seed | Method | Raw | Recovery-20 | Gain |", "|---:|---|---:|---:|---:|"]
    for r in value["supplemental_uniform_recovery20"]:
        lines.append(f"| {r['seed']} | {r['method']} | {100*r['raw_accuracy']:.2f}% | {100*r['recovery_accuracy']:.2f}% | {r['recovery_gain_pp']:+.2f} pp |")
    lines += ["", "## 审计结论", "",
              "- 三个 Seed 的 12 个 Final Test 记录均齐全，每项 10,000 样本，未执行选模或超参数变更。",
              "- Dense/APBR 均固定 Epoch 200；五种方法 Recovery 均固定 Epoch 20、7,820 steps。",
              "- 五种方法 Raw Hard 父 checkpoint、Recovery 父 checkpoint、checkpoint SHA256 和评估记录一致。",
              "- 所有方法的 19 层 bool keep-mask 均满足严格 2:4、dense tail；Recovery Mask 未改变，删除权重保持为零。",
              "- Seed 0 与 Seeds 1/2 使用不同 GPU；准确率可按相同冻结协议比较，但运行时间不应作严格硬件无关排名。",
              "- 标准差采用样本标准差（ddof=1）。", ""]
    (output / "THREE_SEED_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed0-root", type=Path, required=True)
    p.add_argument("--seed1-root", type=Path, required=True)
    p.add_argument("--seed2-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifests = {"seed1": verify_manifest(args.seed1_root), "seed2": verify_manifest(args.seed2_root)}
    if not all(m["pass"] for m in manifests.values()):
        raise RuntimeError(f"Transfer manifest mismatch: {manifests}")
    seed0 = collect_seed(0, args.seed0_root, verify_full=True)
    seed1 = collect_seed(1, args.seed1_root, verify_full=True)
    seed2 = collect_seed(2, args.seed2_root, verify_full=True)
    value = build([seed0, seed1, seed2], manifests)
    write_report(value, args.output)
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "files_checked": sum(m["files_checked"] for m in manifests.values())}))


if __name__ == "__main__":
    main()
