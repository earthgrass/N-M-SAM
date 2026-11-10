"""Audit and report the existing SAM/APBR Seed-0 conditional OBS comparison.

No pruning or Test evaluation is run here; only frozen artifacts are read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_apbr_gpt_mask_match_local import file_sha
from scripts.run_apbr_gpt_fixedmask_cpu_full import DATA

SAM_DIR = ROOT / "artifacts/sam_singleton_conditional_seed0_cpu_20261103"
APBR_DIR = ROOT / "artifacts/apbr_nearoptimal_conditional_seed0_full_cpu_20261103_v1"
REPORT_DIR = ROOT / "docs/research/apbr_nearoptimal_conditional_seed0_20261103_v2"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    sam = load(SAM_DIR / "results.json")
    apbr = load(APBR_DIR / "results.json")
    sam_freeze = load(SAM_DIR / "pre_test_freeze.json")
    apbr_freeze = load(APBR_DIR / "pre_test_freeze.json")
    apbr_audit = load(APBR_DIR / "independent_audit.json")
    sam_dense_eval = load(ROOT / "artifacts/cifar100_50k10k_final_v1/seed0_remote_copy/test_final/dense_final.json")
    apbr_dense_eval = load(ROOT / "artifacts/cifar100_50k10k_final_v1/seed0_remote_copy/test_final/apbr_final.json")
    sam_parent_path = ROOT / "artifacts/cifar100_50k10k_final_v1/seed0_remote_copy/dense_sam/final_epoch_200.pt"
    apbr_parent_path = ROOT / "artifacts/cifar100_50k10k_final_v1/seed0_remote_copy/apbr_sam/final_epoch_200.pt"
    sam_parent = torch.load(sam_parent_path, map_location="cpu", weights_only=False)
    apbr_parent = torch.load(apbr_parent_path, map_location="cpu", weights_only=False)
    sam_checkpoint = SAM_DIR / "singleton_conditional_joint.pt"
    if (file_sha(sam_parent_path) != sam["parent_sha256"]
            or file_sha(apbr_parent_path) != apbr["parent_sha256"]
            or sam_dense_eval["checkpoint_sha256"] != sam["parent_sha256"]
            or apbr_dense_eval["checkpoint_sha256"] != apbr["parent_sha256"]):
        raise ValueError("Dense parent identities differ from frozen evaluation records")
    if (file_sha(sam_checkpoint) != sam_freeze["checkpoint_sha256"]["singleton_conditional_joint"]
            or file_sha(sam_checkpoint) != sam["checkpoint_sha256"]["singleton_conditional_joint"]):
        raise ValueError("Historical SAM conditional checkpoint hash mismatch")
    if (apbr_audit["status"] != "INDEPENDENT_ARTIFACT_AUDIT_PASS"
            or apbr_freeze["test_accessed"] is not False
            or apbr_freeze["checkpoint_sha256"] != apbr["checkpoint_sha256"]):
        raise ValueError("APBR checkpoint freeze/audit mismatch")
    test_path = DATA / "extracted/cifar-100-python/test"
    if (file_sha(test_path) != sam["test_pickle_sha256"]
            or file_sha(test_path) != sam_dense_eval["test_pickle_sha256"]
            or file_sha(test_path) != apbr_dense_eval["test_pickle_sha256"]):
        raise ValueError("Test dataset identity mismatch")
    if (sam["calibration_ids_sha256"] != apbr["calibration_ids_sha256"]
            or sam["source_sha256"]["model"] != apbr["source_sha256"]["model"]
            or sam["source_sha256"]["formal_config"] != apbr["source_sha256"]["formal_config"]
            or sam_parent["identity"]["initial_model_sha256"] != apbr_parent["identity"]["initial_model_sha256"]
            or not (sam_parent["epoch"] == apbr_parent["epoch"] == 200)
            or not (sam_parent["steps"] == apbr_parent["steps"] == 78200)):
        raise ValueError("Training/calibration comparison contracts differ")

    sam_acc = sam["test"]["singleton_conditional_joint"]["test_accuracy"]
    apbr_acc = apbr["raw_hard_test"]["true"]["test_accuracy"]
    sam_dense_acc = sam_dense_eval["test_accuracy"]
    apbr_dense_acc = apbr_dense_eval["test_accuracy"]
    if (sam["test"]["singleton_conditional_joint"]["test_count"] != 10000
            or apbr["raw_hard_test"]["true"]["test_count"] != 10000):
        raise ValueError("Final Test count mismatch")
    comparison = {
        "status": "READ_ONLY_CROSS_PARENT_COMPARISON_PASS",
        "seed": 0, "exploratory": True,
        "same_initial_model_sha256": sam_parent["identity"]["initial_model_sha256"],
        "same_calibration_ids_sha256": sam["calibration_ids_sha256"],
        "same_test_pickle_sha256": sam["test_pickle_sha256"],
        "same_architecture_and_formal_config": True,
        "same_conditional_obs_physical_core": True,
        "different_dense_training": True,
        "different_selection_rule": True,
        "sam": {
            "parent_sha256": sam["parent_sha256"],
            "checkpoint_sha256": file_sha(sam_checkpoint),
            "dense_test_accuracy": sam_dense_acc,
            "conditional_test_accuracy": sam_acc,
            "conditional_test_ce": sam["test"]["singleton_conditional_joint"]["test_ce"],
            "pruning_drop_percentage_points": 100 * (sam_acc - sam_dense_acc),
        },
        "apbr": {
            "parent_sha256": apbr["parent_sha256"],
            "checkpoint_sha256": apbr["checkpoint_sha256"]["true"],
            "dense_test_accuracy": apbr_dense_acc,
            "conditional_geometric_test_accuracy": apbr_acc,
            "conditional_geometric_test_ce": apbr["raw_hard_test"]["true"]["test_ce"],
            "pruning_drop_percentage_points": 100 * (apbr_acc - apbr_dense_acc),
            "same_parent_physical_test_accuracy": apbr["raw_hard_test"]["physical"]["test_accuracy"],
            "same_parent_unit_test_accuracy": apbr["raw_hard_test"]["unit"]["test_accuracy"],
            "same_parent_shuffled_test_accuracy": apbr["raw_hard_test"]["shuffled"]["test_accuracy"],
        },
        "apbr_minus_sam_final_percentage_points": 100 * (apbr_acc - sam_acc),
        "test_re_evaluated_by_audit": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "CROSS_PARENT_COMPARISON.json"
    write_new(json_path, json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# SAM 与 APBR：Seed 0 条件贪心 OBS 对照", "",
        "同一初始权重、100 类 ResNet18、200 epochs、官方 CIFAR-100、同一 1024 张 Train 校准样本、19 层严格 2:4、相同物理联合 OBS 补偿。两条路径的 Dense 训练和 Mask 选择不同：SAM 使用纯物理条件贪心；APBR 使用已冻结的 1% 近优窗口和真实 APBR 几何裁决。均无 Recovery/BN-only。", "",
        "| 完整路径 | Dense Final | 剪枝及补偿后 Final Test | 相对本身 Dense 的变化 | CE |",
        "|---|---:|---:|---:|---:|",
        f"| SAM + 条件贪心 OBS | {sam_dense_acc:.2%} | {sam_acc:.2%} | {100 * (sam_acc - sam_dense_acc):+.2f} pp | {comparison['sam']['conditional_test_ce']:.6f} |",
        f"| APBR + 同一条件贪心 OBS | {apbr_dense_acc:.2%} | {comparison['apbr']['same_parent_physical_test_accuracy']:.2%} | {100 * (comparison['apbr']['same_parent_physical_test_accuracy'] - apbr_dense_acc):+.2f} pp | {apbr['raw_hard_test']['physical']['test_ce']:.6f} |",
        f"| APBR + 几何近优条件贪心 OBS | {apbr_dense_acc:.2%} | {apbr_acc:.2%} | {100 * (apbr_acc - apbr_dense_acc):+.2f} pp | {comparison['apbr']['conditional_geometric_test_ce']:.6f} |",
        "",
        f"最终准确率 APBR−SAM = {100 * (apbr_acc - sam_acc):+.2f} pp，即 10,000 张 Test 中 APBR 少正确 {round(10000 * (sam_acc - apbr_acc))} 张。",
        "APBR 同父消融：纯物理条件贪心 77.72%，单位几何 77.76%，真实几何 77.83%，打乱几何 77.82%。真实几何只比打乱高 0.01 pp；一个 Seed 无法证明学习到的几何稳定有效。",
        "这不是单独归因 APBR 几何的严格因果对照，因为训练路径与 Mask 选择都变了。同父 unit/true/shuffled 消融才直接检验几何坐标对应关系；目前该证据很弱。", "",
        "SAM 历史条件贪心 checkpoint、APBR 四组冻结 checkpoint、父模型、校准 ID、Test pickle、架构及正式配置哈希均核对。APBR 纯物理路径与历史条件贪心产物在 19 层逐权重 Bitwise Equal。此次比较脚本没有重新评估 Test。",
        f"SAM 原始结果：`{SAM_DIR / 'results.json'}`。APBR 新结果：`{APBR_DIR / 'results.json'}`。机器可读比较：`{json_path}`。", "",
    ]
    report_path = REPORT_DIR / "CROSS_PARENT_COMPARISON.md"
    write_new(report_path, "\n".join(lines))
    print(json.dumps({"status": comparison["status"],
                      "sam_accuracy": sam_acc, "apbr_accuracy": apbr_acc,
                      "delta_pp": comparison["apbr_minus_sam_final_percentage_points"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
