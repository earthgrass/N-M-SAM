"""Read-only two-seed comparison of fixed conditional OBS endpoints."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs/research/sam_apbr_l2_conditional_seed01_20261103"
CALIB_SHA = "3b7da27d79762235c87f18571bc73224d6e6f2b3ba663b7d9a4a3de312f23157"
TEST_SHA = "4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def entry(dense: float, test: dict, parent_sha: str, checkpoint_sha: str) -> dict:
    if test["test_count"] != 10000:
        raise ValueError("Test count mismatch")
    acc = test["test_accuracy"] * 100
    return {"dense_accuracy_percent": dense * 100,
            "accuracy_percent": acc, "ce": test["test_ce"],
            "drop_pp": acc - dense * 100,
            "parent_sha256": parent_sha, "checkpoint_sha256": checkpoint_sha}


def main() -> None:
    sam0_path = ART / "sam_singleton_conditional_seed0_cpu_20261103/results.json"
    vpr0_path = ART / "apbr_nearoptimal_conditional_seed0_full_cpu_20261103_v1/results.json"
    l20_path = ART / "apbr_l2_nearoptimal_conditional_seed0_full_cpu_20261103_v1/results.json"
    seed1_path = ART / "sam_apbr_conditional_seed1_cpu_20261103_v1/results.json"
    l21_path = ART / "apbr_l2_nearoptimal_conditional_seed1_full_cpu_20261103_v1/results.json"
    l20audit_path = ART / "apbr_l2_nearoptimal_conditional_seed0_full_cpu_20261103_v1/independent_audit.json"
    l21audit_path = ART / "apbr_l2_nearoptimal_conditional_seed1_full_cpu_20261103_v1/independent_audit.json"
    paths = [sam0_path, vpr0_path, l20_path, seed1_path, l21_path,
             l20audit_path, l21audit_path]
    sam0, vpr0, l20, seed1, l21, l20audit, l21audit = map(read, paths)
    for result in (sam0, vpr0, l20, seed1, l21):
        if result["calibration_ids_sha256"] != CALIB_SHA:
            raise ValueError("Calibration ID mismatch")
    if (sam0["test_pickle_sha256"] != TEST_SHA or seed1["test_pickle_sha256"] != TEST_SHA
            or l20audit["status"] != "INDEPENDENT_ARTIFACT_AUDIT_PASS"
            or l21audit["status"] != "INDEPENDENT_ARTIFACT_AUDIT_PASS"
            or l21["status"] != "COMPLETE_CPU_EXPLORATORY_SEED1"):
        raise ValueError("Historical Test or independent audit mismatch")
    cross = read(ROOT / "docs/research/apbr_nearoptimal_conditional_seed0_20261103_v2/CROSS_PARENT_COMPARISON.json")
    if cross["same_test_pickle_sha256"] != TEST_SHA:
        raise ValueError("Historical Test identity mismatch")
    dense_l20 = read(ART / "apbr_fixed_l2_seed0_20261102/remote_final_copy/artifacts/run_v1/evaluations/dense.json")
    if dense_l20["checkpoint_sha256"] != l20["parent_sha256"]:
        raise ValueError("L2 Seed0 Dense identity mismatch")
    rows = {
        "sam_physical": {
            "0": entry(cross["sam"]["dense_test_accuracy"], sam0["test"]["singleton_conditional_joint"],
                       sam0["parent_sha256"], sam0["checkpoint_sha256"]["singleton_conditional_joint"]),
            "1": entry(seed1["methods"]["sam_conditional"]["parent_dense_test_accuracy"],
                       seed1["test"]["sam_conditional"],
                       seed1["methods"]["sam_conditional"]["parent_sha256"],
                       seed1["methods"]["sam_conditional"]["checkpoint_sha256"]),
        },
        "apbr_vpr_geometric": {
            "0": entry(cross["apbr"]["dense_test_accuracy"], vpr0["raw_hard_test"]["true"],
                       vpr0["parent_sha256"], vpr0["checkpoint_sha256"]["true"]),
            "1": entry(seed1["methods"]["apbr_geo_conditional"]["parent_dense_test_accuracy"],
                       seed1["test"]["apbr_geo_conditional"],
                       seed1["methods"]["apbr_geo_conditional"]["parent_sha256"],
                       seed1["methods"]["apbr_geo_conditional"]["checkpoint_sha256"]),
        },
        "apbr_l2_physical": {
            "0": entry(dense_l20["test_accuracy"], l20["raw_hard_test"]["physical"],
                       l20["parent_sha256"], l20["checkpoint_sha256"]["physical"]),
            "1": entry(l21["raw_hard_test"]["dense"]["test_accuracy"],
                       l21["raw_hard_test"]["physical"],
                       l21["parent_sha256"], l21["checkpoint_sha256"]["physical"]),
        },
        "apbr_l2_geometric": {
            "0": entry(dense_l20["test_accuracy"], l20["raw_hard_test"]["true"],
                       l20["parent_sha256"], l20["checkpoint_sha256"]["true"]),
            "1": entry(l21["raw_hard_test"]["dense"]["test_accuracy"],
                       l21["raw_hard_test"]["true"],
                       l21["parent_sha256"], l21["checkpoint_sha256"]["true"]),
        },
    }
    for key in rows:
        values = [rows[key][seed]["accuracy_percent"] for seed in ("0", "1")]
        rows[key]["mean_accuracy_percent"] = statistics.mean(values)
        rows[key]["sample_sd_accuracy_percent"] = statistics.stdev(values)
    paired = {}
    for label, a, b in (
        ("vpr_geometric_minus_sam", "apbr_vpr_geometric", "sam_physical"),
        ("l2_geometric_minus_sam", "apbr_l2_geometric", "sam_physical"),
        ("l2_physical_minus_sam", "apbr_l2_physical", "sam_physical"),
        ("l2_geometric_minus_l2_physical", "apbr_l2_geometric", "apbr_l2_physical"),
        ("l2_geometric_minus_vpr_geometric", "apbr_l2_geometric", "apbr_vpr_geometric"),
    ):
        values = [rows[a][seed]["accuracy_percent"] - rows[b][seed]["accuracy_percent"]
                  for seed in ("0", "1")]
        paired[label] = {"seed0_pp": values[0], "seed1_pp": values[1],
                         "mean_pp": statistics.mean(values), "sample_sd_pp": statistics.stdev(values)}
    evidence = {"status": "TWO_SEED_READ_ONLY_COMPARISON_PASS",
                "exploratory": True, "no_recovery": True, "no_test_re_evaluation": True,
                "calibration_ids_sha256": CALIB_SHA, "test_pickle_sha256": TEST_SHA,
                "rows": rows, "paired_deltas": paired,
                "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths}}
    write_new(OUT / "comparison.json", json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    labels = {"sam_physical": "SAM + 条件贪心 OBS",
              "apbr_vpr_geometric": "原版 APBR + 几何近优 OBS",
              "apbr_l2_physical": "APBR-L2 + 条件贪心 OBS",
              "apbr_l2_geometric": "APBR-L2 + 几何近优 OBS"}
    lines = ["# SAM / APBR / APBR-L2：条件贪心 OBS Seed 0–1",
             "", "同一 CIFAR-100 训练校准 ID、19 层严格 2:4、相同物理联合 OBS；均无 Recovery。以下是两个 Seed 的探索性结果，不是正式多 Seed 结论。",
             "", "| 路径 | Seed 0 | Seed 1 | 均值 ± 样本标准差 |",
             "|---|---:|---:|---:|"]
    for key, label in labels.items():
        row = rows[key]
        lines.append(f"| {label} | {row['0']['accuracy_percent']:.2f}% | {row['1']['accuracy_percent']:.2f}% | {row['mean_accuracy_percent']:.2f} ± {row['sample_sd_accuracy_percent']:.2f}% |")
    lines += ["", "| 同 Seed 配对差值 | Seed 0 | Seed 1 | 均值 |",
              "|---|---:|---:|---:|"]
    for label, result in paired.items():
        lines.append(f"| {label} | {result['seed0_pp']:+.2f} pp | {result['seed1_pp']:+.2f} pp | {result['mean_pp']:+.2f} pp |")
    lines += ["", "两 Seed 样本很小；跨 SAM/VPR/L2 比较同时改变了稠密训练和部分 Mask 选择，不能把配对差值归因于单一机制。L2 的物理与几何路径为同父对照。Test 结果仅从既有一次性评估记录读取，未重新评估或选参。",
              "", "机器可读结果及各检查点 SHA256 见 `comparison.json`。"]
    write_new(OUT / "COMPARISON.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": evidence["status"], "paired": paired}, ensure_ascii=False))


if __name__ == "__main__":
    main()
