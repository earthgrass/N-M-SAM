"""Wait for the verified SAM Seed 3 Final, then run its CPU comparison once.

The remote queue and its transfer helper are independent processes. This
script never changes either process and never retries a failed experiment.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy"
OUT = ROOT / "artifacts/sam_conditional_seed3_autorun_20261104_v1"
RESULT = ROOT / "artifacts/sam_conditional_seed3_cpu_20261104_v1"
DOC = ROOT / "docs/research/sam_apbr_l2_conditional_seed3_20261104"
POLL_SECONDS = 60
MAX_WAIT_SECONDS = 12 * 3600


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def status(state: str, **extra) -> None:
    path = OUT / "status.json"
    payload = {"status": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temp = path.with_name("status.json.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def wait_for_parent() -> str:
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    proof_path = SYNC / "seed3/dense_sam/LOCAL_SHA256.json"
    checkpoint = proof_path.parent / "final_epoch_200.pt"
    while time.monotonic() < deadline:
        if proof_path.exists() and checkpoint.exists():
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            job = proof["job"]
            expected = job["checkpoint_sha256"]
            if (job["state"] != "completed" or job["seed"] != 3
                    or job["method"] != "dense_sam"
                    or proof["file_sha256"]["final_epoch_200.pt"] != expected
                    or sha(checkpoint) != expected):
                raise ValueError("SAM Seed 3 local transfer identity mismatch")
            return expected
        sync_status = SYNC / "sync_status.json"
        if sync_status.exists():
            synced = json.loads(sync_status.read_text(encoding="utf-8"))
            if synced.get("remote_status") == "failed":
                raise RuntimeError("Remote training queue failed before SAM Seed 3 transfer")
        time.sleep(POLL_SECONDS)
    raise TimeoutError("Verified SAM Seed 3 checkpoint was not transferred within 12 hours")


def run_once(script: str, log_name: str) -> None:
    with (OUT / log_name).open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts" / script), "--seed", "3"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"{script} exited {completed.returncode}; see {OUT / log_name}")


def create_comparison() -> None:
    sam = json.loads((RESULT / "results.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULT / "independent_audit.json").read_text(encoding="utf-8"))
    vpr = json.loads((ROOT / "artifacts/apbr_vpr_nearoptimal_conditional_seed3_full_cpu_20261104_v1/results.json").read_text(encoding="utf-8"))
    l2 = json.loads((ROOT / "artifacts/apbr_l2_nearoptimal_conditional_seed3_full_cpu_20261104_v1/results.json").read_text(encoding="utf-8"))
    if (sam["status"] != "COMPLETE_CPU_EXPLORATORY_SEED3"
            or audit["status"] != "INDEPENDENT_ARTIFACT_AUDIT_PASS"
            or vpr["status"] != "COMPLETE_CPU_EXPLORATORY_SEED3"
            or l2["status"] != "COMPLETE_CPU_EXPLORATORY_SEED3"):
        raise ValueError("Seed 3 result or audit status incomplete")
    acc = lambda entry: f"{100 * entry['test_accuracy']:.2f}%"
    rows = [
        ("SAM", sam["test"]["dense"], sam["test"]["physical"], None, None),
        ("原版 APBR", vpr["raw_hard_test"]["dense"], vpr["raw_hard_test"]["physical"], vpr["raw_hard_test"]["true"], vpr["raw_hard_test"]["shuffled"]),
        ("APBR-L2", l2["raw_hard_test"]["dense"], l2["raw_hard_test"]["physical"], l2["raw_hard_test"]["true"], l2["raw_hard_test"]["shuffled"]),
    ]
    lines = [
        "# Seed 3：SAM、原版 APBR、APBR-L2 条件贪心 OBS 独立比较", "",
        "三路从相同初始化出发，各自训练 200 Epoch。使用同一官方 CIFAR-100 训练校准配置和固定 2:4 规则；补偿从各自 Dense Final 原始权重开始。没有 Recovery 或 BN-only。Mask 和 checkpoint 冻结后，各自进行一次固定 Test。", "",
        "| 路径 | Dense Final | 纯物理条件贪心 OBS | 真实几何近优裁决 + OBS | 打乱几何 + OBS |", "|---|---:|---:|---:|---:|",
    ]
    for name, dense, physical, true, shuffled in rows:
        lines.append(f"| {name} | {acc(dense)} | {acc(physical)} | {acc(true) if true else '不适用'} | {acc(shuffled) if shuffled else '不适用'} |")
    sam_physical = 100 * sam["test"]["physical"]["test_accuracy"]
    vpr_true = 100 * vpr["raw_hard_test"]["true"]["test_accuracy"]
    l2_true = 100 * l2["raw_hard_test"]["true"]["test_accuracy"]
    lines += ["", f"相对于 SAM 纯物理条件贪心，原版 APBR 真实几何为 {vpr_true - sam_physical:+.2f} pp，APBR-L2 真实几何为 {l2_true - sam_physical:+.2f} pp。这些是完整训练路径的结果，不能将跨父模型差异单独归因于几何裁决。", "",
              f"SAM 父 checkpoint SHA256：`{sam['parent_sha256']}`。独立审计：`{audit['status']}`；19 层严格 2:4，5,578,752 个删除坐标。对应 APBR 和 APBR-L2 审计与哈希见各自结果目录。", "",
              "证据：[SAM 结果](../../../artifacts/sam_conditional_seed3_cpu_20261104_v1/results.json)、[SAM 审计](../../../artifacts/sam_conditional_seed3_cpu_20261104_v1/independent_audit.json)、[原版 APBR 结果](../../../artifacts/apbr_vpr_nearoptimal_conditional_seed3_full_cpu_20261104_v1/results.json)、[APBR-L2 结果](../../../artifacts/apbr_l2_nearoptimal_conditional_seed3_full_cpu_20261104_v1/results.json)。", ""]
    write_new(DOC / "COMPARISON.md", "\n".join(lines))
    paths = [DOC / "COMPARISON.md", *RESULT.glob("*"), OUT / "run.log", OUT / "audit.log"]
    paths = sorted((path for path in paths if path.is_file()), key=str)
    manifest = "".join(f"{sha(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths)
    write_new(DOC / "SHA256SUMS_final.txt", manifest)
    for path in paths:
        if sha(path) not in manifest:
            raise ValueError(f"Manifest verification failed: {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        status("WAITING_FOR_VERIFIED_SAM_SEED3")
        parent = wait_for_parent()
        if RESULT.exists():
            raise FileExistsError(f"SAM Seed 3 result directory already exists: {RESULT}")
        status("RUNNING_CPU_POSTPROCESS", parent_sha256=parent)
        run_once("run_sam_conditional_seed23_cpu.py", "run.log")
        status("RUNNING_INDEPENDENT_AUDIT", parent_sha256=parent)
        run_once("audit_sam_conditional_seed23_cpu.py", "audit.log")
        create_comparison()
        status("COMPLETE_SEED3_VERIFIED", parent_sha256=parent,
               report=str(DOC / "COMPARISON.md"))
    except Exception:
        status("FAILED_NO_RETRY", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
