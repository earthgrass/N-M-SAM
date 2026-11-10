"""Build and independently verify the Phase-3 rebased server bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite server_bundle_v3")
    output.mkdir(parents=True)

    files = [
        "baseline_adapters/__init__.py", "baseline_adapters/cifar100_model.py",
        "baseline_adapters/apbr_gpt_joint/__init__.py", "baseline_adapters/apbr_gpt_joint/adapter.py",
        "baseline_adapters/common/__init__.py", "baseline_adapters/common/conv2d.py",
        "baseline_adapters/common/nm.py", "benchmark/__init__.py", "benchmark/model.py", "benchmark/nm.py",
        "joint_optimization/__init__.py", "joint_optimization/training_objective.py",
        "joint_optimization/hessian_cache.py", "joint_optimization/joint_stepper.py",
        "joint_optimization/orchestration.py",
        "joint_optimization/calibration/__init__.py", "joint_optimization/calibration/lambda_calibrator.py",
        "joint_optimization/calibration/eta_scale_calibrator.py",
        "joint_optimization/calibration/calibration_report.py",
        "scripts/run_apbr_gpt_joint_training_seed0.py", "scripts/run_apbr_gpt_joint_pruning_seed0.py",
        "scripts/run_apbr_gpt_joint_cpu_smoke.py", "scripts/run_apbr_gpt_joint_real_cpu_calibration.py",
        "scripts/run_cifar100_50k_formal_training.py", "scripts/evaluate_cifar100_50k_final.py",
        "tests/conftest.py", "tests/test_apbr_gpt_joint.py", "tests/test_joint_calibration.py",
        "tests/test_joint_orchestration.py",
    ]
    for name in files:
        copy(ROOT / name, output / "source" / "project" / name)
    shutil.copytree(PROJECT / "src" / "vpr_sam", output / "source" / "vpr_src" / "vpr_sam")
    copy(ROOT / "configs" / "cifar100_50k10k_final_v1.yaml", output / "config" / "cifar100_50k10k_final_v1.yaml")
    copy(ROOT / "configs" / "apbr_gpt_joint_seed0_calibration.json", output / "config" / "apbr_gpt_joint_seed0_calibration.json")
    copy(ROOT / "artifacts" / "apbr_gpt_joint_seed0_rebased" / "assets" / "initial_common_seed0.pt",
         output / "model_assets" / "initial_common_seed0.pt")
    for name in ("lambda_calibration.json", "eta_s_calibration.json", "calibration_identity.json",
                 "hessian_refresh_log.csv"):
        copy(ROOT / "artifacts" / "apbr_gpt_joint_seed0_rebased" / "real_cpu_calibration_attempt3" / name,
             output / "calibration" / name)
    copy(ROOT / "artifacts" / "apbr_gpt_joint_seed0" / "server_bundle" / "data" /
         "fixed_training_calibration_batch.json", output / "data" / "fixed_training_calibration_batch.json")
    copy(ROOT / "artifacts" / "apbr_gpt_joint_seed0" / "server_bundle" / "environment" /
         "requirements.txt", output / "environment" / "requirements.txt")

    (output / "data" / "DATA_CONTRACT.md").write_text(
        "Official CIFAR-100 train archive only during training/calibration. Verify size 169001437, "
        "MD5 eb9058c3a382ffc7106e4002c42a8d85 and SHA256 "
        "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7. "
        "The bundle intentionally contains no dataset and no Test samples.\n", encoding="utf-8")
    bins = output / "bin"; bins.mkdir()
    scripts = {
        "verify_bundle.sh": """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha256sum -c SHA256SUMS.txt
test -f model_assets/initial_common_seed0.pt
test -f calibration/calibration_identity.json
! grep -R -I -E '([A-Za-z]:\\\\Users\\\\|BEGIN (OPENSSH|RSA) PRIVATE KEY|ssh -p )' .
""",
        "run_cpu_preflight.sh": """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/source/project:$PWD/source/vpr_src"
python -m pytest -q source/project/tests/test_apbr_gpt_joint.py source/project/tests/test_joint_calibration.py source/project/tests/test_joint_orchestration.py
""",
        "run_gpu_smoke.sh": """#!/usr/bin/env bash
set -euo pipefail
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all()
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
PY
echo 'Environment smoke passed. Run one formal batch only under an explicit GPU-smoke authorization.'
""",
        "run_joint_training_seed0.sh": """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; export PYTHONPATH="$PWD/source/project:$PWD/source/vpr_src"
python source/project/scripts/run_apbr_gpt_joint_training_seed0.py --protocol-config config/cifar100_50k10k_final_v1.yaml --joint-config config/apbr_gpt_joint_seed0_calibration.json --initial-asset model_assets/initial_common_seed0.pt --calibration-dir calibration --data-root "${CIFAR100_ROOT:?set CIFAR100_ROOT}" --output-dir "${OUTPUT_ROOT:?set OUTPUT_ROOT}/joint_training_seed0"
""",
        "run_joint_pruning_seed0.sh": """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; export PYTHONPATH="$PWD/source/project:$PWD/source/vpr_src"
python source/project/scripts/run_apbr_gpt_joint_pruning_seed0.py --training-checkpoint "${OUTPUT_ROOT:?set OUTPUT_ROOT}/joint_training_seed0/final_epoch_200.pt" --initial-asset model_assets/initial_common_seed0.pt --calibration-identity calibration/calibration_identity.json --train-pickle "${CIFAR100_ROOT:?set CIFAR100_ROOT}/extracted/cifar-100-python/train" --output-dir "$OUTPUT_ROOT/joint_pruning_seed0"
""",
    }
    for name, value in scripts.items():
        path = bins / name; path.write_text(value, encoding="utf-8", newline="\n")

    readme = """# APBR-GPT Joint Seed 0 rebased bundle v3

This is the isolated `apbr_gpt_joint_seed0_rebased` deployment bundle. It contains one common initial checkpoint for Dense-SAM, APBR-SAM and Joint APBR-GPT; this bundle launches only the Joint branch. Historical checkpoints are excluded.

Run `bin/verify_bundle.sh`, then `bin/run_cpu_preflight.sh`, then the environment-only `bin/run_gpu_smoke.sh`. Formal training and pruning require explicit user authorization and the `CIFAR100_ROOT` and `OUTPUT_ROOT` environment variables. Training does not construct a Test loader. Pruning does not evaluate Test. Final evaluation is a separate guarded action after checkpoint hashes are frozen.

Stop on any identity mismatch, non-finite value, OOM, Cholesky failure, missing Hessian cache, source/config drift, or step-budget mismatch. No automatic retry or parameter adjustment is permitted.
"""
    (output / "BUNDLE_README.md").write_text(readme, encoding="utf-8")

    members = sorted(path for path in output.rglob("*") if path.is_file())
    manifest_lines = [f"{digest(path)}  {path.relative_to(output).as_posix()}" for path in members]
    (output / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    # Verify every entry before archiving.
    failures = 0
    for line in manifest_lines:
        expected, name = line.split("  ", 1)
        failures += digest(output / name) != expected
    if failures:
        raise RuntimeError(f"Bundle SHA256 failures: {failures}")

    archive = output.parent / "apbr_gpt_joint_seed0_server_bundle_v3.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(output, arcname="server_bundle_v3")
    # A short system temporary path avoids Windows MAX_PATH failures while
    # still providing a genuinely independent archive extraction.
    with tempfile.TemporaryDirectory(prefix="apbr_v3_") as temporary:
        verification = Path(temporary)
        with tarfile.open(archive, "r:gz") as stream:
            stream.extractall(verification, filter="data")
        extracted = verification / "server_bundle_v3"
        extract_failures = sum(digest(extracted / line.split("  ",1)[1]) != line.split("  ",1)[0]
                               for line in manifest_lines)
    status = {"status": "SERVER_BUNDLE_V3_READY" if not extract_failures else "FAILED",
        "files_in_sha256_manifest": len(manifest_lines), "sha256_failures_before_archive": failures,
        "sha256_failures_after_extract": extract_failures, "archive": archive.name,
        "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
        "contains_test_data": False, "contains_credentials": False}
    (output.parent / "BUNDLE_V3_INTEGRITY.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
