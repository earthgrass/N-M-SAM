"""Fail-closed single-GPU Seed-0 Dense/Raw/Recovery APBR-GPT matrix queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = "cifar100_50k10k_final_v1"
ENDPOINTS = (
    ("dense_sam_original_gpt", "dense_sam", "original_gpt", None),
    ("apbr_sam_original_gpt", "apbr_sam", "original_gpt", None),
    ("apbr_sam_dynamic_apbr_gpt", "apbr_sam", "dynamic_apbr_gpt", None),
    ("joint_apbr_gpt_original_gpt", "joint_apbr_gpt", "original_gpt", None),
    ("joint_apbr_gpt_dynamic_apbr_gpt", "joint_apbr_gpt", "dynamic_apbr_gpt", None),
    ("joint_apbr_gpt_shuffled_geometry", "joint_apbr_gpt", "dynamic_apbr_gpt", 20261031),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError): return None


def status(root: Path, phase: str, completed: list, **extra) -> None:
    atomic_json(root / "queue_status.json", {"status": "RUNNING", "protocol_id": PROTOCOL,
        "phase": phase, "completed": completed, "test_accessed": False, **extra})


def run(command: list[str], log: Path) -> None:
    if subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip():
        raise RuntimeError("GPU occupied before serial stage launch")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Stage failed with exit code {result.returncode}: {' '.join(command)}")


def wait_external_training(root: Path, directory: Path, pid: int) -> dict:
    while True:
        record = read_json(directory / "status.json")
        if record and record.get("status") == "failed": raise RuntimeError(f"Dense stage failed: {record}")
        if record and record.get("status") == "completed":
            if record.get("completed_epochs") != 200 or record.get("steps") != 78_200 or record.get("test_accessed") is not False:
                raise RuntimeError("Dense completion contract failed")
            for _ in range(120):
                occupied = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                          capture_output=True, text=True).stdout.strip()
                if not occupied: break
                time.sleep(1)
            else: raise RuntimeError("Dense published completion but did not release the GPU")
            return record
        try: os.kill(pid, 0)
        except ProcessLookupError: raise RuntimeError("Dense process exited before completion")
        status(root, "dense_sam", [], pid=pid, completed_epochs=(record or {}).get("completed_epochs", 0))
        time.sleep(60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vpr-src", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--initial-asset", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dense-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve(); project = args.project_root.resolve(); python = sys.executable
    dense = root / "dense_sam"; apbr = root / "apbr_sam"; joint = root / "joint_apbr_gpt"
    completed = []
    dense_record = wait_external_training(root, dense, args.dense_pid)
    completed.append({"stage": "dense_sam", "checkpoint_sha256": dense_record["final_checkpoint_sha256"]})

    status(root, "apbr_sam", completed)
    run([python, str(project / "scripts/run_cifar100_50k_formal_training.py"),
         "--dataset-root", str(args.dataset_root), "--vpr-src", str(args.vpr_src),
         "--config", str(args.config), "--initial-asset", str(args.initial_asset),
         "--output-dir", str(apbr), "--method", "vpr", "--seed", "0"], root / "logs/apbr_sam_console.log")
    apbr_record = read_json(apbr / "status.json")
    if not apbr_record or apbr_record.get("status") != "completed" or apbr_record.get("steps") != 78_200:
        raise RuntimeError("APBR completion contract failed")
    completed.append({"stage": "apbr_sam", "checkpoint_sha256": apbr_record["final_checkpoint_sha256"]})

    status(root, "joint_apbr_gpt", completed)
    run([python, str(project / "scripts/run_apbr_gpt_joint_training_seed0.py"),
         "--protocol-config", str(args.config), "--joint-config", str(args.joint_config),
         "--initial-asset", str(args.initial_asset), "--calibration-dir", str(args.calibration_dir),
         "--data-root", str(args.dataset_root), "--output-dir", str(joint)], root / "logs/joint_apbr_gpt_console.log")
    joint_checkpoint = joint / "final_epoch_200.pt"
    joint_record = torch_load_metadata(joint_checkpoint)
    if joint_record.get("epoch") != 200 or joint_record.get("stepper", {}).get("steps") != 78_200:
        raise RuntimeError("Joint completion contract failed")
    completed.append({"stage": "joint_apbr_gpt", "checkpoint_sha256": sha(joint_checkpoint)})

    parents = {"dense_sam": dense / "final_epoch_200.pt", "apbr_sam": apbr / "final_epoch_200.pt",
               "joint_apbr_gpt": joint_checkpoint}
    for endpoint, parent_name, selector, shuffle_seed in ENDPOINTS:
        target = root / "methods" / endpoint
        status(root, f"prune:{endpoint}", completed)
        command = [python, str(project / "scripts/run_apbr_gpt_joint_matrix_prune.py"),
            "--dataset-root", str(args.dataset_root), "--config", str(args.config),
            "--parent", str(parents[parent_name]), "--initial-asset", str(args.initial_asset),
            "--output-dir", str(target), "--endpoint", endpoint, "--selector", selector]
        if shuffle_seed is not None: command += ["--shuffle-seed", str(shuffle_seed)]
        run(command, root / "logs" / f"{endpoint}_prune.log")
        record = read_json(target / "status.json")
        if not record or record.get("status") != "completed" or record.get("strict_2_4") is not True:
            raise RuntimeError(f"Pruning contract failed: {endpoint}")
        completed.append({"stage": f"raw:{endpoint}", "checkpoint_sha256": record["raw_hard_sha256"]})

        status(root, f"recovery:{endpoint}", completed)
        run([python, str(project / "scripts/run_apbr_gpt_joint_fixedmask_recovery.py"),
             "--dataset-root", str(args.dataset_root), "--config", str(args.config),
             "--raw-checkpoint", str(target / "raw_hard.pt"),
             "--output-dir", str(target / "recovery")], root / "logs" / f"{endpoint}_recovery.log")
        recovery = read_json(target / "recovery/status.json")
        if not recovery or recovery.get("status") != "completed" or recovery.get("recovery_steps") != 7820:
            raise RuntimeError(f"Recovery contract failed: {endpoint}")
        completed.append({"stage": f"recovery:{endpoint}",
                          "checkpoint_sha256": recovery["recovery_final_sha256"]})

    targets = [("dense_sam_final", dense / "final_epoch_200.pt", "dense_final"),
               ("apbr_sam_final", apbr / "final_epoch_200.pt", "dense_final"),
               ("joint_apbr_gpt_final", joint_checkpoint, "dense_final")]
    for endpoint, _parent, _selector, _shuffle in ENDPOINTS:
        directory = root / "methods" / endpoint
        targets += [(f"{endpoint}_raw", directory / "raw_hard.pt", "raw_hard"),
                    (f"{endpoint}_recovery20", directory / "recovery/final_epoch_020.pt", "recovery_final")]
    freeze = {name: {"path": str(path), "sha256": sha(path), "role": role} for name, path, role in targets}
    atomic_json(root / "checkpoint_freeze_before_test.json", {"status": "ALL_ENDPOINTS_FROZEN",
        "protocol_id": PROTOCOL, "targets": freeze, "test_accessed": False})

    results = root / "final_test"; results.mkdir(parents=True, exist_ok=False)
    for name, checkpoint, role in targets:
        status(root, f"final_test:{name}", completed, test_accessed=True)
        run([python, str(project / "scripts/evaluate_cifar100_50k_final.py"),
             "--dataset-root", str(args.dataset_root), "--config", str(args.config),
             "--checkpoint", str(checkpoint), "--checkpoint-role", role,
             "--output", str(results / f"{name}.json")], root / "logs" / f"{name}_test.log")
    summary = {name: read_json(results / f"{name}.json") for name, _path, _role in targets}
    atomic_json(root / "final_endpoint_metrics.json", summary)
    atomic_json(root / "queue_status.json", {"status": "COMPLETE", "protocol_id": PROTOCOL,
        "completed": completed, "dense_endpoints": 3, "raw_endpoints": 6, "recovery20_endpoints": 6,
        "final_test_targets": 15, "test_accessed": True, "finished_at_unix": time.time()})
    return 0


def torch_load_metadata(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as error:
        if "--output-root" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--output-root") + 1])
            atomic_json(target / "queue_failure.json", {"status": "FAILED", "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(), "automatic_retry": False,
                "failed_at_unix": time.time()})
        raise
