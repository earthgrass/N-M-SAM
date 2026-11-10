"""Wait for Joint Seed 0, run four frozen pruning endpoints, then stop before Test."""
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

ENDPOINTS = (
    ("original_sparsegpt", "original_gpt", None),
    ("dynamic_groupwise_obs_no_geometry", "group_obs", None),
    ("dynamic_apbr_gpt_true_geometry", "dynamic_apbr_gpt", None),
    ("dynamic_apbr_gpt_shuffled_geometry", "dynamic_apbr_gpt", 20261031),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict | None:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError): return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-asset", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--joint-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve(); joint = root / "joint_apbr_gpt"; completed = []
    while True:
        state = read(joint / "status.json")
        if state and state.get("status") == "completed":
            if state.get("completed_epochs") != 200 or state.get("steps") != 78_200 or state.get("test_accessed") is not False:
                raise RuntimeError("Joint training completion contract failed")
            completed.append({"stage": "joint_dense", "checkpoint_sha256": state["final_checkpoint_sha256"]})
            break
        try: os.kill(args.joint_pid, 0)
        except ProcessLookupError: raise RuntimeError("Joint training exited before completion")
        atomic(root / "queue_status.json", {"status": "RUNNING", "phase": "joint_apbr_gpt_training",
            "completed_epochs": (state or {}).get("completed_epochs", 0), "completed": completed,
            "scope": "joint_seed0_only", "test_accessed": False})
        time.sleep(60)
    for _ in range(120):
        busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        if not busy: break
        time.sleep(1)
    else: raise RuntimeError("Joint training did not release GPU")

    parent = joint / "final_epoch_200.pt"; project = args.project_root.resolve()
    for endpoint, selector, shuffle_seed in ENDPOINTS:
        target = root / "pruning" / endpoint
        atomic(root / "queue_status.json", {"status": "RUNNING", "phase": f"pruning:{endpoint}",
            "completed": completed, "scope": "joint_seed0_only", "test_accessed": False})
        command = [sys.executable, str(project / "scripts/run_apbr_gpt_joint_matrix_prune.py"),
            "--dataset-root", str(args.dataset_root), "--config", str(args.config),
            "--parent", str(parent), "--initial-asset", str(args.initial_asset),
            "--output-dir", str(target), "--endpoint", endpoint, "--selector", selector]
        if shuffle_seed is not None: command += ["--shuffle-seed", str(shuffle_seed)]
        log = root / "logs" / f"{endpoint}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f"Pruning failed: {endpoint}")
        state = read(target / "status.json")
        if not state or state.get("status") != "completed" or state.get("strict_2_4") is not True:
            raise RuntimeError(f"Pruning audit failed: {endpoint}")
        completed.append({"stage": endpoint, "checkpoint_sha256": state["raw_hard_sha256"],
                          "mask_sha256": state["mask_sha256"]})
    frozen = {name: {"raw_hard_sha256": sha(root / "pruning" / name / "raw_hard.pt"),
                     "mask_sha256": read(root / "pruning" / name / "status.json")["mask_sha256"]}
              for name, _selector, _seed in ENDPOINTS}
    atomic(root / "pruning_checkpoint_freeze.json", {"status": "FOUR_PRUNING_ENDPOINTS_FROZEN",
        "joint_parent_sha256": sha(parent), "endpoints": frozen, "test_accessed": False})
    atomic(root / "queue_status.json", {"status": "PRUNING_COMPLETE_AWAITING_TEST_AUTHORIZATION",
        "phase": "stopped_before_test", "completed": completed, "scope": "joint_seed0_only",
        "recovery_started": False, "test_accessed": False})
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as error:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1])
            atomic(root / "queue_failure.json", {"status": "FAILED", "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(), "automatic_retry": False,
                "test_accessed": False})
        raise
