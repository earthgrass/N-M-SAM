"""Wait for SAM-GPT Seed 0, run matched pruning controls and fixed Final Test."""
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
    ("dynamic_groupwise_obs_no_probe", "group_obs", None),
    ("dynamic_sam_gpt_true_probe", "dynamic_sam_gpt", None),
    ("dynamic_sam_gpt_shuffled_probe", "dynamic_sam_gpt", 20261031),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict | None:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError): return None


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"Stage failed: {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-asset", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve(); project = args.project_root.resolve(); training = root / "sam_gpt_joint"
    completed = []
    while True:
        state = read(training / "status.json")
        if state and state.get("status") == "completed":
            if state.get("completed_epochs") != 200 or state.get("steps") != 78_200 or state.get("test_accessed") is not False:
                raise RuntimeError("SAM-GPT completion contract failed")
            completed.append({"stage": "sam_gpt_dense", "checkpoint_sha256": state["final_checkpoint_sha256"]})
            break
        try: os.kill(args.training_pid, 0)
        except ProcessLookupError: raise RuntimeError("SAM-GPT training exited before completion")
        atomic(root / "queue_status.json", {"status": "RUNNING", "phase": "sam_gpt_training",
            "completed_epochs": (state or {}).get("completed_epochs", 0), "completed": completed,
            "test_accessed": False})
        time.sleep(60)
    for _ in range(120):
        busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        if not busy: break
        time.sleep(1)
    else: raise RuntimeError("Training did not release GPU")
    parent = training / "final_epoch_200.pt"
    for endpoint, selector, shuffle_seed in ENDPOINTS:
        target = root / "pruning" / endpoint
        atomic(root / "queue_status.json", {"status": "RUNNING", "phase": f"pruning:{endpoint}",
            "completed": completed, "test_accessed": False})
        command = [sys.executable, str(project / "scripts/run_apbr_gpt_joint_matrix_prune.py"),
            "--dataset-root", str(args.dataset_root), "--config", str(args.config),
            "--parent", str(parent), "--initial-asset", str(args.initial_asset),
            "--output-dir", str(target), "--endpoint", endpoint, "--selector", selector]
        if shuffle_seed is not None: command += ["--shuffle-seed", str(shuffle_seed)]
        run(command, root / "logs" / f"{endpoint}_prune.log")
        state = read(target / "status.json")
        if not state or state.get("status") != "completed" or state.get("strict_2_4") is not True:
            raise RuntimeError(f"Pruning audit failed: {endpoint}")
        completed.append({"stage": endpoint, "checkpoint_sha256": state["raw_hard_sha256"],
                          "mask_sha256": state["mask_sha256"]})
    frozen = {name: {"raw_hard_sha256": sha(root / "pruning" / name / "raw_hard.pt"),
                     "mask_sha256": read(root / "pruning" / name / "status.json")["mask_sha256"]}
              for name, _selector, _seed in ENDPOINTS}
    atomic(root / "pruning_checkpoint_freeze.json", {"status": "FOUR_PRUNING_ENDPOINTS_FROZEN",
        "sam_gpt_parent_sha256": sha(parent), "endpoints": frozen, "test_accessed": False})
    evaluations = root / "final_test"; evaluations.mkdir(parents=True, exist_ok=False)
    targets = [("sam_gpt_dense", parent, "dense_final")]
    targets += [(name, root / "pruning" / name / "raw_hard.pt", "raw_hard") for name, _s, _r in ENDPOINTS]
    for name, checkpoint, role in targets:
        run([sys.executable, str(project / "scripts/evaluate_cifar100_50k_final.py"),
            "--dataset-root", str(args.dataset_root), "--config", str(args.config),
            "--checkpoint", str(checkpoint), "--checkpoint-role", role,
            "--output", str(evaluations / f"{name}.json")], root / "logs" / f"{name}_test.log")
    results = {name: read(evaluations / f"{name}.json") for name, _checkpoint, _role in targets}
    if not all(row and row.get("status") == "PASS" and row.get("test_count") == 10_000
               for row in results.values()): raise RuntimeError("Final Test audit failed")
    atomic(evaluations / "FINAL_TEST_RESULTS.json", {"status": "FINAL_TEST_COMPLETE",
        "evaluations": results, "evaluation_count": 5, "each_checkpoint_evaluated_once": True,
        "selection_performed": False, "hyperparameter_change_performed": False,
        "recovery_started": False})
    atomic(root / "queue_status.json", {"status": "COMPLETE", "phase": "final_test_complete",
        "completed": completed, "final_test_evaluations": 5, "recovery_started": False,
        "test_accessed": True})
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as error:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1])
            atomic(root / "queue_failure.json", {"status": "FAILED", "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
