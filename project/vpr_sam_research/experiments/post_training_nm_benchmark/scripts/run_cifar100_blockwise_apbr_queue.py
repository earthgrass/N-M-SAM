"""Fail-closed serial controller for the frozen six-run blockwise APBR matrix."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENT = "apbr_blockwise_perturbation_cifar100_20261028"
HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parents[2]
DATA = BASE / "data"
VPR_SRC = HERE.parents[1] / "src"
CONFIG = HERE / "configs/apbr_blockwise_perturbation_cifar100_20261028.yaml"
OUTPUT = BASE / "artifacts/formal"
STATUS = BASE / "artifacts/queue_status.json"
JOBS = [(seed, scheme) for seed in (0, 1, 2)
        for scheme in ("global_block_feedback", "equal_block_energy")]


def atomic_status(value: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if STATUS.exists() or any(OUTPUT.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing queue/formal output")
    state = {"status": "running", "experiment_id": EXPERIMENT, "jobs": [],
             "current": None, "started_at_unix": time.time(), "automatic_retry": False,
             "test_accessed": False}
    atomic_status(state)
    for seed, scheme in JOBS:
        name = f"seed{seed}_{scheme}"
        directory = OUTPUT / name
        log_path = BASE / f"artifacts/logs/{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(HERE / "scripts/run_cifar100_blockwise_apbr_training.py"),
                   "--dataset-root", str(DATA), "--vpr-src", str(VPR_SRC),
                   "--config", str(CONFIG), "--output-dir", str(directory),
                   "--scheme", scheme, "--seed", str(seed)]
        row = {"seed": seed, "scheme": scheme, "name": name, "status": "running",
               "command": command, "started_at_unix": time.time()}
        state["jobs"].append(row); state["current"] = name; atomic_status(state)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        row["returncode"] = result.returncode; row["finished_at_unix"] = time.time()
        if result.returncode != 0:
            row["status"] = "failed"; state["status"] = "failed"; state["current"] = None
            state["failure_job"] = name; atomic_status(state)
            return result.returncode
        row["status"] = "completed"; atomic_status(state)
    state["status"] = "training_complete_evaluation_pending"; state["current"] = None
    state["finished_at_unix"] = time.time(); atomic_status(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
