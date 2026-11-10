"""Narrow an already-started six-run controller to Seed 0 only, fail closed."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parents[2]
ARTIFACTS = BASE / "artifacts"
FORMAL = ARTIFACTS / "formal"
STATUS = ARTIFACTS / "seed0_only_queue_status.json"


def atomic(value: dict) -> None:
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def process_state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return None


def completed(path: Path, expected_scheme: str) -> bool:
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return (value.get("status") == "completed" and value.get("seed") == 0
            and value.get("scheme") == expected_scheme and value.get("completed_epochs") == 200
            and value.get("steps") == 78_200 and value.get("test_accessed") is False)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: continue_seed0_only.py CONTROLLER_PID CURRENT_CHILD_PID")
    controller_pid, child_pid = map(int, sys.argv[1:])
    if STATUS.exists():
        raise FileExistsError("Seed-0-only status already exists")
    state = {"status": "waiting_scheme4", "scope": "seed0_only", "controller_pid": controller_pid,
             "scheme4_pid": child_pid, "automatic_retry": False, "seeds_1_2_authorized": False,
             "test_accessed": False, "started_at_unix": time.time()}
    atomic(state)
    while process_state(child_pid) not in {None, "Z"}:
        time.sleep(30)
    scheme4_status = FORMAL / "seed0_global_block_feedback/status.json"
    if not completed(scheme4_status, "global_block_feedback"):
        state.update(status="failed", failure="scheme4_not_completed", finished_at_unix=time.time())
        atomic(state)
        return 1
    # The stopped original controller cannot reap its child or launch the next
    # job. Remove only that controller after its child has published Epoch 200.
    try:
        os.kill(controller_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    state.update(status="running_scheme1", scheme4_status="completed", scheme1_started_at_unix=time.time())
    atomic(state)
    output = FORMAL / "seed0_equal_block_energy"
    log_path = ARTIFACTS / "logs/seed0_equal_block_energy.log"
    command = [sys.executable, str(HERE / "scripts/run_cifar100_blockwise_apbr_training.py"),
               "--dataset-root", str(BASE / "data"), "--vpr-src", str(HERE.parents[1] / "src"),
               "--config", str(HERE / "configs/apbr_blockwise_perturbation_cifar100_20261028.yaml"),
               "--output-dir", str(output), "--scheme", "equal_block_energy", "--seed", "0"]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not completed(output / "status.json", "equal_block_energy"):
        state.update(status="failed", failure="scheme1_failed", returncode=result.returncode,
                     finished_at_unix=time.time())
        atomic(state)
        return result.returncode or 1
    state.update(status="seed0_training_complete_evaluation_pending", scheme1_status="completed",
                 finished_at_unix=time.time(), test_accessed=False)
    atomic(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
