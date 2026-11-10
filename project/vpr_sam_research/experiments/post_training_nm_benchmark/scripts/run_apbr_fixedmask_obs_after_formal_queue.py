"""Run one exploratory Seed-0 fixed-mask OBS job after the formal queue exits."""
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


EXPERIMENT_ID = "apbr_seed0_fixedmask_obs_20261027"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_pids() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "controller_status.json"
    started = time.time()
    write_json(status_path, {
        "status": "WAITING_FOR_FORMAL_QUEUE", "experiment_id": EXPERIMENT_ID,
        "started_at_unix": started, "automatic_retry": False,
    })
    formal_status = args.formal_root / "full_queue_status.json"
    formal_failure = args.formal_root / "full_queue_failure.json"
    while True:
        if formal_failure.exists():
            raise RuntimeError("Formal queue failed; exploratory job will not start")
        state = json.loads(formal_status.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE":
            occupied = gpu_pids()
            if not occupied:
                break
            write_json(status_path, {
                "status": "WAITING_FOR_GPU_IDLE", "experiment_id": EXPERIMENT_ID,
                "formal_queue_status": "COMPLETE", "gpu_pids": occupied,
                "started_at_unix": started, "automatic_retry": False,
            })
        else:
            write_json(status_path, {
                "status": "WAITING_FOR_FORMAL_QUEUE", "experiment_id": EXPERIMENT_ID,
                "formal_phase": state.get("phase"), "started_at_unix": started,
                "automatic_retry": False,
            })
        time.sleep(args.poll_seconds)

    run_dir = output / "run"
    checkpoint = run_dir / "apbr_seed0_fixedmask_obs.pt"
    parent = args.formal_root / "seed0" / "apbr_sam" / "final_epoch_200.pt"
    historical_raw = args.formal_root / "seed0" / "methods" / "apbr_sam" / "raw_hard.pt"
    write_json(status_path, {
        "status": "RUNNING_COMPENSATION", "experiment_id": EXPERIMENT_ID,
        "formal_queue_status": "COMPLETE", "started_at_unix": started,
        "compensation_started_at_unix": time.time(), "automatic_retry": False,
    })
    compensation_command = [
        sys.executable, str(args.runner), "--dataset-root", str(args.dataset_root),
        "--formal-config", str(args.formal_config), "--experiment-config", str(args.experiment_config),
        "--parent", str(parent), "--historical-raw", str(historical_raw),
        "--formal-queue-status", str(formal_status), "--output-dir", str(run_dir),
    ]
    run_logged(compensation_command, output / "compensation.log")

    write_json(status_path, {
        "status": "RUNNING_FINAL_TEST", "experiment_id": EXPERIMENT_ID,
        "formal_queue_status": "COMPLETE", "started_at_unix": started,
        "automatic_retry": False,
    })
    evaluation_path = output / "evaluation.json"
    evaluation_command = [
        sys.executable, str(args.evaluator), "--dataset-root", str(args.dataset_root),
        "--config", str(args.formal_config), "--checkpoint", str(checkpoint),
        "--checkpoint-role", "raw_hard", "--output", str(evaluation_path),
    ]
    run_logged(evaluation_command, output / "evaluation.log")
    compensation = json.loads((run_dir / "compensation_evidence.json").read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "PASS" or evaluation.get("checkpoint_sha256") != compensation["checkpoint_sha256"]:
        raise RuntimeError("Evaluation/checkpoint identity mismatch")
    evidence = {
        "schema": 1, "status": "PASS", "experiment_id": EXPERIMENT_ID,
        "formal_queue_was_complete_before_launch": True,
        "automatic_retry": False, "expanded_to_other_seeds": False,
        "compensation": compensation, "evaluation": evaluation,
        "controller_sha256": sha256_file(Path(__file__)),
        "runner_sha256": sha256_file(args.runner),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "formal_config_sha256": sha256_file(args.formal_config),
        "finished_at_unix": time.time(),
    }
    write_json(output / "experiment_evidence.json", evidence)
    write_json(status_path, {
        "status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
        "checkpoint_sha256": compensation["checkpoint_sha256"],
        "test_accuracy": evaluation["test_accuracy"], "test_loss": evaluation["test_loss"],
        "started_at_unix": started, "finished_at_unix": evidence["finished_at_unix"],
        "automatic_retry": False,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1])
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "controller_status.json", {
                "status": "FAILED", "experiment_id": EXPERIMENT_ID,
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False,
                "failed_at_unix": time.time(),
            })
        raise
