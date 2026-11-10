"""Fail-closed serial Seed-0 queue: two dense runs, Raw-Hard, then fixed final Test."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[2]
ARTIFACTS = BASE / "artifacts"
RUN_ROOT = ARTIFACTS / "formal_v2_seed0"
LOGS = ARTIFACTS / "logs_v2_seed0"
STATUS = ARTIFACTS / "seed0_v2_queue_status.json"
EXPERIMENT = "apbr_blockwise_perturbation_cifar100_20261028_v2"
SCHEMES = ("global_block_feedback", "equal_block_energy")


def atomic(value: dict) -> None:
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def main() -> int:
    if STATUS.exists() or RUN_ROOT.exists():
        raise FileExistsError("Refusing to overwrite Seed-0 v2 queue artifacts")
    RUN_ROOT.mkdir(parents=True)
    LOGS.mkdir(parents=True)
    config = ROOT / "configs/apbr_blockwise_perturbation_cifar100_20261028_v2.yaml"
    eval_config = ROOT / "configs/cifar100_50k10k_final_v1.yaml"
    data = BASE / "data"
    state = {"status": "running", "experiment_id": EXPERIMENT, "scope": "seed0_only",
             "automatic_retry": False, "seeds_1_2_authorized": False, "test_accessed": False,
             "completed": [], "started_at_unix": time.time()}
    atomic(state)
    try:
        for scheme in SCHEMES:
            dense = RUN_ROOT / scheme / "dense"
            state.update(current_scheme=scheme, current_stage="dense_training")
            atomic(state)
            run([sys.executable, str(ROOT / "scripts/run_cifar100_blockwise_apbr_training.py"),
                 "--dataset-root", str(data), "--vpr-src", str(ROOT.parents[1] / "src"),
                 "--config", str(config), "--output-dir", str(dense),
                 "--scheme", scheme, "--seed", "0"], LOGS / f"{scheme}_dense.log")
            dense_status = json.loads((dense / "status.json").read_text(encoding="utf-8"))
            if dense_status.get("status") != "completed" or dense_status.get("steps") != 78_200:
                raise RuntimeError(f"Dense completion contract failed: {scheme}")
            state["completed"].append({"scheme": scheme, "stage": "dense_training"})
            atomic(state)

            raw = RUN_ROOT / scheme / "raw"
            state.update(current_stage="raw_hard")
            atomic(state)
            run([sys.executable, str(ROOT / "scripts/run_cifar100_blockwise_apbr_raw.py"),
                 "--config", str(config), "--parent", str(dense / "final_epoch_200.pt"),
                 "--output-dir", str(raw), "--scheme", scheme, "--seed", "0"],
                LOGS / f"{scheme}_raw.log")
            raw_status = json.loads((raw / "status.json").read_text(encoding="utf-8"))
            if raw_status.get("status") != "completed" or raw_status.get("recovery_epochs") != 0:
                raise RuntimeError(f"Raw-Hard completion contract failed: {scheme}")
            state["completed"].append({"scheme": scheme, "stage": "raw_hard"})
            atomic(state)

        state.update(current_scheme=None, current_stage="fixed_final_test", test_accessed=True)
        atomic(state)
        for scheme in SCHEMES:
            scheme_root = RUN_ROOT / scheme
            evaluations = scheme_root / "test_final"
            evaluations.mkdir(parents=True, exist_ok=True)
            for role, checkpoint, name in (
                ("dense_final", scheme_root / "dense/final_epoch_200.pt", "dense_final.json"),
                ("raw_hard", scheme_root / "raw/raw_hard.pt", "raw_hard.json"),
            ):
                run([sys.executable, str(ROOT / "scripts/evaluate_cifar100_50k_final.py"),
                     "--dataset-root", str(data), "--config", str(eval_config),
                     "--checkpoint", str(checkpoint), "--checkpoint-role", role,
                     "--output", str(evaluations / name)], LOGS / f"{scheme}_{role}_test.log")
            state["completed"].append({"scheme": scheme, "stage": "fixed_final_test", "targets": 2})
            atomic(state)
        state.update(status="completed", current_stage=None, finished_at_unix=time.time())
        atomic(state)
        return 0
    except Exception as error:
        state.update(status="failed", failure_type=type(error).__name__, failure=str(error),
                     finished_at_unix=time.time())
        atomic(state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
