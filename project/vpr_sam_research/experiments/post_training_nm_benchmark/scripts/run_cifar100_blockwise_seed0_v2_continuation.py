"""Continue Seed 0 after the audited pre-projection Raw validation failure."""
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
STATUS = ARTIFACTS / "seed0_v2_continuation_status.json"
EXPERIMENT = "apbr_blockwise_perturbation_cifar100_20261028_v2"


def atomic(value: dict) -> None:
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def main() -> int:
    if STATUS.exists():
        raise FileExistsError("Refusing to overwrite Seed-0 continuation status")
    config = ROOT / "configs/apbr_blockwise_perturbation_cifar100_20261028_v2.yaml"
    eval_config = ROOT / "configs/cifar100_50k10k_final_v1.yaml"
    data = BASE / "data"
    scheme4 = RUN_ROOT / "global_block_feedback"
    scheme1 = RUN_ROOT / "equal_block_energy"
    dense4 = json.loads((scheme4 / "dense/status.json").read_text(encoding="utf-8"))
    raw4 = json.loads((scheme4 / "raw_fix1/status.json").read_text(encoding="utf-8"))
    if dense4.get("status") != "completed" or dense4.get("steps") != 78_200:
        raise ValueError("Scheme-4 Dense checkpoint is not complete")
    if raw4.get("status") != "completed" or raw4.get("recovery_epochs") != 0:
        raise ValueError("Scheme-4 corrected Raw-Hard is not complete")
    state = {"status": "running", "experiment_id": EXPERIMENT, "scope": "seed0_continuation_after_raw_validation_fix",
             "automatic_retry": False, "test_accessed": False,
             "completed": [{"scheme": "global_block_feedback", "stage": "dense_training"},
                           {"scheme": "global_block_feedback", "stage": "raw_hard", "artifact": "raw_fix1"}],
             "current_scheme": "equal_block_energy", "current_stage": "dense_training",
             "started_at_unix": time.time()}
    atomic(state)
    try:
        dense1 = scheme1 / "dense"
        run([sys.executable, str(ROOT / "scripts/run_cifar100_blockwise_apbr_training.py"),
             "--dataset-root", str(data), "--vpr-src", str(ROOT.parents[1] / "src"),
             "--config", str(config), "--output-dir", str(dense1),
             "--scheme", "equal_block_energy", "--seed", "0"],
            LOGS / "equal_block_energy_dense.log")
        dense1_status = json.loads((dense1 / "status.json").read_text(encoding="utf-8"))
        if dense1_status.get("status") != "completed" or dense1_status.get("steps") != 78_200:
            raise RuntimeError("Scheme-1 Dense completion contract failed")
        state["completed"].append({"scheme": "equal_block_energy", "stage": "dense_training"}); atomic(state)

        raw1 = scheme1 / "raw"
        state.update(current_stage="raw_hard"); atomic(state)
        run([sys.executable, str(ROOT / "scripts/run_cifar100_blockwise_apbr_raw.py"),
             "--config", str(config), "--parent", str(dense1 / "final_epoch_200.pt"),
             "--output-dir", str(raw1), "--scheme", "equal_block_energy", "--seed", "0"],
            LOGS / "equal_block_energy_raw.log")
        raw1_status = json.loads((raw1 / "status.json").read_text(encoding="utf-8"))
        if raw1_status.get("status") != "completed" or raw1_status.get("recovery_epochs") != 0:
            raise RuntimeError("Scheme-1 Raw completion contract failed")
        state["completed"].append({"scheme": "equal_block_energy", "stage": "raw_hard"}); atomic(state)

        state.update(current_scheme=None, current_stage="fixed_final_test", test_accessed=True); atomic(state)
        targets = (
            (scheme4, scheme4 / "dense/final_epoch_200.pt", "dense_final", "dense_final.json", "global_block_feedback"),
            (scheme4, scheme4 / "raw_fix1/raw_hard.pt", "raw_hard", "raw_hard.json", "global_block_feedback"),
            (scheme1, scheme1 / "dense/final_epoch_200.pt", "dense_final", "dense_final.json", "equal_block_energy"),
            (scheme1, scheme1 / "raw/raw_hard.pt", "raw_hard", "raw_hard.json", "equal_block_energy"),
        )
        for scheme_root, checkpoint, role, name, scheme in targets:
            evaluations = scheme_root / "test_final"; evaluations.mkdir(parents=True, exist_ok=True)
            run([sys.executable, str(ROOT / "scripts/evaluate_cifar100_50k_final.py"),
                 "--dataset-root", str(data), "--config", str(eval_config),
                 "--checkpoint", str(checkpoint), "--checkpoint-role", role,
                 "--output", str(evaluations / name)], LOGS / f"{scheme}_{role}_test.log")
        state["completed"].extend([
            {"scheme": "global_block_feedback", "stage": "fixed_final_test", "targets": 2},
            {"scheme": "equal_block_energy", "stage": "fixed_final_test", "targets": 2},
        ])
        state.update(status="completed", current_stage=None, finished_at_unix=time.time()); atomic(state)
        return 0
    except Exception as error:
        state.update(status="failed", failure_type=type(error).__name__, failure=str(error),
                     finished_at_unix=time.time()); atomic(state); raise


if __name__ == "__main__":
    raise SystemExit(main())
