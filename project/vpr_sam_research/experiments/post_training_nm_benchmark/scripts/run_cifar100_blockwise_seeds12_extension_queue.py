"""Wait for Seed 0, then run the frozen Seeds 1/2 extension serially."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[2]
ARTIFACTS = BASE / "artifacts"
SEED0_STATUS = ARTIFACTS / "seed0_v2_continuation_status.json"
RUN_ROOT = ARTIFACTS / "formal_v2_seeds12_resume1"
LOGS = ARTIFACTS / "logs_v2_seeds12_resume1"
STATUS = ARTIFACTS / "seeds12_v2_resume1_status.json"
EXPERIMENT = "apbr_blockwise_perturbation_cifar100_20261028_v2"
SCHEMES = ("global_block_feedback", "equal_block_energy")
SEEDS = (1, 2)


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
        raise FileExistsError("Refusing to overwrite Seeds-1/2 extension artifacts")
    RUN_ROOT.mkdir(parents=True); LOGS.mkdir(parents=True)
    state = {"status": "waiting_seed0", "experiment_id": EXPERIMENT,
             "scope": "seeds1_2_extension", "automatic_retry": False,
             "test_accessed": False, "completed": [], "started_at_unix": time.time()}
    atomic(state)
    while True:
        if not SEED0_STATUS.exists():
            time.sleep(60); continue
        seed0 = json.loads(SEED0_STATUS.read_text(encoding="utf-8"))
        if seed0.get("status") == "failed":
            state.update(status="failed", failure="seed0_failed_no_extension_started", finished_at_unix=time.time())
            atomic(state); return 1
        if seed0.get("status") == "completed":
            break
        time.sleep(60)
    config = ROOT / "configs/apbr_blockwise_perturbation_cifar100_20261028_v2_seeds12.yaml"
    eval_config = ROOT / "configs/cifar100_50k10k_final_v1.yaml"
    data = BASE / "data"
    try:
        for seed in SEEDS:
            for scheme in SCHEMES:
                dense = RUN_ROOT / f"seed{seed}" / scheme / "dense"
                state.update(status="running", current_seed=seed, current_scheme=scheme,
                             current_stage="dense_training")
                atomic(state)
                run([sys.executable, str(ROOT / "scripts/run_cifar100_blockwise_apbr_training.py"),
                     "--dataset-root", str(data), "--vpr-src", str(ROOT.parents[1] / "src"),
                     "--config", str(config), "--output-dir", str(dense),
                     "--scheme", scheme, "--seed", str(seed)],
                    LOGS / f"seed{seed}_{scheme}_dense.log")
                dense_status = json.loads((dense / "status.json").read_text(encoding="utf-8"))
                if dense_status.get("status") != "completed" or dense_status.get("steps") != 78_200:
                    raise RuntimeError(f"Dense completion contract failed: seed{seed} {scheme}")
                state["completed"].append({"seed": seed, "scheme": scheme, "stage": "dense_training"}); atomic(state)

                raw = RUN_ROOT / f"seed{seed}" / scheme / "raw"
                state.update(current_stage="raw_hard"); atomic(state)
                run([sys.executable, str(ROOT / "scripts/run_cifar100_blockwise_apbr_raw_seeds12.py"),
                     "--config", str(config), "--parent", str(dense / "final_epoch_200.pt"),
                     "--output-dir", str(raw), "--scheme", scheme, "--seed", str(seed)],
                    LOGS / f"seed{seed}_{scheme}_raw.log")
                raw_status = json.loads((raw / "status.json").read_text(encoding="utf-8"))
                if raw_status.get("status") != "completed" or raw_status.get("recovery_epochs") != 0:
                    raise RuntimeError(f"Raw completion contract failed: seed{seed} {scheme}")
                state["completed"].append({"seed": seed, "scheme": scheme, "stage": "raw_hard"}); atomic(state)

        state.update(current_seed=None, current_scheme=None, current_stage="fixed_final_test", test_accessed=True)
        atomic(state)
        for seed in SEEDS:
            for scheme in SCHEMES:
                root = RUN_ROOT / f"seed{seed}" / scheme
                evaluations = root / "test_final"; evaluations.mkdir(parents=True, exist_ok=True)
                for role, checkpoint, name in (
                    ("dense_final", root / "dense/final_epoch_200.pt", "dense_final.json"),
                    ("raw_hard", root / "raw/raw_hard.pt", "raw_hard.json"),
                ):
                    run([sys.executable, str(ROOT / "scripts/evaluate_cifar100_50k_final.py"),
                         "--dataset-root", str(data), "--config", str(eval_config),
                         "--checkpoint", str(checkpoint), "--checkpoint-role", role,
                         "--output", str(evaluations / name)],
                        LOGS / f"seed{seed}_{scheme}_{role}_test.log")
                state["completed"].append({"seed": seed, "scheme": scheme,
                                           "stage": "fixed_final_test", "targets": 2}); atomic(state)
        state.update(status="completed", current_stage=None, finished_at_unix=time.time()); atomic(state)
        return 0
    except Exception as error:
        state.update(status="failed", failure_type=type(error).__name__, failure=str(error),
                     finished_at_unix=time.time()); atomic(state); raise


if __name__ == "__main__":
    raise SystemExit(main())
