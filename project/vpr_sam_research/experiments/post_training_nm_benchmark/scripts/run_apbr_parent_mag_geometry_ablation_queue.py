"""Serial GPU queue for the three-seed APBR-parent Magnitude ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ABLATION_ID = "ablation_apbr_parent_mag_vs_geometry_20261027"
ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"Command failed with exit {process.returncode}: {' '.join(command)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    for seed in range(3):
        parser.add_argument(f"--seed{seed}-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_root.resolve()
    preflight_path = output / "preflight_evidence.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("Preflight evidence is required")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("ablation_id") != ABLATION_ID:
        raise ValueError("Preflight did not pass")

    runner = ROOT / "scripts/run_apbr_parent_magnitude_ablation.py"
    evaluator = ROOT / "scripts/evaluate_cifar100_50k_final.py"
    snapshot = {
        "ablation_id": ABLATION_ID,
        "dataset_root": str(args.dataset_root.resolve()),
        "config": {"path": str(args.config.resolve()), "sha256": sha(args.config)},
        "runner": {"path": str(runner), "sha256": sha(runner)},
        "evaluator": {"path": str(evaluator), "sha256": sha(evaluator)},
        "seeds": [
            {
                "seed": seed,
                "formal_root": str(getattr(args, f"seed{seed}_root").resolve()),
                "parent": preflight["seeds"][seed]["parent"],
                "geometry": preflight["seeds"][seed]["geometry"],
            }
            for seed in range(3)
        ],
        "created_at_unix": time.time(),
        "automatic_retry": False,
        "test_selection": False,
    }
    atomic_json(output / "CONFIG_SNAPSHOT.json", snapshot)
    status = {
        "status": "running",
        "ablation_id": ABLATION_ID,
        "completed_seeds": [],
        "current_seed": 0,
        "automatic_retry": False,
        "test_selection": False,
    }
    atomic_json(output / "queue_status.json", status)

    for seed in range(3):
        formal_root = getattr(args, f"seed{seed}_root").resolve()
        seed_output = output / f"seed{seed}"
        method_output = seed_output / "magnitude_apbr_parent"
        if method_output.exists() and any(method_output.iterdir()):
            raise FileExistsError(f"Refusing existing Seed {seed} output")
        status["current_seed"] = seed
        status["current_stage"] = "magnitude_raw_and_recovery20"
        atomic_json(output / "queue_status.json", status)
        run(
            [
                sys.executable,
                str(runner),
                "--dataset-root",
                str(args.dataset_root),
                "--config",
                str(args.config),
                "--parent",
                str(formal_root / "apbr_sam/final_epoch_200.pt"),
                "--geometry-reference",
                str(formal_root / "methods/apbr_sam/raw_hard.pt"),
                "--output-dir",
                str(method_output),
                "--seed",
                str(seed),
            ],
            seed_output / "magnitude_console.log",
        )
        for role, checkpoint, filename in [
            ("raw_hard", method_output / "raw_hard.pt", "magnitude_raw.json"),
            (
                "recovery_final",
                method_output / "recovery/final_epoch_020.pt",
                "magnitude_recovery_final.json",
            ),
        ]:
            status["current_stage"] = f"evaluate_{role}"
            atomic_json(output / "queue_status.json", status)
            run(
                [
                    sys.executable,
                    str(evaluator),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--config",
                    str(args.config),
                    "--checkpoint",
                    str(checkpoint),
                    "--checkpoint-role",
                    role,
                    "--output",
                    str(seed_output / "test_final" / filename),
                ],
                seed_output / "test_final" / filename.replace(".json", ".log"),
            )
        status["completed_seeds"].append(seed)
        status["current_stage"] = "seed_complete"
        atomic_json(output / "queue_status.json", status)

    status.update(
        status="completed",
        current_seed=None,
        current_stage="all_seeds_complete",
        completed_at_unix=time.time(),
    )
    atomic_json(output / "queue_status.json", status)
    print(json.dumps(status, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-root" in sys.argv:
            failed_output = Path(sys.argv[sys.argv.index("--output-root") + 1])
            atomic_json(
                failed_output / "queue_failure.json",
                {
                    "status": "failed",
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "automatic_retry": False,
                },
            )
        raise
