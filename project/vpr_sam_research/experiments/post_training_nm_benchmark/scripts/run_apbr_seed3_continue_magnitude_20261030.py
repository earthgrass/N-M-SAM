"""Continue Seed 3 at the verified Geometry-complete boundary."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cifar100_50k_dense_queue as base
import run_cifar100_seed345_full_matrix as shared


PROTOCOL = "cifar100_50k10k_final_v1"
SEED = 3
APBR_CONFIG_SHA256 = "fbee864a9e08d5190c605cd635cb11466bbab624f4a59c778d366e3bf115c88e"
APBR_FINAL_SHA256 = "b5289b0bb6be67cef28d675c4d1dc6a4703ab615c60d707296f12c6bbd00b5e3"


def write(path: Path, value) -> None:
    base.atomic_json(path, value)


def sha(path: Path) -> str:
    return base.sha_file(path)


def wait_stage(pid: int, output: Path) -> dict:
    while True:
        status = base.read_status(output)
        if status and status.get("status") == "failed":
            raise RuntimeError(f"Magnitude failed: {status}")
        if status and status.get("status") == "completed":
            if (status.get("protocol_id") != PROTOCOL
                    or status.get("recovery_epochs") != 20
                    or status.get("recovery_steps") != 7820
                    or status.get("test_accessed") is not False):
                raise RuntimeError("Magnitude completion contract failed")
            return status
        stat = Path(f"/proc/{pid}/stat")
        if (not base.process_alive(pid)
                or (stat.exists() and stat.read_text().split()[2] == "Z")):
            raise RuntimeError(f"Magnitude process {pid} exited without completion")
        time.sleep(30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--magnitude-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    apbr = root / "seed3/apbr_sam"
    geometry = root / "seed3/methods/apbr_geometry"
    magnitude = root / "seed3/methods/apbr_magnitude"
    evaluations = root / "seed3/test_final"
    if magnitude.exists() or evaluations.exists() or not geometry.is_dir():
        raise RuntimeError("Continuation boundary is not Geometry-complete/Magnitude-absent")

    config = json.loads(json.dumps(__import__("yaml").safe_load(
        args.config.read_text(encoding="utf-8")
    )))
    if config.get("protocol_id") != PROTOCOL or config.get("seeds") != [SEED]:
        raise RuntimeError("Config identity mismatch")

    paths = {
        "config": args.config,
        "training": args.training_script,
        "evaluation": args.evaluation_script,
        "magnitude": args.magnitude_script,
        "controller": Path(__file__).resolve(),
    }
    sources = {
        name: {"path": str(path.resolve()), "sha256": sha(path.resolve())}
        for name, path in paths.items()
    }
    apbr_record = base.verify_complete(
        apbr, APBR_CONFIG_SHA256, sources["training"]["sha256"]
    )
    if apbr_record["final_checkpoint_sha256"] != APBR_FINAL_SHA256:
        raise RuntimeError("APBR Final identity mismatch")

    geometry_status = base.read_status(geometry)
    if (not geometry_status or geometry_status.get("status") != "completed"
            or geometry_status.get("protocol_id") != PROTOCOL
            or geometry_status.get("method") != "apbr_sam"
            or geometry_status.get("seed") != SEED
            or geometry_status.get("recovery_epochs") != 20
            or geometry_status.get("recovery_steps") != 7820
            or geometry_status.get("test_accessed") is not False):
        raise RuntimeError("Geometry boundary validation failed")
    for filename, key in (("raw_hard.pt", "raw_hard_sha256"),
                          ("recovery/final_epoch_020.pt", "recovery_final_sha256")):
        if sha(geometry / filename) != geometry_status[key]:
            raise RuntimeError(f"Geometry file identity mismatch: {filename}")

    write(root / "magnitude_continuation_snapshot.json", {
        "protocol_id": PROTOCOL, "seed": SEED, "automatic_retry": False,
        "apbr": apbr_record, "geometry": geometry_status,
        "sources": sources, "config": config, "created_at_unix": time.time(),
    })
    completed = [
        {"seed": SEED, "stage": "apbr_sam", "imported": True, **apbr_record},
        {"seed": SEED, "stage": "apbr_geometry", **geometry_status},
    ]
    status_path = root / "queue_status.json"
    command = [
        sys.executable, str(args.magnitude_script),
        "--dataset-root", str(args.dataset_root),
        "--config", str(args.config),
        "--parent", str(apbr / "final_epoch_200.pt"),
        "--geometry-reference", str(geometry / "raw_hard.pt"),
        "--output-dir", str(magnitude), "--seed", str(SEED),
    ]
    pid = shared.launch(
        command, root / "seed3/methods/apbr_magnitude_retry1_console.log",
        root / "seed3/methods/apbr_magnitude_retry1_launcher.json",
    )
    write(status_path, {
        "status": "RUNNING", "phase": "seed3_apbr_magnitude_retry1", "pid": pid,
        "completed": completed, "test_accessed": False, "automatic_retry": False,
    })
    magnitude_status = wait_stage(pid, magnitude)
    shared.wait_exit(pid)
    completed.append({"seed": SEED, "stage": "apbr_magnitude", **magnitude_status})

    targets = [
        ("apbr_final", apbr / "final_epoch_200.pt", "dense_final"),
        ("apbr_geometry_raw", geometry / "raw_hard.pt", "raw_hard"),
        ("apbr_geometry_recovery_final", geometry / "recovery/final_epoch_020.pt", "recovery_final"),
        ("apbr_magnitude_raw", magnitude / "raw_hard.pt", "raw_hard"),
        ("apbr_magnitude_recovery_final", magnitude / "recovery/final_epoch_020.pt", "recovery_final"),
    ]
    frozen = [
        {"name": name, "role": role, "path": str(path), "sha256": sha(path)}
        for name, path, role in targets
    ]
    write(root / "seed3/pre_test_freeze.json", {
        "protocol_id": PROTOCOL, "seed": SEED, "targets": frozen,
        "source_snapshot": sources, "test_accessed": False,
        "frozen_at_unix": time.time(),
    })
    evaluations.mkdir(parents=True, exist_ok=False)
    for name, checkpoint, role in targets:
        command = [
            sys.executable, str(args.evaluation_script),
            "--dataset-root", str(args.dataset_root),
            "--config", str(args.config),
            "--checkpoint", str(checkpoint), "--checkpoint-role", role,
            "--output", str(evaluations / f"{name}.json"),
        ]
        shared.run_sync(command, evaluations / f"{name}.log")

    completed.append({"seed": SEED, "stage": "final_test", "targets": len(targets)})
    write(status_path, {
        "status": "COMPLETE", "phase": "seed3_apbr_ablation_complete",
        "completed": completed, "final_test_targets": len(targets),
        "test_accessed": True, "automatic_retry": False,
        "finished_at_unix": time.time(),
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve()
            write(root / "magnitude_continuation_failure.json", {
                "status": "FAILED", "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(),
                "automatic_retry": False, "failed_at_unix": time.time(),
            })
        raise
