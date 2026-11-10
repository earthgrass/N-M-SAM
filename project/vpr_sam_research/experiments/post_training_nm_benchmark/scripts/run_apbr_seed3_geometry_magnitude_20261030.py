"""Fail-closed APBR Seed 3 Geometry-vs-Magnitude matrix; no Dense baselines."""
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


def source_snapshot(args) -> dict:
    paths = {
        "config": args.config,
        "training": args.training_script,
        "postprocess": args.postprocess_script,
        "evaluation": args.evaluation_script,
        "apbr_magnitude": args.apbr_magnitude_script,
        "controller": Path(__file__).resolve(),
    }
    return {
        name: {"path": str(path.resolve()), "sha256": sha(path.resolve())}
        for name, path in paths.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--postprocess-script", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--apbr-magnitude-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    apbr = root / "seed3/apbr_sam"
    methods = root / "seed3/methods"
    forbidden = [methods, root / "seed3/test_final", root / "queue_failure.json"]
    if not apbr.is_dir() or any(path.exists() for path in forbidden):
        raise RuntimeError("Output root is not the exact imported APBR-SAM boundary")

    config = json.loads(json.dumps(__import__("yaml").safe_load(
        args.config.read_text(encoding="utf-8")
    )))
    if config.get("protocol_id") != PROTOCOL or config.get("seeds") != [SEED]:
        raise RuntimeError("APBR Seed 3 config identity mismatch")

    sources = source_snapshot(args)
    apbr_record = base.verify_complete(
        apbr, APBR_CONFIG_SHA256, sources["training"]["sha256"]
    )
    if apbr_record["final_checkpoint_sha256"] != APBR_FINAL_SHA256:
        raise RuntimeError("Imported APBR-SAM Final SHA256 mismatch")

    write(root / "apbr_ablation_source_config_snapshot.json", {
        "protocol_id": PROTOCOL,
        "seed": SEED,
        "automatic_retry": False,
        "test_selection": False,
        "imported_apbr": apbr_record,
        "sources": sources,
        "config": config,
        "created_at_unix": time.time(),
    })
    status_path = root / "queue_status.json"
    completed = [{"seed": SEED, "stage": "apbr_sam", "imported": True,
                  **apbr_record}]
    write(status_path, {
        "status": "RUNNING", "phase": "preflight_complete",
        "completed": completed, "test_accessed": False,
        "automatic_retry": False,
    })

    geometry = methods / "apbr_geometry"
    shared.assert_sources(sources)
    geometry_command = [
        sys.executable, str(args.postprocess_script),
        "--dataset-root", str(args.dataset_root),
        "--config", str(args.config),
        "--parent", str(apbr / "final_epoch_200.pt"),
        "--output-dir", str(geometry),
        "--method", "apbr_sam", "--seed", str(SEED),
    ]
    pid = shared.launch(
        geometry_command, methods / "apbr_geometry_console.log",
        methods / "apbr_geometry_launcher.json",
    )
    write(status_path, {
        "status": "RUNNING", "phase": "seed3_apbr_geometry", "pid": pid,
        "completed": completed, "test_accessed": False,
        "automatic_retry": False,
    })
    geometry_status = shared.wait_stage(pid, geometry, "postprocess", SEED, "apbr_geometry")
    shared.wait_exit(pid)
    completed.append({"seed": SEED, "stage": "apbr_geometry", **geometry_status})

    magnitude = methods / "apbr_magnitude"
    shared.assert_sources(sources)
    magnitude_command = [
        sys.executable, str(args.apbr_magnitude_script),
        "--dataset-root", str(args.dataset_root),
        "--config", str(args.config),
        "--parent", str(apbr / "final_epoch_200.pt"),
        "--geometry-reference", str(geometry / "raw_hard.pt"),
        "--output-dir", str(magnitude), "--seed", str(SEED),
    ]
    pid = shared.launch(
        magnitude_command, methods / "apbr_magnitude_console.log",
        methods / "apbr_magnitude_launcher.json",
    )
    write(status_path, {
        "status": "RUNNING", "phase": "seed3_apbr_magnitude", "pid": pid,
        "completed": completed, "test_accessed": False,
        "automatic_retry": False,
    })
    magnitude_status = shared.wait_stage(pid, magnitude, "postprocess", SEED, "apbr_magnitude")
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
    evaluations = root / "seed3/test_final"
    evaluations.mkdir(parents=True, exist_ok=False)
    for name, checkpoint, role in targets:
        shared.assert_sources(sources)
        command = [
            sys.executable, str(args.evaluation_script),
            "--dataset-root", str(args.dataset_root),
            "--config", str(args.config),
            "--checkpoint", str(checkpoint),
            "--checkpoint-role", role,
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
            output = Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve()
            write(output / "queue_failure.json", {
                "status": "FAILED", "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(),
                "automatic_retry": False, "failed_at_unix": time.time(),
            })
        raise
