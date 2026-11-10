"""Seed 3 full matrix with a verified imported APBR Final and new Dense-SAM."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cifar100_50k_dense_queue as base

PROTOCOL = "cifar100_50k10k_final_v1"
SEEDS = (3,)
SAM_METHODS = ("magnitude", "wanda", "sparsegpt", "rose")
APBR_CONFIG_SHA256 = "fbee864a9e08d5190c605cd635cb11466bbab624f4a59c778d366e3bf115c88e"
APBR_FINAL_SHA256 = "b5289b0bb6be67cef28d675c4d1dc6a4703ab615c60d707296f12c6bbd00b5e3"


def write(path: Path, value) -> None:
    base.atomic_json(path, value)


def sha(path: Path) -> str:
    return base.sha_file(path)


def wait_exit(pid: int) -> None:
    for _ in range(12):
        if not base.process_alive(pid):
            return
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return
        time.sleep(5)
    raise RuntimeError(f"Process {pid} did not exit after completion publication")


def wait_stage(pid: int, output: Path, kind: str, seed: int, method: str = "") -> dict:
    while True:
        status = base.read_status(output)
        if status and status.get("status") == "failed":
            raise RuntimeError(f"{kind} seed{seed} {method} failed: {status}")
        if status and status.get("status") == "completed":
            if kind == "postprocess":
                if (status.get("protocol_id") != PROTOCOL
                        or status.get("recovery_epochs") != 20
                        or status.get("recovery_steps") != 7820
                        or status.get("test_accessed") is not False):
                    raise RuntimeError(f"Postprocess contract failed: seed{seed} {method}")
            return status
        if not base.process_alive(pid):
            raise RuntimeError(f"{kind} process {pid} exited without completion: seed{seed} {method}")
        time.sleep(60)


def launch(command: list[str], log: Path, receipt: Path) -> int:
    active = base.gpu_pids()
    if active:
        raise RuntimeError(f"GPU occupied before launch: {sorted(active)}")
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("ab", buffering=0)
    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    stream.close()
    write(receipt, {"pid": child.pid, "command": command, "started_at_unix": time.time()})
    return child.pid


def run_sync(command: list[str], log: Path) -> None:
    if base.gpu_pids():
        raise RuntimeError(f"GPU occupied before synchronous stage: {sorted(base.gpu_pids())}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {' '.join(command)}")


def snapshot_sources(args) -> dict:
    paths = {
        "config": args.config,
        "training": args.training_script,
        "postprocess": args.postprocess_script,
        "evaluation": args.evaluation_script,
        "apbr_magnitude": args.apbr_magnitude_script,
        "controller": Path(__file__).resolve(),
    }
    return {name: {"path": str(path.resolve()), "sha256": sha(path.resolve())}
            for name, path in paths.items()}


def assert_sources(snapshot: dict) -> None:
    for record in snapshot.values():
        path = Path(record["path"])
        if sha(path) != record["sha256"]:
            raise RuntimeError(f"Frozen source changed during queue: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vpr-src", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--postprocess-script", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--apbr-magnitude-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    apbr3 = root / "seed3/apbr_sam"
    forbidden = [root / "seed3/dense_sam", root / "seed3/methods",
                 root / "seed3/test_final", root / "queue_failure.json"]
    if not apbr3.is_dir() or any(path.exists() for path in forbidden):
        raise RuntimeError("Output root is not the exact imported APBR-SAM boundary")
    status_path = root / "queue_status.json"
    sources = snapshot_sources(args)
    config = json.loads(json.dumps(__import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))))
    if config.get("protocol_id") != PROTOCOL or config.get("seeds") != list(SEEDS):
        raise ValueError("Seed-extension config identity mismatch")
    write(root / "matrix_source_config_snapshot.json", {
        "protocol_id": PROTOCOL,
        "seeds": list(SEEDS),
        "automatic_retry": False,
        "test_selection": False,
        "sources": sources,
        "config": config,
        "created_at_unix": time.time(),
    })
    training_sha = sources["training"]["sha256"]
    apbr_record = base.verify_complete(apbr3, APBR_CONFIG_SHA256, training_sha)
    if apbr_record["final_checkpoint_sha256"] != APBR_FINAL_SHA256:
        raise RuntimeError("Imported APBR-SAM checkpoint SHA256 mismatch")
    completed: list[dict] = [{"seed": 3, "stage": "apbr_sam", "imported": True,
                             **apbr_record}]
    write(status_path, {"status": "RUNNING", "phase": "preflight_complete", "completed": completed,
                        "seeds": list(SEEDS), "test_accessed": False, "automatic_retry": False})

    config_sha = sources["config"]["sha256"]
    for seed in SEEDS:
        seed_root = root / f"seed{seed}"
        dense = seed_root / "dense_sam"
        apbr = seed_root / "apbr_sam"
        for output, method in ((dense, "sam"), (apbr, "vpr")):
            assert_sources(sources)
            if method == "vpr":
                continue
            if output.exists():
                raise FileExistsError(f"Refusing existing training output: {output}")
            command = [sys.executable, str(args.training_script),
                       "--dataset-root", str(args.dataset_root), "--vpr-src", str(args.vpr_src),
                       "--config", str(args.config), "--output-dir", str(output),
                       "--method", method, "--seed", str(seed)]
            label = "dense_sam" if method == "sam" else "apbr_sam"
            pid = launch(command, seed_root / f"{label}_console.log", seed_root / f"{label}_launcher.json")
            write(status_path, {"status": "RUNNING", "phase": f"seed{seed}_{label}", "pid": pid,
                                "completed": completed, "test_accessed": False, "automatic_retry": False})
            wait_stage(pid, output, "training", seed, label)
            wait_exit(pid)
            record = base.verify_complete(output, config_sha, training_sha)
            completed.append({"seed": seed, "stage": label, **record})

        methods_root = seed_root / "methods"
        for method in SAM_METHODS:
            assert_sources(sources)
            output = methods_root / method
            if output.exists():
                raise FileExistsError(f"Refusing existing method output: {output}")
            command = [sys.executable, str(args.postprocess_script),
                       "--dataset-root", str(args.dataset_root), "--config", str(args.config),
                       "--parent", str(dense / "final_epoch_200.pt"),
                       "--output-dir", str(output), "--method", method, "--seed", str(seed)]
            pid = launch(command, methods_root / f"{method}_console.log", methods_root / f"{method}_launcher.json")
            write(status_path, {"status": "RUNNING", "phase": f"seed{seed}_{method}", "pid": pid,
                                "completed": completed, "test_accessed": False, "automatic_retry": False})
            stage = wait_stage(pid, output, "postprocess", seed, method)
            wait_exit(pid)
            completed.append({"seed": seed, "stage": method, "endpoint_semantics":
                              "native_compensated" if method in {"sparsegpt", "rose"} else "raw_hard",
                              **stage})

        geometry = methods_root / "apbr_geometry"
        assert_sources(sources)
        geometry_command = [sys.executable, str(args.postprocess_script),
                            "--dataset-root", str(args.dataset_root), "--config", str(args.config),
                            "--parent", str(apbr / "final_epoch_200.pt"),
                            "--output-dir", str(geometry), "--method", "apbr_sam", "--seed", str(seed)]
        pid = launch(geometry_command, methods_root / "apbr_geometry_console.log",
                     methods_root / "apbr_geometry_launcher.json")
        write(status_path, {"status": "RUNNING", "phase": f"seed{seed}_apbr_geometry", "pid": pid,
                            "completed": completed, "test_accessed": False, "automatic_retry": False})
        geometry_status = wait_stage(pid, geometry, "postprocess", seed, "apbr_geometry")
        wait_exit(pid)
        completed.append({"seed": seed, "stage": "apbr_geometry", "endpoint_semantics": "raw_hard",
                          **geometry_status})

        magnitude = methods_root / "apbr_magnitude"
        assert_sources(sources)
        magnitude_command = [sys.executable, str(args.apbr_magnitude_script),
                             "--dataset-root", str(args.dataset_root), "--config", str(args.config),
                             "--parent", str(apbr / "final_epoch_200.pt"),
                             "--geometry-reference", str(geometry / "raw_hard.pt"),
                             "--output-dir", str(magnitude), "--seed", str(seed)]
        pid = launch(magnitude_command, methods_root / "apbr_magnitude_console.log",
                     methods_root / "apbr_magnitude_launcher.json")
        write(status_path, {"status": "RUNNING", "phase": f"seed{seed}_apbr_magnitude", "pid": pid,
                            "completed": completed, "test_accessed": False, "automatic_retry": False})
        magnitude_status = wait_stage(pid, magnitude, "postprocess", seed, "apbr_magnitude")
        wait_exit(pid)
        completed.append({"seed": seed, "stage": "apbr_magnitude", "endpoint_semantics": "raw_hard",
                          **magnitude_status})

        targets = [
            ("dense_final", dense / "final_epoch_200.pt", "dense_final"),
            ("apbr_final", apbr / "final_epoch_200.pt", "dense_final"),
        ]
        for method in SAM_METHODS:
            targets.extend([
                (f"{method}_immediate", methods_root / method / "raw_hard.pt", "raw_hard"),
                (f"{method}_recovery_final", methods_root / method / "recovery/final_epoch_020.pt", "recovery_final"),
            ])
        for label in ("apbr_geometry", "apbr_magnitude"):
            targets.extend([
                (f"{label}_raw", methods_root / label / "raw_hard.pt", "raw_hard"),
                (f"{label}_recovery_final", methods_root / label / "recovery/final_epoch_020.pt", "recovery_final"),
            ])
        frozen = [{"name": name, "role": role, "path": str(path), "sha256": sha(path)}
                  for name, path, role in targets]
        write(seed_root / "pre_test_freeze.json", {
            "protocol_id": PROTOCOL, "seed": seed, "targets": frozen,
            "source_snapshot": sources, "test_accessed": False, "frozen_at_unix": time.time(),
        })
        evaluations = seed_root / "test_final"
        evaluations.mkdir(parents=True, exist_ok=False)
        for name, checkpoint, role in targets:
            assert_sources(sources)
            command = [sys.executable, str(args.evaluation_script),
                       "--dataset-root", str(args.dataset_root), "--config", str(args.config),
                       "--checkpoint", str(checkpoint), "--checkpoint-role", role,
                       "--output", str(evaluations / f"{name}.json")]
            run_sync(command, evaluations / f"{name}.log")
        completed.append({"seed": seed, "stage": "final_test", "targets": len(targets)})
        write(status_path, {"status": "RUNNING", "phase": f"seed{seed}_complete", "completed": completed,
                            "test_accessed": True, "automatic_retry": False})

    write(status_path, {"status": "COMPLETE", "phase": "all_seeds_complete", "completed": completed,
                        "seeds": list(SEEDS), "methods_per_seed": 6, "final_test_targets_per_seed": 14,
                        "test_accessed": True, "automatic_retry": False, "finished_at_unix": time.time()})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-root" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve()
            write(output / "queue_failure.json", {
                "status": "FAILED", "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False,
                "failed_at_unix": time.time(),
            })
        raise
