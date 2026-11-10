"""Fail-closed Seed 1/2 queue after an explicitly authorized GPU handoff.

Seed 1 reuses only the completed Dense-SAM Final checkpoint whose SHA256 is
provided on the command line. Seed 1 APBR-SAM and every Seed 2 stage start from
scratch on the destination GPU. All pruning branches retain the frozen 20-epoch
fixed-mask recovery protocol used by the original formal queue.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cifar100_50k_dense_queue as base


PROTOCOL = "cifar100_50k10k_final_v1"
METHODS = ("magnitude", "wanda", "sparsegpt", "rose", "apbr_sam")


def write(path: Path, value) -> None:
    base.atomic_json(path, value)


def wait_run(pid: int, output: Path, kind: str, seed: int, method: str | None = None):
    while True:
        status = base.read_status(output)
        if status and status.get("status") == "failed":
            raise RuntimeError(f"{kind} seed{seed} {method or ''} failed: {status}")
        if status and status.get("status") == "completed":
            if kind == "training":
                return status
            if (status.get("protocol_id") != PROTOCOL
                    or status.get("recovery_epochs") != 20
                    or status.get("recovery_steps") != 7820
                    or status.get("test_accessed") is not False):
                raise RuntimeError(f"Postprocess final contract failed: seed{seed} {method}")
            return status
        if not base.process_alive(pid):
            raise RuntimeError(f"{kind} process {pid} exited without completion")
        time.sleep(60)


def wait_exit(pid: int) -> None:
    for _ in range(12):
        if not base.process_alive(pid):
            return
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return
        time.sleep(5)
    raise RuntimeError(f"Process {pid} did not exit after publishing completion")


def launch(command: list[str], log: Path, receipt: Path) -> int:
    foreign = base.gpu_pids()
    if foreign:
        raise RuntimeError(f"GPU occupied before launch: {sorted(foreign)}")
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("ab", buffering=0)
    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True)
    stream.close()
    write(receipt, {"pid": child.pid, "command": command,
                    "started_at_unix": time.time()})
    return child.pid


def verify_external_seed1_dense(output: Path, expected_sha256: str):
    final = output / "final_epoch_200.pt"
    status = base.read_status(output)
    if status is None or status.get("status") != "completed":
        raise RuntimeError("External Seed 1 Dense status is not completed")
    if (status.get("protocol_id") != PROTOCOL
            or status.get("completed_epochs") != 200
            or status.get("steps") != 78200
            or status.get("primary_checkpoint") != "final_epoch_200.pt"
            or status.get("test_accessed") is not False):
        raise RuntimeError("External Seed 1 Dense frozen contract mismatch")
    actual = base.sha_file(final)
    if actual != expected_sha256 or status.get("final_checkpoint_sha256") != actual:
        raise RuntimeError("External Seed 1 Dense SHA256 mismatch")
    payload = torch.load(final, map_location="cpu", weights_only=False)
    identity = payload.get("identity", {})
    if (payload.get("protocol_id") != PROTOCOL or payload.get("epoch") != 200
            or payload.get("steps") != 78200 or payload.get("test_accessed") is not False
            or identity.get("seed") != 1 or identity.get("method") != "dense_sam"):
        raise RuntimeError("External Seed 1 Dense checkpoint identity mismatch")
    return {"status": "PASS_EXTERNAL", "final_checkpoint": str(final),
            "final_checkpoint_sha256": actual, "completed_epochs": 200,
            "steps": 78200, "source_gpu": identity.get("gpu")}


def verify_training(output: Path, config_sha: str, training_sha: str):
    return base.verify_complete(output, config_sha, training_sha)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--vpr-src", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--training-script", type=Path, required=True)
    p.add_argument("--postprocess-script", type=Path, required=True)
    p.add_argument("--evaluation-script", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--external-seed1-dense-sha256", required=True)
    args = p.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "blackwell_handoff_queue_status.json"
    if status_path.exists():
        raise FileExistsError(f"Refusing existing queue status: {status_path}")
    config_sha = base.sha_file(args.config)
    training_sha = base.sha_file(args.training_script)
    completed = []
    seed1_dense = verify_external_seed1_dense(
        root / "seed1" / "dense_sam", args.external_seed1_dense_sha256)
    completed.append({"seed": 1, "stage": "dense", **seed1_dense})
    write(status_path, {"status": "RUNNING", "phase": "seed1_apbr",
                        "completed": completed, "started_at_unix": time.time(),
                        "config_sha256": config_sha,
                        "training_script_sha256": training_sha,
                        "hardware_handoff": True})

    for seed in (1, 2):
        seedroot = root / f"seed{seed}"
        dense = seedroot / "dense_sam"
        apbr = seedroot / "apbr_sam"
        if seed == 2:
            if dense.exists():
                raise FileExistsError(f"Refusing existing Seed 2 Dense output: {dense}")
            cmd = [sys.executable, str(args.training_script), "--dataset-root", str(args.dataset_root),
                   "--vpr-src", str(args.vpr_src), "--config", str(args.config),
                   "--output-dir", str(dense), "--method", "sam", "--seed", "2"]
            pid = launch(cmd, seedroot / "dense_sam_console.log",
                         seedroot / "dense_sam_launcher.json")
            write(status_path, {"status": "RUNNING", "phase": "seed2_dense",
                                "pid": pid, "completed": completed})
            wait_run(pid, dense, "training", seed)
            wait_exit(pid)
            completed.append({"seed": seed, "stage": "dense",
                              **verify_training(dense, config_sha, training_sha)})

        if apbr.exists():
            raise FileExistsError(f"Refusing existing APBR output: {apbr}")
        cmd = [sys.executable, str(args.training_script), "--dataset-root", str(args.dataset_root),
               "--vpr-src", str(args.vpr_src), "--config", str(args.config),
               "--output-dir", str(apbr), "--method", "vpr", "--seed", str(seed)]
        pid = launch(cmd, seedroot / "apbr_sam_console.log",
                     seedroot / "apbr_sam_launcher.json")
        write(status_path, {"status": "RUNNING", "phase": f"seed{seed}_apbr",
                            "pid": pid, "completed": completed})
        wait_run(pid, apbr, "training", seed)
        wait_exit(pid)
        completed.append({"seed": seed, "stage": "apbr",
                          **verify_training(apbr, config_sha, training_sha)})

        for method in METHODS:
            output = seedroot / "methods" / method
            if output.exists():
                raise FileExistsError(f"Existing method output: {output}")
            parent = apbr / "final_epoch_200.pt" if method == "apbr_sam" else dense / "final_epoch_200.pt"
            cmd = [sys.executable, str(args.postprocess_script), "--dataset-root", str(args.dataset_root),
                   "--config", str(args.config), "--parent", str(parent),
                   "--output-dir", str(output), "--method", method, "--seed", str(seed)]
            pid = launch(cmd, seedroot / "methods" / f"{method}_console.log",
                         seedroot / "methods" / f"{method}_launcher.json")
            write(status_path, {"status": "RUNNING", "phase": f"seed{seed}_{method}",
                                "pid": pid, "completed": completed})
            method_status = wait_run(pid, output, "postprocess", seed, method)
            wait_exit(pid)
            completed.append({"seed": seed, "stage": method, **method_status})

        evaluations = seedroot / "test_final"
        evaluations.mkdir(parents=True, exist_ok=False)
        targets = [("dense_final", dense / "final_epoch_200.pt", "dense_final"),
                   ("apbr_final", apbr / "final_epoch_200.pt", "dense_final")]
        for method in METHODS:
            method_root = seedroot / "methods" / method
            targets.extend([
                (f"{method}_raw", method_root / "raw_hard.pt", "raw_hard"),
                (f"{method}_recovery_final", method_root / "recovery" / "final_epoch_020.pt", "recovery_final")])
        for name, checkpoint, role in targets:
            cmd = [sys.executable, str(args.evaluation_script), "--dataset-root", str(args.dataset_root),
                   "--config", str(args.config), "--checkpoint", str(checkpoint),
                   "--checkpoint-role", role, "--output", str(evaluations / f"{name}.json")]
            log = evaluations / f"{name}.log"
            with log.open("wb") as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Final Test evaluation failed: seed{seed} {name}")
        completed.append({"seed": seed, "stage": "final_test", "targets": len(targets)})
        write(status_path, {"status": "RUNNING", "phase": f"seed{seed}_complete",
                            "completed": completed})

    write(status_path, {"status": "COMPLETE", "completed": completed,
                        "finished_at_unix": time.time(), "seeds": [1, 2],
                        "methods": list(METHODS), "hardware_handoff": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1])
            write(root / "blackwell_handoff_queue_failure.json",
                  {"status": "FAILED", "type": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(),
                   "automatic_retry": False, "failed_at_unix": time.time()})
        raise
