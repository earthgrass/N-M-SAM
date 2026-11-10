"""Fail-closed serial queue for CIFAR-100 method-specific endpoints."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cifar100_50k_dense_queue as base

BASE_PROTOCOL = "cifar100_50k10k_final_v1"
COMPARISON_PROTOCOL = "cifar100_method_specific_endpoint_v1"
METHODS = ("magnitude", "wanda", "sparsegpt", "rose", "apbr_sam")
BASELINES = METHODS[:-1]


def write(path: Path, value) -> None:
    base.atomic_json(path, value)


def process_state(pid: int):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return None
    return stat.read_text(encoding="utf-8").split()[2]


def process_running(pid: int) -> bool:
    return base.process_alive(pid) and process_state(pid) != "Z"


def wait_exit(pid: int) -> None:
    for _ in range(12):
        if not process_running(pid):
            return
        time.sleep(5)
    raise RuntimeError(f"Process {pid} did not exit after publishing completion")


def launch(command: list[str], log: Path, receipt: Path) -> int:
    foreign = base.gpu_pids()
    if foreign:
        raise RuntimeError(f"GPU occupied before launch: {sorted(foreign)}")
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("ab", buffering=0)
    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    stream.close()
    write(receipt, {"pid": child.pid, "command": command, "started_at_unix": time.time()})
    return child.pid


def wait_training(pid: int, output: Path) -> None:
    while True:
        status = base.read_status(output)
        if status and status.get("status") == "failed":
            raise RuntimeError(f"Training failed: {output}: {status}")
        if status and status.get("status") == "completed":
            return
        if not process_running(pid):
            raise RuntimeError(f"Training process {pid} exited without completion: {output}")
        time.sleep(60)


def verify_training(output: Path, config_sha: str, training_sha: str):
    return base.verify_complete(output, config_sha, training_sha)


def run_training(args, root: Path, seed: int, method: str, attach_pid: int | None,
                 config_sha: str, training_sha: str, status_path: Path, completed: list, identities: dict):
    name = "dense_sam" if method == "sam" else "apbr_sam"
    output = root / f"seed{seed}" / name
    status = base.read_status(output)
    if status and status.get("status") == "completed":
        return verify_training(output, config_sha, training_sha)
    if output.exists() and attach_pid is None:
        raise FileExistsError(f"Existing incomplete training output requires explicit attachment: {output}")
    if attach_pid is not None:
        if not process_running(attach_pid):
            raise RuntimeError(f"Attached training PID is not running: {attach_pid}")
        pid = attach_pid
    else:
        command = [sys.executable, str(args.training_script), "--dataset-root", str(args.dataset_root),
                   "--vpr-src", str(args.vpr_src), "--config", str(args.base_config),
                   "--output-dir", str(output), "--method", method, "--seed", str(seed)]
        pid = launch(command, root / f"seed{seed}" / f"{name}_method_endpoint_console.log",
                     root / f"seed{seed}" / f"{name}_method_endpoint_launcher.json")
    write(status_path, {"status": "RUNNING", "comparison_protocol_id": COMPARISON_PROTOCOL,
          "phase": f"seed{seed}_{name}", "pid": pid, "completed": completed, **identities})
    wait_training(pid, output); wait_exit(pid)
    return verify_training(output, config_sha, training_sha)


def verify_method(output: Path, method: str):
    status = base.read_status(output)
    if not status or status.get("status") != "completed":
        raise RuntimeError(f"Method did not complete: {output}")
    if status.get("protocol_id") != BASE_PROTOCOL or status.get("comparison_protocol_id") != COMPARISON_PROTOCOL:
        raise RuntimeError(f"Method protocol mismatch: {output}")
    if status.get("test_accessed") is not False or status.get("method") != method:
        raise RuntimeError(f"Method identity/Test policy mismatch: {output}")
    if method in BASELINES:
        if status.get("primary_endpoint") != "raw_hard" or status.get("recovery_started") is not False or status.get("recovery_epochs") != 0:
            raise RuntimeError(f"Baseline endpoint contract mismatch: {output}")
        if (output / "recovery").exists():
            raise RuntimeError(f"Baseline recovery unexpectedly exists: {output}")
    else:
        if (status.get("primary_endpoint") != "recovery_final_epoch_20" or status.get("recovery_started") is not True
                or status.get("recovery_epochs") != 20 or status.get("recovery_steps") != 7820):
            raise RuntimeError(f"APBR endpoint contract mismatch: {output}")
    raw = output / "raw_hard.pt"
    if not raw.is_file() or base.sha_file(raw) != status.get("raw_hard_sha256"):
        raise RuntimeError(f"Raw endpoint hash mismatch: {output}")
    return status


def wait_method(pid: int, output: Path, method: str) -> dict:
    while True:
        status = base.read_status(output)
        if status and status.get("status") == "failed":
            raise RuntimeError(f"Method failed: {output}: {status}")
        if status and status.get("status") == "completed":
            return verify_method(output, method)
        if not process_running(pid):
            raise RuntimeError(f"Method process {pid} exited without completion: {output}")
        time.sleep(60)


def evaluate(args, checkpoint: Path, role: str, output: Path) -> dict:
    command = [sys.executable, str(args.evaluation_script), "--dataset-root", str(args.dataset_root),
               "--config", str(args.base_config), "--checkpoint", str(checkpoint),
               "--checkpoint-role", role, "--output", str(output)]
    log = output.with_suffix(".log")
    with log.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Final Test evaluation failed: {output}")
    value = json.loads(output.read_text(encoding="utf-8"))
    if (value.get("status") != "PASS" or value.get("test_count") != 10_000
            or value.get("selection_performed") is not False
            or value.get("hyperparameter_change_performed") is not False):
        raise RuntimeError(f"Final Test contract failed: {output}")
    return value


def audit_seed0(root: Path) -> dict:
    seed = root / "seed0"
    tests = seed / "test_final"
    expected = {"dense_final", "apbr_final"}
    expected |= {f"{method}_raw" for method in METHODS}
    expected |= {f"{method}_recovery_final" for method in METHODS}
    actual = {path.stem for path in tests.glob("*.json")}
    if actual != expected:
        raise RuntimeError("Seed 0 evaluation inventory mismatch")
    for name in expected:
        value = json.loads((tests / f"{name}.json").read_text(encoding="utf-8"))
        if value.get("status") != "PASS" or value.get("test_count") != 10_000:
            raise RuntimeError(f"Seed 0 evaluation failed audit: {name}")
    for method in METHODS:
        status = base.read_status(seed / "methods" / method)
        if not status or status.get("status") != "completed" or status.get("test_accessed") is not False:
            raise RuntimeError(f"Seed 0 method failed audit: {method}")
    return {"seed": 0, "stage": "existing_seed0_audited", "main_targets": 8,
            "supplemental_recovery_targets": 5, "test_targets_verified": 12}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vpr-src", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--endpoint-config", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--postprocess-script", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attach-seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--attach-dense-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve(); status_path = root / "method_endpoint_queue_status.json"
    endpoint = yaml.safe_load(args.endpoint_config.read_text(encoding="utf-8"))
    if endpoint.get("comparison_protocol_id") != COMPARISON_PROTOCOL or endpoint.get("status") != "FROZEN":
        raise ValueError("Endpoint protocol is not frozen")
    if endpoint.get("execution", {}).get("mode") == "reporting_overlay_on_existing_full_recovery_queue":
        raise RuntimeError("Reporting-only protocol forbids launching the alternate endpoint queue")
    config_sha = base.sha_file(args.base_config); training_sha = base.sha_file(args.training_script)
    if config_sha != endpoint.get("base_training_config_sha256"):
        raise ValueError("Base training config hash mismatch")
    identities = {"base_config_sha256": config_sha,
                  "endpoint_config_sha256": base.sha_file(args.endpoint_config),
                  "training_script_sha256": training_sha,
                  "postprocess_script_sha256": base.sha_file(args.postprocess_script),
                  "evaluation_script_sha256": base.sha_file(args.evaluation_script),
                  "controller_script_sha256": base.sha_file(Path(__file__))}
    completed = [audit_seed0(root)]
    write(status_path, {"status": "RUNNING", "comparison_protocol_id": COMPARISON_PROTOCOL,
          "phase": "seed0_existing_audit_complete", "completed": completed, **identities,
          "started_at_unix": time.time(), "automatic_retry": False})
    for seed in (1, 2):
        attach = args.attach_dense_pid if seed == args.attach_seed else None
        dense_record = run_training(args, root, seed, "sam", attach, config_sha, training_sha, status_path, completed, identities)
        completed.append({"seed": seed, "stage": "dense", **dense_record})
        apbr_record = run_training(args, root, seed, "vpr", None, config_sha, training_sha, status_path, completed, identities)
        completed.append({"seed": seed, "stage": "apbr", **apbr_record})
        seed_root = root / f"seed{seed}"
        for method in METHODS:
            output = seed_root / "methods" / method
            if output.exists():
                method_status = verify_method(output, method)
            else:
                parent = (seed_root / "apbr_sam" if method == "apbr_sam" else seed_root / "dense_sam") / "final_epoch_200.pt"
                command = [sys.executable, str(args.postprocess_script), "--dataset-root", str(args.dataset_root),
                           "--base-config", str(args.base_config), "--endpoint-config", str(args.endpoint_config),
                           "--parent", str(parent), "--output-dir", str(output), "--method", method, "--seed", str(seed)]
                pid = launch(command, seed_root / "methods" / f"{method}_method_endpoint_console.log",
                             seed_root / "methods" / f"{method}_method_endpoint_launcher.json")
                write(status_path, {"status": "RUNNING", "comparison_protocol_id": COMPARISON_PROTOCOL,
                      "phase": f"seed{seed}_{method}", "pid": pid, "completed": completed, **identities})
                method_status = wait_method(pid, output, method); wait_exit(pid)
            completed.append({"seed": seed, "stage": method, **method_status})
        tests = seed_root / "test_method_specific"
        tests.mkdir(parents=True, exist_ok=False)
        targets = [("dense_final", seed_root / "dense_sam" / "final_epoch_200.pt", "dense_final"),
                   ("apbr_final", seed_root / "apbr_sam" / "final_epoch_200.pt", "dense_final")]
        targets += [(f"{method}_raw", seed_root / "methods" / method / "raw_hard.pt", "raw_hard") for method in METHODS]
        targets.append(("apbr_sam_recovery_final", seed_root / "methods" / "apbr_sam" / "recovery" / "final_epoch_020.pt", "recovery_final"))
        for name, checkpoint, role in targets:
            evaluate(args, checkpoint, role, tests / f"{name}.json")
        completed.append({"seed": seed, "stage": "method_specific_final_test", "targets": len(targets)})
        write(status_path, {"status": "RUNNING", "comparison_protocol_id": COMPARISON_PROTOCOL,
              "phase": f"seed{seed}_complete", "completed": completed, **identities})
    write(status_path, {"status": "COMPLETE", "comparison_protocol_id": COMPARISON_PROTOCOL,
          "completed": completed, "finished_at_unix": time.time(), "seeds": [0, 1, 2],
          "main_endpoint_methods": list(METHODS), **identities})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1])
            write(root / "method_endpoint_queue_failure.json", {"status": "FAILED",
                  "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                  "automatic_retry": False, "failed_at_unix": time.time()})
        raise
