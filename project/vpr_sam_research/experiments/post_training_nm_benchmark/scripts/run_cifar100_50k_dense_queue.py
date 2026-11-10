"""Serial fail-closed Dense-SAM seed 0/1/2 queue controller."""
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


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def gpu_pids() -> set[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    return {int(value) for value in output.splitlines() if value.strip().isdigit()}


def read_status(output: Path):
    path = output / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def verify_complete(output: Path, config_sha: str, training_sha: str):
    status = read_status(output)
    final = output / "final_epoch_200.pt"
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    if status is None or status.get("status") != "completed":
        raise RuntimeError(f"Run is not complete: {output}")
    if status.get("protocol_id") != "cifar100_50k10k_final_v1" or status.get("completed_epochs") != 200 or status.get("steps") != 78_200:
        raise RuntimeError(f"Final budget/protocol mismatch: {output}")
    if status.get("primary_checkpoint") != "final_epoch_200.pt" or status.get("test_accessed") is not False:
        raise RuntimeError(f"Final selection/Test policy mismatch: {output}")
    if not final.is_file() or sha_file(final) != status.get("final_checkpoint_sha256"):
        raise RuntimeError(f"Final checkpoint identity mismatch: {output}")
    if identity.get("config_sha256") != config_sha or identity.get("test_accessed") is not False:
        raise RuntimeError(f"Run identity mismatch: {output}")
    if training_sha not in set(identity.get("source_sha256", {}).values()):
        raise RuntimeError(f"Training source hash mismatch: {output}")
    return {"status": "PASS", "final_checkpoint": str(final),
            "final_checkpoint_sha256": sha_file(final), "completed_epochs": 200, "steps": 78_200}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vpr-src", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed0-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    controller_status = root / "dense_seed_queue_status.json"
    config_sha, training_sha = sha_file(args.config), sha_file(args.training_script)
    records = []
    atomic_json(controller_status, {"status": "RUNNING", "active_seed": 0,
                "seeds": [0, 1, 2], "config_sha256": config_sha,
                "training_script_sha256": training_sha, "started_at_unix": time.time()})
    for seed in (0, 1, 2):
        output = root / f"seed{seed}" / "dense_sam"
        if seed == 0:
            pid = args.seed0_pid
            if not output.is_dir():
                raise FileNotFoundError("Seed 0 output is missing")
        else:
            if output.exists():
                raise FileExistsError(f"Refusing existing Seed {seed} output: {output}")
            foreign = gpu_pids()
            if foreign:
                raise RuntimeError(f"GPU occupied before Seed {seed}: {sorted(foreign)}")
            log = root / f"seed{seed}" / "dense_sam_console.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("ab", buffering=0)
            command = [sys.executable, str(args.training_script), "--dataset-root", str(args.dataset_root),
                       "--vpr-src", str(args.vpr_src), "--config", str(args.config),
                       "--output-dir", str(output), "--method", "sam", "--seed", str(seed)]
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            stream.close()
            pid = child.pid
            atomic_json(root / f"seed{seed}" / "launcher.json",
                        {"pid": pid, "command": command, "started_at_unix": time.time()})
        atomic_json(controller_status, {"status": "RUNNING", "active_seed": seed,
                    "active_pid": pid, "completed": records, "config_sha256": config_sha,
                    "training_script_sha256": training_sha})
        while True:
            status = read_status(output)
            if status and status.get("status") == "completed":
                records.append({"seed": seed, **verify_complete(output, config_sha, training_sha)})
                for _ in range(12):
                    if not process_alive(pid):
                        break
                    time.sleep(5)
                if process_alive(pid):
                    raise RuntimeError(f"Seed {seed} published completion but process {pid} did not exit")
                break
            if status and status.get("status") == "failed":
                raise RuntimeError(f"Seed {seed} failed: {status}")
            if not process_alive(pid):
                raise RuntimeError(f"Seed {seed} process {pid} exited without a complete status")
            time.sleep(60)
    atomic_json(controller_status, {"status": "COMPLETE", "active_seed": None,
                "completed": records, "config_sha256": config_sha,
                "training_script_sha256": training_sha, "finished_at_unix": time.time()})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1])
            atomic_json(root / "dense_seed_queue_failure.json",
                        {"status": "FAILED", "type": type(error).__name__, "message": str(error),
                         "traceback": traceback.format_exc(), "automatic_retry": False,
                         "failed_at_unix": time.time()})
        raise
