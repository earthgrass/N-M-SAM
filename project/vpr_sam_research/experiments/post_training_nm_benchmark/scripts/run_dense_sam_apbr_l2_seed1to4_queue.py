"""Serial Dense-only APBR-L2 1-4, then missing SAM 2-4; no retries/Test."""
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

import torch

ROOT = Path(__file__).resolve().parents[1]
JOBS = [(seed, "apbr_fixed_l2") for seed in (1, 2, 3, 4)]
JOBS += [(seed, "dense_sam") for seed in (2, 3, 4)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed1-sam-identity", type=Path, required=True)
    args = p.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One CUDA GPU required")
    reused = json.loads(args.seed1_sam_identity.read_text(encoding="utf-8"))
    if reused["seed"] != 1 or reused["method"] != "dense_sam" or reused["epoch"] != 200:
        raise ValueError("Seed1 SAM reuse identity invalid")
    status_path = output / "queue_status.json"
    state = {"status": "running", "jobs": [{"seed": s, "method": m, "state": "pending"} for s, m in JOBS],
             "reused_seed1_sam": reused, "automatic_retry": False,
             "test_accessed": False, "started_at_unix": time.time()}
    atomic_json(status_path, state)
    init_by_seed = {1: reused["initial_model_sha256"]}
    for index, (seed, method) in enumerate(JOBS):
        item = state["jobs"][index]
        item["state"] = "running"
        item["started_at_unix"] = time.time()
        atomic_json(status_path, state)
        directory = output / f"seed{seed}" / method
        if directory.exists():
            raise FileExistsError(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        if method == "dense_sam":
            script = ROOT / "scripts/run_cifar100_50k_sam_seed245_training.py"
            config = ROOT / "configs/cifar100_50k10k_final_v1_sam_seed245_rtx5090_20261103.yaml"
            method_arg = "sam"
            protocol = "cifar100_50k10k_final_v1"
        else:
            script = ROOT / "scripts/run_cifar100_apbr_fixed_l2_seed1to4_training.py"
            config = ROOT / "configs/cifar100_50k10k_apbr_fixed_l2_seed1to4_rtx5090_20261103.yaml"
            method_arg = "fixed_l2"
            protocol = "cifar100_50k10k_apbr_fixed_l2_seed1to4_20261103"
        command = [sys.executable, str(script), "--dataset-root", str(args.dataset_root),
                   "--vpr-src", str(ROOT.parents[1] / "src"), "--config", str(config),
                   "--output-dir", str(directory), "--method", method_arg, "--seed", str(seed)]
        item["script_sha256"] = sha(script)
        item["config_sha256"] = sha(config)
        log_path = output / f"seed{seed}_{method}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        item["exit_code"] = result.returncode
        item["log_sha256_at_exit"] = sha(log_path)
        if result.returncode:
            item["state"] = "failed"
            state["status"] = "failed"
            atomic_json(status_path, state)
            raise RuntimeError(f"{seed}/{method} exited {result.returncode}; see {log_path}")
        final_path = directory / "final_epoch_200.pt"
        checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        if (checkpoint["epoch"] != 200 or checkpoint["steps"] != 78200
                or checkpoint["protocol_id"] != protocol or checkpoint["test_accessed"] is not False
                or checkpoint["identity"]["seed"] != seed
                or checkpoint["identity"]["method"] != method
                or checkpoint["status"] != "healthy"):
            raise ValueError(f"Final checkpoint contract failed: {seed}/{method}")
        if checkpoint["identity"]["config_sha256"] != sha(config):
            raise ValueError(f"Config hash mismatch: {seed}/{method}")
        init_sha = checkpoint["identity"]["initial_model_sha256"]
        if method == "apbr_fixed_l2":
            if seed == 1 and init_by_seed[1] != init_sha:
                raise ValueError("Seed1 paired SAM/APBR-L2 initial model mismatch")
            init_by_seed[seed] = init_sha
        elif init_by_seed.get(seed) != init_sha:
            raise ValueError(f"Paired SAM/APBR-L2 initial model mismatch: seed {seed}")
        item["state"] = "completed"
        item["completed_at_unix"] = time.time()
        item["checkpoint_sha256"] = sha(final_path)
        item["initial_model_sha256"] = init_sha
        item["model_sha256"] = checkpoint["model_sha256"]
        item["test_accessed"] = False
        atomic_json(status_path, state)
    state["status"] = "completed"
    state["completed_at_unix"] = time.time()
    atomic_json(status_path, state)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if "--output-root" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output-root") + 1]) / "queue_status.json"
            if path.exists():
                state = json.loads(path.read_text(encoding="utf-8"))
                state["status"] = "failed"
                state["traceback"] = traceback.format_exc()
                atomic_json(path, state)
        traceback.print_exc()
        raise
