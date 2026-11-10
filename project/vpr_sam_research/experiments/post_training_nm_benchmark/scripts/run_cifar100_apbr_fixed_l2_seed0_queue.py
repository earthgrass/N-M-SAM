"""Serial, no-retry APBR fixed-L2 Seed 0 and native SparseGPT experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


PROTOCOL = "cifar100_50k10k_apbr_fixed_l2_seed0_20261102"
SCRIPTS = Path(__file__).resolve().parent
CONFIG = SCRIPTS.parent / "configs" / f"{PROTOCOL}.yaml"
EXPECTED_INITIAL_MODEL_SHA256 = "87aa9dd7492a89adf9e3ab78ca4a3227f30bc8de53b18a3bd73cd61ab0bd759d"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_dense(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (value.get("protocol_id") != PROTOCOL or value.get("epoch") != 200
            or value.get("steps") != 78200 or value.get("test_accessed") is not False
            or value.get("identity", {}).get("method") != "apbr_fixed_l2"
            or value["identity"].get("initial_model_sha256") != EXPECTED_INITIAL_MODEL_SHA256
            or value.get("stepper", {}).get("geometry", {}).get("method") != "fixed_l2"
            or value["stepper"]["geometry"].get("updates") != 78200):
        raise ValueError("APBR fixed-L2 dense final contract failed")
    return {"sha256": digest(path), "epoch": 200, "steps": 78200}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vpr-src", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "queue_status.json"
    status = {"protocol_id": PROTOCOL, "seed": 0, "state": "running", "completed": [],
              "started_at_unix": time.time(), "automatic_retries": 0}
    write_json(status_path, status)
    python = sys.executable
    common = ["--dataset-root", str(args.dataset_root.resolve()), "--config", str(CONFIG)]
    dense = root / "apbr_fixed_l2" / "final_epoch_200.pt"
    geometry = root / "geometry"
    magnitude = root / "magnitude"
    gpt = root / "original_sparsegpt"

    def run(name: str, command: list[str]) -> None:
        status["current"] = name
        status["current_command"] = command
        write_json(status_path, status)
        started = time.time()
        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  check=False).returncode
        if code:
            raise RuntimeError(f"{name} exited {code}; see {name}.log")
        status["completed"].append({"stage": name, "seconds": time.time() - started})
        status.pop("current", None)
        status.pop("current_command", None)
        write_json(status_path, status)

    try:
        run("dense_training", [python, "-u", str(SCRIPTS / "run_cifar100_apbr_fixed_l2_seed0_training.py"),
                               *common, "--vpr-src", str(args.vpr_src.resolve()),
                               "--output-dir", str(dense.parent), "--method", "fixed_l2", "--seed", "0"])
        status["dense_final"] = verify_dense(dense)
        write_json(status_path, status)
        for name, output in (("apbr_fixed_l2", geometry), ("magnitude", magnitude)):
            run(f"postprocess_{name}", [python, "-u", str(SCRIPTS / "run_cifar100_apbr_fixed_l2_seed0_postprocess.py"),
                                        *common, "--parent", str(dense), "--output-dir", str(output),
                                        "--method", name, "--seed", "0"])
        run("original_sparsegpt", [python, "-u", str(SCRIPTS / "run_cifar100_apbr_fixed_l2_seed0_sparsegpt.py"),
                                   *common, "--parent", str(dense), "--parent-method", "apbr_fixed_l2",
                                   "--output-dir", str(gpt), "--seed", "0"])
        frozen = {
            "dense": dense,
            "geometry_raw": geometry / "raw_hard.pt",
            "geometry_recovery20": geometry / "recovery" / "final_epoch_020.pt",
            "magnitude_raw": magnitude / "raw_hard.pt",
            "magnitude_recovery20": magnitude / "recovery" / "final_epoch_020.pt",
            "original_sparsegpt_native": gpt / "sparsegpt_final.pt",
        }
        status["frozen_checkpoint_sha256"] = {name: digest(path) for name, path in frozen.items()}
        write_json(status_path, status)
        for name, path in frozen.items():
            role = "dense_final" if name == "dense" else ("recovery_final" if "recovery" in name else "raw_hard")
            run(f"evaluate_{name}", [python, "-u", str(SCRIPTS / "evaluate_cifar100_apbr_fixed_l2_seed0.py"),
                                     *common, "--checkpoint", str(path), "--checkpoint-role", role,
                                     "--output", str(root / "evaluations" / f"{name}.json")])
        status["state"] = "complete"
        status["finished_at_unix"] = time.time()
    except BaseException as error:
        status["state"] = "failed"
        status["failure"] = repr(error)
        raise
    finally:
        write_json(status_path, status)


if __name__ == "__main__":
    main()
