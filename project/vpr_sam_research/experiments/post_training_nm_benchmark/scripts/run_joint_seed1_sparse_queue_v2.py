"""Serial RTX 5090 queue: Seed 1 full Joint, then Seed 0/1 sparse APBR Joint."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Stage failed ({result.returncode}); log={log}; command={command}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--initial-seed0", type=Path, required=True)
    parser.add_argument("--initial-seed1", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Queue output is not empty")
    root.mkdir(parents=True)
    scripts = project / "scripts"
    py = sys.executable
    completed: list[dict] = []
    asset0 = args.initial_seed0.resolve()
    asset1 = args.initial_seed1.resolve()
    if read(asset1.parent / "asset_manifest.json")["checkpoint_sha256"] != sha(asset1):
        raise RuntimeError("Seed 1 initial asset hash mismatch")
    frozen_inputs = {str(path.resolve()): sha(path) for path in
        (args.protocol_config, args.joint_config, asset0, asset1,
         args.calibration_dir / "calibration_identity.json",
         args.calibration_dir / "lambda_calibration.json",
         args.calibration_dir / "eta_s_calibration.json")}
    sources = {str(path.resolve()): sha(path) for path in
        (scripts / name for name in (
            "run_joint_seed1_sparse_queue_v2.py",
            "create_apbr_gpt_joint_seed1_asset_v2.py",
            "run_apbr_gpt_joint_training_seed01_v2.py",
            "run_sam_gpt_joint_training_seed01_v2.py",
            "run_apbr_gpt_joint_matrix_prune_seed01_v2.py",
            "evaluate_cifar100_50k_final.py"))}
    atomic(root / "frozen_plan.json", {
        "protocol_id": "cifar100_50k10k_final_v1", "status": "FROZEN",
        "stages": ["sam_seed1_full", "apbr_seed1_full", "apbr_seed0_epoch10", "apbr_seed1_epoch10"],
        "sparse_guidance_epochs_one_based": list(range(10, 201, 10)),
        "full_guidance_epochs_one_based": list(range(6, 201)),
        "frozen_inputs_sha256": frozen_inputs, "source_sha256": sources,
        "seed1_lambda_eta_and_reference_source": "seed0_train_side_calibration_frozen",
        "test_policy": "once per frozen endpoint", "automatic_retry": False})
    atomic(root / "queue_status.json", {"status": "RUNNING", "phase": "asset_frozen",
        "completed": completed, "test_accessed": False})

    stages = (
        ("sam_seed1_full", "sam", 1, "full", asset1),
        ("apbr_seed1_full", "apbr", 1, "full", asset1),
        ("apbr_seed0_epoch10", "apbr", 0, "epoch10", asset0),
        ("apbr_seed1_epoch10", "apbr", 1, "epoch10", asset1),
    )
    for stage, method, seed, guidance, asset in stages:
        stage_root = root / stage
        train_dir = stage_root / "training"
        script = ("run_sam_gpt_joint_training_seed01_v2.py" if method == "sam"
                  else "run_apbr_gpt_joint_training_seed01_v2.py")
        atomic(root / "queue_status.json", {"status": "RUNNING", "phase": stage + ":training",
            "completed": completed, "test_accessed": False})
        command = [py, str(scripts / script), "--seed", str(seed),
            "--protocol-config", str(args.protocol_config), "--joint-config", str(args.joint_config),
            "--initial-asset", str(asset), "--calibration-dir", str(args.calibration_dir),
            "--data-root", str(args.data_root), "--output-dir", str(train_dir)]
        if method == "apbr":
            command += ["--guidance-mode", guidance]
        run(command, stage_root / "logs" / "training.log")
        status = read(train_dir / "status.json")
        parent = train_dir / "final_epoch_200.pt"
        if (status.get("status") != "completed" or status.get("completed_epochs") != 200
                or status.get("steps") != 78_200 or status.get("test_accessed") is not False
                or status.get("final_checkpoint_sha256") != sha(parent)):
            raise RuntimeError(f"Training contract failed: {stage}")
        completed.append({"stage": stage + ":dense", "sha256": sha(parent)})
        endpoints = (("original_sparsegpt", "original_gpt", None),
                     ("dynamic_groupwise_obs", "group_obs", None),
                     ("dynamic_true_probe", "dynamic_sam_gpt" if method == "sam" else "dynamic_apbr_gpt", None),
                     ("dynamic_shuffled_probe", "dynamic_sam_gpt" if method == "sam" else "dynamic_apbr_gpt", 20261031))
        for endpoint, selector, shuffle in endpoints:
            target = stage_root / "pruning" / endpoint
            atomic(root / "queue_status.json", {"status": "RUNNING", "phase": stage + ":pruning:" + endpoint,
                "completed": completed, "test_accessed": False})
            command = [py, str(scripts / "run_apbr_gpt_joint_matrix_prune_seed01_v2.py"),
                "--dataset-root", str(args.data_root), "--config", str(args.protocol_config),
                "--parent", str(parent), "--initial-asset", str(asset), "--seed", str(seed),
                "--output-dir", str(target), "--endpoint", endpoint, "--selector", selector]
            if shuffle is not None:
                command += ["--shuffle-seed", str(shuffle)]
            run(command, stage_root / "logs" / (endpoint + "_prune.log"))
            pruned = read(target / "status.json")
            if (pruned.get("status") != "completed" or pruned.get("strict_2_4") is not True
                    or pruned.get("finite") is not True or pruned.get("test_accessed") is not False
                    or pruned.get("raw_hard_sha256") != sha(target / "raw_hard.pt")):
                raise RuntimeError(f"Pruning contract failed: {stage}/{endpoint}")
            completed.append({"stage": stage + ":" + endpoint, "sha256": pruned["raw_hard_sha256"],
                              "mask_sha256": pruned["mask_sha256"]})
        targets = [("dense", parent, "dense_final")]
        targets += [(name, stage_root / "pruning" / name / "raw_hard.pt", "raw_hard")
                    for name, _selector, _shuffle in endpoints]
        atomic(stage_root / "checkpoint_freeze_before_test.json", {"status": "FROZEN",
            "targets": {name: {"sha256": sha(path), "role": role} for name, path, role in targets},
            "test_accessed": False})
        evaluations = stage_root / "final_test"
        evaluations.mkdir()
        for name, checkpoint, role in targets:
            atomic(root / "queue_status.json", {"status": "RUNNING", "phase": stage + ":test:" + name,
                "completed": completed, "test_accessed": True})
            run([py, str(scripts / "evaluate_cifar100_50k_final.py"),
                 "--dataset-root", str(args.data_root), "--config", str(args.protocol_config),
                 "--checkpoint", str(checkpoint), "--checkpoint-role", role,
                 "--output", str(evaluations / (name + ".json"))],
                stage_root / "logs" / (name + "_test.log"))
        results = {name: read(evaluations / (name + ".json")) for name, _path, _role in targets}
        if not all(row.get("status") == "PASS" and row.get("test_count") == 10_000 for row in results.values()):
            raise RuntimeError(f"Final Test contract failed: {stage}")
        atomic(evaluations / "FINAL_TEST_RESULTS.json", {"status": "FINAL_TEST_COMPLETE",
            "stage": stage, "evaluations": results, "evaluation_count": 5,
            "each_checkpoint_evaluated_once": True, "selection_performed": False})
        completed.append({"stage": stage + ":complete", "evaluations": 5})
    atomic(root / "queue_status.json", {"status": "COMPLETE", "phase": "all_done",
        "completed": completed, "test_accessed": True, "finished_at_unix": time.time()})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1])
            atomic(root / "queue_failure.json", {"status": "FAILED", "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(), "automatic_retry": False})
            atomic(root / "queue_status.json", {"status": "FAILED", "message": str(error),
                "automatic_retry": False})
        raise
