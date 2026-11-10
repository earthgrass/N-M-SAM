"""Read-only task metric on the frozen 1,024 training calibration examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_geo_obs_seed0 as common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    if (out / "calibration_task_metrics.json").exists():
        raise FileExistsError("Calibration task metrics already recorded")
    preflight = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    frozen = json.loads((out / "frozen_pretest.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETE" or frozen["status"] != "ALL_CHECKPOINTS_FROZEN":
        raise ValueError("Methods not frozen and complete")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if common.digest(args.config) != preflight["source_sha256"]["config"]:
        raise ValueError("Config hash mismatch")
    train_path = args.dataset_root / "extracted/cifar-100-python/train"
    data, labels = common.read_cifar(train_path, 50000)
    if common.digest(train_path) != preflight["train_pickle_sha256"]:
        raise ValueError("Train pickle hash mismatch")
    ids = np.random.default_rng(common.CALIBRATION_SEED).permutation(50000)[:1024].tolist()
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    if ids_sha != preflight["calibration_ids_sha256"]:
        raise ValueError("Calibration IDs mismatch")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    loader = DataLoader(common.CifarSubset(data, labels, ids, transform), batch_size=128,
                        shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda:0")
    results = []
    for method, record in frozen["methods"].items():
        path = out / "methods" / method / "checkpoint.pt"
        if common.digest(path) != record["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint hash mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        accuracy, ce, count = common.evaluate_once(checkpoint["model"], loader, device)
        results.append({"method": method, "train_calibration_accuracy_percent": accuracy,
                        "train_calibration_ce": ce, "samples": count,
                        "checkpoint_sha256": record["checkpoint_sha256"]})
    common.atomic_json(out / "calibration_task_metrics.json", {
        "status": "PASS", "source_sha256": common.digest(Path(__file__)),
        "test_accessed_by_this_script": False, "calibration_ids_sha256": ids_sha,
        "results": results,
    })
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
