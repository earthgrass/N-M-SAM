"""Fail-closed preflight for the frozen CIFAR-100 50k/10k campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.cifar100_model import build_cifar100_model


def sha_file(path: Path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite preflight: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Ids(Dataset):
    def __len__(self):
        return 50_000

    def __getitem__(self, index):
        return index


def import_training(path: Path):
    spec = importlib.util.spec_from_file_location("cifar100_50k_preflight_training", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checks = {}
    checks["protocol_id"] = config["protocol_id"] == "cifar100_50k10k_final_v1"
    checks["sizes"] = (config["dataset"]["train_size"], config["dataset"]["test_size"],
                       config["dataset"]["validation_size"]) == (50_000, 10_000, 0)
    checks["fixed_final"] = (config["training"]["selection_mode"] == "fixed_final_epoch"
                             and config["training"]["primary_checkpoint"] == "final_epoch_200.pt")
    checks["drop_last_false"] = config["dataset"]["drop_last"] is False
    checks["test_policy"] = not any(config["evaluation"][key] for key in (
        "test_used_for_training", "test_used_for_checkpoint_selection", "test_used_for_hyperparameter_tuning"))
    training_source = args.training_script.read_text(encoding="utf-8")
    checks["training_has_no_test_path"] = '/ "test"' not in training_source
    checks["training_has_no_best_selection"] = not any(token in training_source for token in (
        "best_test", "best_val", "validation_accuracy", "early_stop"))

    archive = args.dataset_root / "cifar-100-python.tar.gz"
    checks["archive_size"] = archive.stat().st_size == 169_001_437
    checks["archive_md5"] = sha_file(archive, "md5") == config["dataset"]["archive_md5"]
    checks["archive_sha256"] = sha_file(archive) == config["dataset"]["archive_sha256"]
    train_path = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_path.open("rb") as stream:
        value = pickle.load(stream, encoding="bytes")
    data = np.asarray(value[b"data"], dtype=np.uint8)
    labels = np.asarray(value[b"fine_labels"], dtype=np.int64)
    checks["train_shape_dtype"] = data.shape == (50_000, 3072) and data.dtype == np.uint8
    checks["train_labels"] = labels.shape == (50_000,) and bool(np.all(np.bincount(labels, minlength=100) == 500))

    training = import_training(args.training_script)
    seen, batches = [], 0
    for ids in training.training_loader(Ids(), 128, seed=0, epoch=0, workers=0):
        seen.extend(ids.tolist())
        batches += 1
    checks["epoch_coverage"] = batches == 391 and len(seen) == 50_000 and len(set(seen)) == 50_000 and set(seen) == set(range(50_000))
    old = {"schema": 1, "status": "healthy", "protocol_id": "cifar100_45k5k", "epoch": 14,
           "model": {"x": torch.tensor([1.0])}, "stepper": {}, "scheduler": {}, "history": [],
           "identity": {}, "rng": {}}
    old["model_sha256"] = training.tensor_sha(old["model"])
    try:
        training.validate_checkpoint(old)
    except ValueError:
        checks["old_protocol_resume_rejected"] = True
    else:
        checks["old_protocol_resume_rejected"] = False

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires one real CUDA device")
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    checks["gpu_uuid"] = uuid == config["gpu_uuid"]
    model = build_cifar100_model().cuda().eval()
    sample = torch.from_numpy(data[:32].copy().reshape(-1, 3, 32, 32)).float().div_(255)
    mean = torch.tensor(config["dataset"]["normalize_mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(config["dataset"]["normalize_std"]).reshape(1, 3, 1, 1)
    with torch.inference_mode():
        output = model(((sample - mean) / std).cuda())
    checks["cuda_model_forward"] = output.shape == (32, 100) and bool(torch.isfinite(output).all())
    del model, output
    torch.cuda.empty_cache()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    pids = {int(value) for value in processes.splitlines() if value.strip().isdigit()}
    checks["no_foreign_gpu_process"] = not (pids - {os.getpid()})
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {"status": status, "checks": checks, "config_sha256": sha_file(args.config),
              "training_script_sha256": sha_file(args.training_script),
              "evaluation_script_sha256": sha_file(args.evaluation_script),
              "archive_sha256": sha_file(archive), "train_pickle_sha256": sha_file(train_path),
              "gpu": {"name": properties.name, "uuid": uuid,
                      "total_memory_bytes": properties.total_memory,
                      "compute_capability": [properties.major, properties.minor]},
              "epoch0_order_sha256": hashlib.sha256(np.asarray(seen, dtype="<i8").tobytes()).hexdigest(),
              "formal_training_started": False}
    atomic_json(args.output, result)
    print(json.dumps({"status": status, "checks": checks}, allow_nan=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
