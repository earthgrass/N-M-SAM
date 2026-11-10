"""Freeze the CIFAR-100 45k/5k split and formal calibration budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

SPLIT_SEED = 20_261_026
CALIBRATION_SEED = 20_261_027
BN_SEED = 20_261_028
ARCHIVE_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


def json_write(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_stats(data: np.ndarray, indices: np.ndarray) -> tuple[list[float], list[float]]:
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = len(indices) * 1024
    for start in range(0, len(indices), 1000):
        block = data[indices[start : start + 1000]].reshape(-1, 3, 1024).astype(np.float64) / 255.0
        sums += block.sum(axis=(0, 2))
        squares += np.square(block).sum(axis=(0, 2))
    mean = sums / count
    std = np.sqrt(squares / count - np.square(mean))
    return mean.tolist(), std.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite protocol directory: {args.output_dir}")
    audit_bytes = args.dataset_audit.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("status") != "DATA_IDENTITY_PASS" or audit["archive"]["sha256"] != ARCHIVE_SHA256:
        raise RuntimeError("Frozen official CIFAR-100 identity prerequisite failed")
    train_path = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_path.open("rb") as handle:
        train = pickle.load(handle, encoding="bytes")
    data = np.asarray(train[b"data"], dtype=np.uint8)
    labels = np.asarray(train[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or set(Counter(labels.tolist()).values()) != {500}:
        raise RuntimeError("Unexpected CIFAR-100 training content")
    split_rng = np.random.default_rng(SPLIT_SEED)
    train_ids: list[int] = []
    validation_ids: list[int] = []
    for label in range(100):
        class_ids = np.flatnonzero(labels == label)
        class_ids = class_ids[split_rng.permutation(len(class_ids))]
        validation_ids.extend(class_ids[:50].tolist())
        train_ids.extend(class_ids[50:].tolist())
    train_ids = np.asarray(train_ids, dtype=np.int64)
    validation_ids = np.asarray(validation_ids, dtype=np.int64)
    train_ids = train_ids[split_rng.permutation(len(train_ids))]
    validation_ids = validation_ids[split_rng.permutation(len(validation_ids))]
    if len(set(train_ids.tolist()) & set(validation_ids.tolist())) or len(set(train_ids.tolist()) | set(validation_ids.tolist())) != 50_000:
        raise RuntimeError("Split is not a disjoint partition")
    calib_rng = np.random.default_rng(CALIBRATION_SEED)
    bn_rng = np.random.default_rng(BN_SEED)
    calibration_ids = train_ids[calib_rng.choice(len(train_ids), 1024, replace=False)]
    bn_ids = train_ids[bn_rng.choice(len(train_ids), 4096, replace=False)]
    mean, std = train_stats(data, train_ids)
    args.output_dir.mkdir(parents=True)
    indices = args.output_dir / "indices"
    indices.mkdir()
    index_payloads = {
        "train_45000.json": train_ids.tolist(),
        "validation_5000.json": validation_ids.tolist(),
        "calibration_1024.json": calibration_ids.tolist(),
        "bn_only_4096.json": bn_ids.tolist(),
    }
    index_files: dict[str, dict[str, object]] = {}
    for name, values in index_payloads.items():
        path = indices / name
        json_write(path, values)
        raw = np.asarray(values, dtype="<i8")
        index_files[name] = {"count": len(values), "int64_le_sha256": array_sha(raw), "json_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    train_distribution = Counter(labels[train_ids].tolist())
    val_distribution = Counter(labels[validation_ids].tolist())
    manifest = {
        "schema_version": 1,
        "status": "CIFAR100_FORMAL_SPLIT_PASS",
        "dataset": "official CIFAR-100 Python archive",
        "archive_sha256": ARCHIVE_SHA256,
        "dataset_audit_path": str(args.dataset_audit.resolve()),
        "dataset_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "split": {
            "seed": SPLIT_SEED, "strategy": "per-class permutation; first 50 validation, remaining 450 training; then independently permute each split",
            "train_count": len(train_ids), "validation_count": len(validation_ids), "test_count": 10_000,
            "train_class_counts": {str(k): int(v) for k, v in sorted(train_distribution.items())},
            "validation_class_counts": {str(k): int(v) for k, v in sorted(val_distribution.items())},
            "disjoint": True, "covers_official_train": True,
        },
        "formal_calibration": {"seed": CALIBRATION_SEED, "samples": 1024, "batch_size": 128, "augmentation": "none", "passes": 1, "source": "train_45000 only"},
        "bn_only": {"seed": BN_SEED, "samples": 4096, "batch_size": 128, "augmentation": "none", "passes": 1, "source": "train_45000 only"},
        "normalization": {"source": "train_45000 only", "population_mean": mean, "population_std": std},
        "index_files": index_files,
        "test_policy": "official test 10000 is excluded from split, calibration, hyperparameters, pruning and checkpoint selection",
        "duplicates_policy": "preserve official samples and labels exactly",
        "shared_across_seeds": [0, 1, 2],
    }
    json_write(args.output_dir / "CIFAR100_DATA_SPLIT_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "normalization": manifest["normalization"], "index_files": index_files}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
