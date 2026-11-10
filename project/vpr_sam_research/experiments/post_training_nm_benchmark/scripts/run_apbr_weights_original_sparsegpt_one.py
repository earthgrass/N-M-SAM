"""Run Original SparseGPT on one frozen CIFAR-100 dense parent.

This isolated entry point performs calibration and native dynamic 2:4 SparseGPT
pruning only.  It never consumes APBR geometry, an external mask, Test data,
BN calibration, Recovery, or gradient training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.sparsegpt import SparseGPTAdapter, sparsegpt_matrix
from benchmark.nm import eligible_layers, validate_masks

PROTOCOL = "cifar100_50k10k_final_v1"
EXPERIMENT = "apbr_weights_original_sparsegpt_seed012"
CALIBRATION_COUNT = 1024
CALIBRATION_SEED = 20261027
BLOCKSIZE = 128
PERCDAMP = 0.01


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


def atomic_torch(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return sha_file(path)


class CalibrationSubset(Dataset):
    def __init__(self, data, labels, ids, transform):
        self.data, self.labels, self.ids, self.transform = data, labels, list(ids), transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = int(self.ids[index])
        image = self.data[sample_id].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(image), int(self.labels[sample_id]), sample_id


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No layer diagnostics")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def prune_adapter(name: str, adapter: SparseGPTAdapter) -> tuple[torch.Tensor, list[dict]]:
    layer = adapter.layer
    matrix = weight_to_matrix(layer.weight)
    before = matrix.detach().clone()
    rows_per_group = matrix.shape[0] // adapter.groups
    keep = torch.empty_like(matrix, dtype=torch.bool)
    diagnostics = []
    for group, hessian in enumerate(adapter.hessians()):
        rows = slice(group * rows_per_group, (group + 1) * rows_per_group)
        result = sparsegpt_matrix(
            matrix[rows], hessian, keep_n=2, m=4, blocksize=BLOCKSIZE,
            percdamp=PERCDAMP, compute_dtype=torch.float32,
        )
        matrix[rows].copy_(result.weight)
        keep[rows] = result.keep_mask
        retained = result.keep_mask
        delta = result.weight - before[rows]
        diagonal = torch.diag(hessian)
        diagnostics.append({
            "layer": name,
            "group": group,
            "weight_rows": int(result.weight.shape[0]),
            "hessian_dimension": int(hessian.shape[0]),
            "activation_vectors": int(adapter.vectors),
            "hessian_diag_min": float(diagonal.min()),
            "hessian_diag_mean": float(diagonal.mean()),
            "hessian_diag_max": float(diagonal.max()),
            "applied_damping": result.damp,
            "decomposition_seconds": result.decomposition_seconds,
            "compensation_seconds": result.compensation_seconds,
            "retained_update_rms": float(delta[retained].float().square().mean().sqrt()),
            "deleted_weight_count": int((~retained).sum()),
            "retained_weight_count": int(retained.sum()),
            "deleted_weights_exact_zero": bool((result.weight[~retained] == 0).all()),
            "finite": bool(torch.isfinite(result.weight).all()),
        })
    return keep.reshape_as(layer.weight), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-method", choices=("dense_sam", "apbr_sam"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback forbidden")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL:
        raise ValueError("Wrong base protocol")
    formal = config["formal_calibration"]
    expected_calibration = {
        "samples": CALIBRATION_COUNT,
        "selection_seed": CALIBRATION_SEED,
        "batch_size": 128,
        "augmentation": "none",
    }
    for key, value in expected_calibration.items():
        if formal[key] != value:
            raise ValueError(f"Frozen calibration mismatch: {key}")

    parent_sha = sha_file(args.parent)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    identity = parent.get("identity", {})
    if (parent.get("protocol_id") != PROTOCOL or parent.get("epoch") != 200
            or parent.get("steps") != 78200 or parent.get("test_accessed") is not False
            or identity.get("method") != args.parent_method or identity.get("seed") != args.seed):
        raise ValueError("Dense parent identity mismatch")
    if not isinstance(parent.get("model"), dict):
        raise ValueError("Dense parent has no model state")

    train_file = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or labels.shape != (50_000,):
        raise ValueError("Official Train identity/shape mismatch")
    calibration_ids = np.random.default_rng(CALIBRATION_SEED).permutation(50_000)[:CALIBRATION_COUNT].tolist()
    ids_sha = hashlib.sha256(np.asarray(calibration_ids, dtype="<i8").tobytes()).hexdigest()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    batches = DataLoader(
        CalibrationSubset(data, labels, calibration_ids, transform), batch_size=128,
        shuffle=False, drop_last=False, num_workers=2, pin_memory=True,
    )

    device = torch.device("cuda:0")
    train.seed_all(args.seed)
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    layers = eligible_layers(model)
    if len(layers) != 19:
        raise ValueError(f"Expected 19 eligible layers, found {len(layers)}")
    adapters, hooks = {}, []
    for name, layer in layers.items():
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        adapters[name] = adapter
        hooks.append(layer.register_forward_pre_hook(lambda _module, inputs, item=adapter: item.add_batch(inputs[0])))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_started = time.perf_counter()
    calibration_started = time.perf_counter()
    seen_ids = []
    with torch.inference_mode():
        for inputs, _targets, sample_ids in batches:
            model(inputs.to(device, non_blocking=True))
            seen_ids.extend(map(int, sample_ids.tolist()))
    torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - calibration_started
    for hook in hooks:
        hook.remove()
    if seen_ids != calibration_ids:
        raise ValueError("Calibration sample order mismatch")

    masks, diagnostics = {}, []
    pruning_started = time.perf_counter()
    for name in list(layers):
        keep, rows = prune_adapter(name, adapters.pop(name))
        masks[name + ".weight"] = keep.detach().cpu()
        diagnostics.extend(rows)
    torch.cuda.synchronize()
    pruning_seconds = time.perf_counter() - pruning_started
    validate_masks(model, masks, n=2, m=4, complete=True)
    eligible_positions = sum(mask.numel() for mask in masks.values())
    removed_positions = sum(int((~mask).sum()) for mask in masks.values())
    if eligible_positions != 11_157_504 or removed_positions != 5_578_752:
        raise ValueError("Formal 2:4 position/deletion count mismatch")
    for key, mask in masks.items():
        weight = dict(model.named_parameters())[key]
        if mask.dtype != torch.bool or not bool((weight.detach().cpu()[~mask] == 0).all()):
            raise ValueError(f"Mask/zero invariant failed: {key}")
    if not all(row["finite"] and row["deleted_weights_exact_zero"] for row in diagnostics):
        raise FloatingPointError("SparseGPT numerical invariant failed")

    model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {
        "schema": 1,
        "status": "healthy",
        "protocol_id": PROTOCOL,
        "experiment_id": EXPERIMENT,
        "stage": "raw_hard",
        "method": "original_sparsegpt",
        "seed": args.seed,
        "parent_method": args.parent_method,
        "model": model_state,
        "masks": masks,
        "mask_sha256": train.tensor_sha(masks),
        "parent": {"path": str(args.parent), "sha256": parent_sha},
        "calibration_ids": calibration_ids,
        "calibration_ids_sha256": ids_sha,
        "algorithm": {
            "dynamic_mask": True,
            "external_mask": False,
            "apbr_geometry_used": False,
            "keep_n": 2,
            "m": 4,
            "blocksize": BLOCKSIZE,
            "percdamp": PERCDAMP,
            "compute_dtype": "float32",
            "recovery_epochs": 0,
            "bn_calibration": False,
        },
        "test_accessed": False,
    }
    checkpoint_path = output / "sparsegpt_final.pt"
    checkpoint_sha = atomic_torch(checkpoint_path, payload)
    mask_sha = atomic_torch(output / "mask.pt", masks)
    write_csv(output / "layer_hessian_diagnostics.csv", diagnostics)
    write_csv(output / "layer_compensation_diagnostics.csv", diagnostics)
    atomic_json(output / "calibration_manifest.json", {
        "source": "official_cifar100_train_only",
        "train_pickle_sha256": sha_file(train_file),
        "count": CALIBRATION_COUNT,
        "selection_seed": CALIBRATION_SEED,
        "ids": calibration_ids,
        "ids_sha256": ids_sha,
        "order_verified": True,
        "augmentation": "none",
        "test_accessed": False,
    })
    torch.cuda.synchronize()
    status = {
        "status": "completed",
        "protocol_id": PROTOCOL,
        "experiment_id": EXPERIMENT,
        "seed": args.seed,
        "parent_method": args.parent_method,
        "parent_sha256": parent_sha,
        "checkpoint_sha256": checkpoint_sha,
        "mask_file_sha256": mask_sha,
        "mask_tensor_sha256": payload["mask_sha256"],
        "eligible_layers": len(layers),
        "eligible_positions": eligible_positions,
        "removed_positions": removed_positions,
        "calibration_seconds": calibration_seconds,
        "pruning_seconds": pruning_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "test_accessed": False,
        "recovery_epochs": 0,
        "automatic_retry": False,
    }
    atomic_json(output / "status.json", status)
    print(json.dumps(status, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            atomic_json(target / "failure.json", {
                "status": "failed",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "automatic_retry": False,
            })
        raise
