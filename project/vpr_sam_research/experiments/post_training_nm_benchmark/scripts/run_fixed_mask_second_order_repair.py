"""Frozen-mask OBS-style compensation for APBR-parent CIFAR-100 ablations.

This runner never trains, calibrates BN, selects a support, or accesses Test.
The support is regenerated from the dense parent, matched elementwise to an
audited reference, and then passed into the existing FixedMaskOBSAdapter.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.fixedmask_obs import FixedMaskOBSAdapter
from benchmark.nm import eligible_layers, magnitude_mask
from run_apbr_seed0_fixedmask_obs import (
    CalibrationSubset,
    atomic_checkpoint,
    atomic_json,
    geometry_masks,
    quadratic_error,
    state_dict_cpu,
)

PROTOCOL = "cifar100_50k10k_final_v1"
EXPERIMENT = "fixed_mask_second_order_repair_cifar100_20261028"
CALIBRATION_SEED = 20261027
CALIBRATION_COUNT = 1024
PERCDAMP = 0.01
BLOCKSIZE = 128


def calibration_ids() -> list[int]:
    return np.random.default_rng(CALIBRATION_SEED).permutation(50_000)[:CALIBRATION_COUNT].tolist()


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def validate_parent(parent: dict, seed: int) -> None:
    if parent.get("protocol_id") != PROTOCOL:
        raise ValueError("Parent protocol mismatch")
    if parent.get("epoch") != 200 or parent.get("steps") != 78_200:
        raise ValueError("Parent is not fixed Epoch-200 Final")
    identity = parent.get("identity", {})
    if identity.get("method") != "apbr_sam" or identity.get("seed") != seed:
        raise ValueError("Parent is not the matching APBR-SAM seed")
    if parent.get("test_accessed") is not False:
        raise ValueError("Parent Test isolation mismatch")
    train.validate_checkpoint(parent, identity)


def frozen_masks(
    model: torch.nn.Module,
    parent: dict,
    reference: dict,
    parent_sha: str,
    seed: int,
    mask_source: str,
) -> dict[str, torch.Tensor]:
    if reference.get("seed") != seed or reference.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError("Mask reference is not bound to this parent/seed")
    if reference.get("test_accessed") is not False:
        raise ValueError("Mask reference Test isolation mismatch")
    if mask_source == "geometry":
        if reference.get("method") != "apbr_sam":
            raise ValueError("Geometry reference method mismatch")
        masks = geometry_masks(model, parent)
    else:
        if reference.get("method") != "magnitude":
            raise ValueError("Magnitude reference method mismatch")
        masks = {
            name + ".weight": magnitude_mask(layer.weight.detach(), 2, 4).cpu()
            for name, layer in eligible_layers(model).items()
        }
    expected = {name: value.bool().cpu() for name, value in reference.get("masks", {}).items()}
    if set(masks) != set(expected) or any(not torch.equal(masks[name], expected[name]) for name in masks):
        raise ValueError("Regenerated fixed support differs from audited reference")
    if train.tensor_sha(masks) != reference.get("mask_sha256"):
        raise ValueError("Fixed support hash mismatch")
    positions = sum(mask.numel() for mask in masks.values())
    deleted = sum(int((~mask).sum()) for mask in masks.values())
    if len(masks) != 19 or positions != 11_157_504 or deleted != 5_578_752:
        raise ValueError("Frozen 19-layer 2:4 count contract mismatch")
    for name, mask in masks.items():
        flat = mask.reshape(mask.shape[0], -1)
        end = flat.shape[1] // 4 * 4
        if end and not bool((flat[:, :end].reshape(flat.shape[0], -1, 4).sum(-1) == 2).all()):
            raise ValueError(f"Strict 2:4 support violation: {name}")
        if end < flat.shape[1] and not bool(flat[:, end:].all()):
            raise ValueError(f"Dense-tail support violation: {name}")
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--mask-reference", type=Path, required=True)
    parser.add_argument("--mask-source", choices=("geometry", "magnitude"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-layer", type=str)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device is required; CPU fallback is forbidden")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train.seed_all(args.seed)

    config = yaml.safe_load(args.formal_config.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL:
        raise ValueError("Formal config protocol mismatch")
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169_001_437
            or train.digest_file(archive, "md5") != config["dataset"]["archive_md5"]
            or train.digest_file(archive) != config["dataset"]["archive_sha256"]):
        raise ValueError("Official CIFAR-100 archive identity mismatch")
    with (args.dataset_root / "extracted/cifar-100-python/train").open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or not bool(np.all(np.bincount(labels, minlength=100) == 500)):
        raise ValueError("Official CIFAR-100 Train identity mismatch")

    parent_sha = train.digest_file(args.parent)
    reference_sha = train.digest_file(args.mask_reference)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    reference = torch.load(args.mask_reference, map_location="cpu", weights_only=False)
    validate_parent(parent, args.seed)
    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"])
    parent_state = {name: value.detach().cpu().clone() for name, value in parent["model"].items()}
    masks = frozen_masks(model, parent, reference, parent_sha, args.seed, args.mask_source)

    ids = calibration_ids()
    if "calibration_ids" in reference and reference["calibration_ids"] != ids:
        raise ValueError("Reference calibration IDs differ from frozen protocol")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    batches = DataLoader(
        CalibrationSubset(data, labels, ids, transform), batch_size=128, shuffle=False,
        drop_last=False, num_workers=2, pin_memory=True,
    )
    layers = eligible_layers(model)
    if args.smoke_layer:
        if args.smoke_layer not in layers:
            raise ValueError("Smoke layer is not eligible")
        layers = {args.smoke_layer: layers[args.smoke_layer]}

    torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()
    adapters = {name: FixedMaskOBSAdapter(layer, compute_dtype=torch.float32) for name, layer in layers.items()}
    hooks = [layer.register_forward_pre_hook(lambda _module, values, item=adapters[name]: item.add_batch(values[0]))
             for name, layer in layers.items()]
    model.eval()
    torch.cuda.synchronize(device)
    calibration_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for inputs, _, _ in batches:
                outputs = model(inputs.to(device, non_blocking=True))
                if not bool(torch.isfinite(outputs).all()):
                    raise FloatingPointError("Non-finite calibration output")
    finally:
        for hook in hooks:
            hook.remove()
    torch.cuda.synchronize(device)
    calibration_seconds = time.perf_counter() - calibration_started
    hessian_build_seconds = sum(item.hessian_build_seconds() for item in adapters.values())

    layer_reports = []
    for name, layer in layers.items():
        key = name + ".weight"
        before = layer.weight.detach().clone()
        adapter = adapters.pop(name)
        hessians = adapter.hessians().detach().clone()
        returned, results = adapter.prune(
            masks[key].to(device), keep_n=2, m=4, blocksize=BLOCKSIZE, percdamp=PERCDAMP,
        )
        after = layer.weight.detach()
        if not torch.equal(returned.cpu(), masks[key]) or torch.count_nonzero(after[~returned]):
            raise RuntimeError(f"Fixed support violation: {key}")
        if not bool(torch.isfinite(after).all()):
            raise FloatingPointError(f"Non-finite compensated weights: {key}")
        raw_error = obs_error = 0.0
        before_matrix = before.reshape(before.shape[0], -1)
        raw_matrix = (before * returned).reshape(before.shape[0], -1)
        after_matrix = after.reshape(after.shape[0], -1)
        rows_per_group = before.shape[0] // layer.groups
        for group, hessian in enumerate(hessians):
            rows = slice(group * rows_per_group, (group + 1) * rows_per_group)
            raw_error += quadratic_error(before_matrix[rows], raw_matrix[rows], hessian)
            obs_error += quadratic_error(before_matrix[rows], after_matrix[rows], hessian)
        layer_reports.append({
            "seed": args.seed, "mask_source": args.mask_source, "layer": name,
            "weight_shape": list(before.shape), "K": int(before[0].numel()),
            "mask_unchanged": True, "deleted_zero": True, "finite": True,
            "survivor_changed_count": int(torch.count_nonzero(after[returned] != before[returned])),
            "relative_survivor_l2_update": float(torch.linalg.vector_norm((after-before)[returned]) / torch.linalg.vector_norm(before[returned])),
            "damp": [result.damp for result in results],
            "decomposition_seconds": sum(result.decomposition_seconds for result in results),
            "compensation_seconds": sum(result.compensation_seconds for result in results),
            "local_quadratic_error_raw": raw_error,
            "local_quadratic_error_compensated": obs_error,
            "local_error_ratio": obs_error / raw_error if raw_error else None,
        })
        del adapter, hessians, before
    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - total_started
    if not any(row["survivor_changed_count"] > 0 for row in layer_reports):
        raise RuntimeError("Compensation changed no retained weights")

    final_state = state_dict_cpu(model)
    changed_keys = {name + ".weight" for name in layers}
    if any(not torch.equal(value, parent_state[key]) for key, value in final_state.items() if key not in changed_keys):
        raise RuntimeError("Compensation changed excluded state or BN buffers")
    stage = "gpu_smoke" if args.smoke_layer else "raw_hard"
    timing = {
        "calibration_seconds": calibration_seconds,
        "hessian_build_seconds": hessian_build_seconds,
        "decomposition_seconds": sum(row["decomposition_seconds"] for row in layer_reports),
        "compensation_seconds": sum(row["compensation_seconds"] for row in layer_reports),
        "total_postprocess_seconds": total_seconds,
    }
    payload = {
        "schema": 1, "status": "PASS", "protocol_id": PROTOCOL, "experiment_id": EXPERIMENT,
        "stage": stage, "method": f"{args.mask_source}_fixedmask_obs", "seed": args.seed,
        "epoch": 0, "recovery_epochs": 0, "model": final_state,
        "model_sha256": train.tensor_sha(final_state), "masks": masks,
        "mask_sha256": train.tensor_sha(masks),
        "parent": {"path": str(args.parent.resolve()), "sha256": parent_sha},
        "mask_reference": {"path": str(args.mask_reference.resolve()), "sha256": reference_sha},
        "calibration_ids": ids, "calibration_ids_sha256": ids_sha256(ids),
        "dynamic_mask_selection": False, "gradient_training": False, "bn_calibration": False,
        "test_accessed": False, "tf32": False, "amp": False, "compile": False,
        "timing": timing, "layer_reports": layer_reports,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "source_hashes": {
            "runner": train.digest_file(Path(__file__)),
            "adapter": train.digest_file(ROOT / "baseline_adapters/fixedmask_obs/adapter.py"),
            "formal_config": train.digest_file(args.formal_config),
        },
    }
    checkpoint_path = output / ("smoke_checkpoint.pt" if args.smoke_layer else "compensated.pt")
    checkpoint_sha = atomic_checkpoint(checkpoint_path, payload)
    atomic_json(output / "evidence.json", {
        key: value for key, value in payload.items() if key not in {"model", "masks"}
    } | {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha})
    print(json.dumps({"status": "PASS", "stage": stage, "checkpoint_sha256": checkpoint_sha,
                      "timing": timing, "layers": len(layer_reports)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            directory = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            failure = directory / "failure.json"
            if not failure.exists():
                atomic_json(failure, {"status": "FAILED", "type": type(error).__name__,
                                     "message": str(error), "traceback": traceback.format_exc(),
                                     "automatic_retry": False})
        raise
