"""Read-only, one-layer GPU replay diagnosis for layer1.0.conv1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_apbr_weights_original_sparsegpt_one as common
import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import grouped_patch_matrices, weight_to_matrix
from baseline_adapters.fixedmask_obs import fixedmask_obs_matrix
from baseline_adapters.group_joint_obs import prepare_obs_system
from baseline_adapters.sparsegpt import SparseGPTAdapter, sparsegpt_matrix


LAYER = "layer1.0.conv1"
CALIBRATION_SEED = 20261027
CALIBRATION_COUNT = 1024
BLOCKSIZE = 128
PERCDAMP = 0.01


def tensor_hash(value: torch.Tensor) -> str:
    x = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(x.dtype).encode())
    digest.update(str(tuple(x.shape)).encode())
    digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    a = left.detach().float().cpu()
    b = right.detach().float().cpu()
    delta = a - b
    denominator = float(b.norm())
    max_absolute_error = float(delta.abs().max()) if delta.numel() else 0.0
    return {
        "bitwise_equal": bool(torch.equal(left.detach().cpu(), right.detach().cpu())),
        "max_absolute_error": max_absolute_error,
        "relative_l2_error": float(delta.norm()) / denominator if denominator else 0.0,
    }


def trace_updates(weight, factor, external_keep=None):
    work = weight.clone()
    output = torch.zeros_like(work)
    keep = torch.ones_like(work, dtype=torch.bool)
    traces = []
    complete_end = work.shape[1] // 4 * 4
    for block_start in range(0, work.shape[1], BLOCKSIZE):
        block_end = min(block_start + BLOCKSIZE, work.shape[1])
        block = work[:, block_start:block_end].clone()
        quantized = torch.zeros_like(block)
        errors = torch.zeros_like(block)
        local_factor = factor[block_start:block_end, block_start:block_end]
        if external_keep is None:
            local_prune = torch.zeros_like(block, dtype=torch.bool)
        else:
            local_prune = ~external_keep[:, block_start:block_end]
        for local_column in range(block_end - block_start):
            global_column = block_start + local_column
            if external_keep is None and global_column < complete_end and global_column % 4 == 0:
                metric = block[:, local_column:local_column + 4].square() / torch.diag(local_factor)[
                    local_column:local_column + 4
                ].square()[None, :]
                removed = metric.argsort(dim=1, stable=True)[:, :2]
                local_prune.scatter_(1, local_column + removed, True)
            current = block[:, local_column]
            projected = current.clone()
            projected[local_prune[:, local_column]] = 0
            quantized[:, local_column] = projected
            keep[:, global_column] = ~local_prune[:, local_column]
            error = (current - projected) / local_factor[local_column, local_column]
            block[:, local_column:] -= error[:, None] @ local_factor[local_column, local_column:][None, :]
            errors[:, local_column] = error
        output[:, block_start:block_end] = quantized
        work[:, block_end:] -= errors @ factor[block_start:block_end, block_end:]
        traces.append({
            "block_start": block_start,
            "block_end": block_end,
            "quantized": quantized.detach().clone(),
            "future_work": work[:, block_end:].detach().clone(),
            "mask": keep[:, block_start:block_end].detach().clone(),
        })
    return output, keep, traces


def library_versions():
    commands = {
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        "ldconfig_cuda": ["bash", "-lc", "ldconfig -p | grep -E 'libcublas|libcusolver|libcudnn' || true"],
        "pip_nvidia": [sys.executable, "-m", "pip", "list", "--format=freeze"],
    }
    result = {}
    for name, command in commands.items():
        text = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        if name == "pip_nvidia":
            text = "\n".join(line for line in text.splitlines() if line.lower().startswith(("nvidia-", "torch", "torchvision")))
        result[name] = text.strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--historical-native", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU required")
    train.seed_all(0)
    device = torch.device("cuda:0")
    config = yaml.safe_load(args.config.read_text())
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    historical = torch.load(args.historical_native, map_location="cpu", weights_only=False)
    parent_sha = train.digest_file(args.parent)
    if historical.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError("Parent mismatch")
    train_file = args.dataset_root / "extracted/cifar-100-python/train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    calibration_ids = np.random.default_rng(CALIBRATION_SEED).permutation(50_000)[:CALIBRATION_COUNT].tolist()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    loader = DataLoader(
        common.CalibrationSubset(data, labels, calibration_ids, transform),
        batch_size=128, shuffle=False, drop_last=False, num_workers=2, pin_memory=True,
    )
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    layer = dict(model.named_modules())[LAYER]
    adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
    x_hash = hashlib.sha256()
    patch_hash = hashlib.sha256()
    x_batches = patch_columns = 0

    def capture(_module, values):
        nonlocal x_batches, patch_columns
        x = values[0]
        adapter.add_batch(x)
        x_cpu = x.detach().cpu().contiguous()
        x_hash.update(x_cpu.numpy().tobytes())
        patches = grouped_patch_matrices(layer, x)[0]
        p_cpu = patches.detach().cpu().contiguous()
        patch_hash.update(p_cpu.numpy().tobytes())
        x_batches += x.shape[0]
        patch_columns += patches.shape[1]

    hook = layer.register_forward_pre_hook(capture)
    with torch.inference_mode():
        for inputs, _labels, _ids in loader:
            model(inputs.to(device, non_blocking=True))
    hook.remove()
    hessian = adapter.hessians()[0]
    system = prepare_obs_system(hessian, percdamp=PERCDAMP, compute_dtype=torch.float32)
    weight = weight_to_matrix(layer.weight.detach())
    native = sparsegpt_matrix(
        weight, hessian, keep_n=2, m=4, blocksize=BLOCKSIZE,
        percdamp=PERCDAMP, compute_dtype=torch.float32,
    )
    fixed = fixedmask_obs_matrix(
        weight, hessian, native.keep_mask, keep_n=2, m=4, blocksize=BLOCKSIZE,
        percdamp=PERCDAMP, compute_dtype=torch.float32,
    )
    historical_weight = weight_to_matrix(historical["model"][LAYER + ".weight"].to(device))
    historical_mask = weight_to_matrix(historical["masks"][LAYER + ".weight"].to(device))
    dead_weight = weight.clone()
    dead_weight[:, system.dead_columns] = 0
    dynamic_output, dynamic_mask, dynamic_trace = trace_updates(dead_weight, system.inverse_factor)
    fixed_output, fixed_mask, fixed_trace = trace_updates(dead_weight, system.inverse_factor, native.keep_mask)

    if not torch.equal(native.weight, dynamic_output) or not torch.equal(native.keep_mask, dynamic_mask):
        raise RuntimeError("Independent dynamic trace does not reproduce native adapter")
    if not torch.equal(fixed.weight, fixed_output) or not torch.equal(fixed.keep_mask, fixed_mask):
        raise RuntimeError("Independent fixed trace does not reproduce fixed adapter")

    block_rows = []
    for dynamic, replay in zip(dynamic_trace, fixed_trace):
        qcmp = compare(dynamic["quantized"], replay["quantized"])
        fcmp = compare(dynamic["future_work"], replay["future_work"])
        block_rows.append({
            "block_start": dynamic["block_start"], "block_end": dynamic["block_end"],
            "dynamic_quantized_sha256": tensor_hash(dynamic["quantized"]),
            "fixed_quantized_sha256": tensor_hash(replay["quantized"]),
            "quantized_bitwise_equal": qcmp["bitwise_equal"],
            "quantized_max_absolute_error": qcmp["max_absolute_error"],
            "quantized_relative_l2_error": qcmp["relative_l2_error"],
            "future_work_bitwise_equal": fcmp["bitwise_equal"],
            "future_work_max_absolute_error": fcmp["max_absolute_error"],
            "future_work_relative_l2_error": fcmp["relative_l2_error"],
            "mask_bitwise_equal": bool(torch.equal(dynamic["mask"], replay["mask"])),
        })

    current_historical = compare(native.weight, historical_weight)
    current_fixed = compare(native.weight, fixed.weight)
    rows = [
        {"stage": "dense_weight_W", "shape": list(weight.shape), "dtype": str(weight.dtype),
         "current_sha256": tensor_hash(weight), "historical_sha256": tensor_hash(weight),
         "max_absolute_error": 0.0, "relative_l2_error": 0.0, "bitwise_equal": True,
         "historical_available": True},
        {"stage": "calibration_input_X", "shape": [x_batches, layer.in_channels, 32, 32], "dtype": "torch.float32",
         "current_sha256": x_hash.hexdigest(), "historical_sha256": "", "max_absolute_error": "",
         "relative_l2_error": "", "bitwise_equal": "", "historical_available": False},
        {"stage": "unfold_patch_matrix_stream", "shape": [weight.shape[1], patch_columns], "dtype": "torch.float32",
         "current_sha256": patch_hash.hexdigest(), "historical_sha256": "", "max_absolute_error": "",
         "relative_l2_error": "", "bitwise_equal": "", "historical_available": False},
        {"stage": "raw_hessian_H", "shape": list(hessian.shape), "dtype": str(hessian.dtype),
         "current_sha256": tensor_hash(hessian), "historical_sha256": "", "max_absolute_error": "",
         "relative_l2_error": "", "bitwise_equal": "", "historical_available": False},
        {"stage": "damped_hessian", "shape": list(system.damped_hessian.shape), "dtype": str(system.damped_hessian.dtype),
         "current_sha256": tensor_hash(system.damped_hessian), "historical_sha256": "", "max_absolute_error": "",
         "relative_l2_error": "", "bitwise_equal": "", "historical_available": False},
        {"stage": "inverse_hessian_Q", "shape": list(system.inverse_hessian.shape), "dtype": str(system.inverse_hessian.dtype),
         "current_sha256": tensor_hash(system.inverse_hessian), "historical_sha256": "", "max_absolute_error": "",
         "relative_l2_error": "", "bitwise_equal": "", "historical_available": False},
        {"stage": "upper_factor_U", "shape": list(system.inverse_factor.shape), "dtype": str(system.inverse_factor.dtype),
         "current_sha256": tensor_hash(system.inverse_factor), "historical_sha256": "", "max_absolute_error": "",
         "relative_l2_error": "", "bitwise_equal": "", "historical_available": False},
        {"stage": "native_mask", "shape": list(native.keep_mask.shape), "dtype": str(native.keep_mask.dtype),
         "current_sha256": tensor_hash(native.keep_mask), "historical_sha256": tensor_hash(historical_mask),
         **compare(native.keep_mask, historical_mask), "historical_available": True},
        {"stage": "per_block_obs_current_A_vs_B", "shape": [len(block_rows)], "dtype": "mixed",
         "current_sha256": hashlib.sha256(json.dumps(block_rows, sort_keys=True).encode()).hexdigest(),
         "historical_sha256": "", "max_absolute_error": max(r["quantized_max_absolute_error"] for r in block_rows),
         "relative_l2_error": max(r["quantized_relative_l2_error"] for r in block_rows),
         "bitwise_equal": all(r["quantized_bitwise_equal"] and r["future_work_bitwise_equal"] for r in block_rows),
         "historical_available": False},
        {"stage": "final_weight_current_A_vs_historical_C", "shape": list(native.weight.shape), "dtype": str(native.weight.dtype),
         "current_sha256": tensor_hash(native.weight), "historical_sha256": tensor_hash(historical_weight),
         **current_historical, "historical_available": True},
        {"stage": "final_weight_current_A_vs_fixed_B", "shape": list(native.weight.shape), "dtype": str(native.weight.dtype),
         "current_sha256": tensor_hash(native.weight), "historical_sha256": tensor_hash(fixed.weight),
         **current_fixed, "historical_available": True},
    ]
    with (output / "INTERMEDIATE_TENSOR_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output / "block_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(block_rows[0]))
        writer.writeheader(); writer.writerows(block_rows)

    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_enabled": torch.is_autocast_enabled(),
        "autocast_gpu_dtype": str(torch.get_autocast_dtype("cuda")),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "actual_weight_dtype": str(weight.dtype), "actual_hessian_dtype": str(hessian.dtype),
        "actual_factor_dtype": str(system.inverse_factor.dtype),
        "libraries": library_versions(), "test_accessed": False,
    }
    train.atomic_json(output / "environment_current.json", environment)
    train.atomic_json(output / "diagnostic.json", {
        "status": "SINGLE_LAYER_DIAGNOSIS_COMPLETE", "layer": LAYER,
        "current_native_vs_fixed": current_fixed,
        "current_native_vs_historical": current_historical,
        "mask_current_vs_historical": compare(native.keep_mask, historical_mask),
        "all_blocks_current_native_vs_fixed_bitwise": all(
            row["quantized_bitwise_equal"] and row["future_work_bitwise_equal"] and row["mask_bitwise_equal"]
            for row in block_rows
        ),
        "historical_intermediates_available": False,
        "test_accessed": False, "automatic_retry": False,
    })
    print(json.dumps(json.loads((output / "diagnostic.json").read_text())), flush=True)


if __name__ == "__main__":
    main()
