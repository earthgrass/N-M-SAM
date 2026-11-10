"""Fail-closed SparseGPT GPU gate for one K=4608 CIFAR ResNet18 Conv2d."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_adapters.common.conv2d import grouped_patch_matrices, weight_to_matrix
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.sparsegpt.adapter import sparsegpt_matrix
from baseline_adapters.cifar100_model import CIFAR100_MODEL_ID, build_cifar100_model

SEED = 20_261_026
LAYER_NAME = "layer4.0.conv2"
PERCDAMP = 0.01
BLOCKSIZE = 128


def sha256_tensor(value: torch.Tensor) -> str:
    item = value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(item.numpy())).hexdigest()


def nvidia_smi() -> dict[str, object]:
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        return {"available": run.returncode == 0, "returncode": run.returncode, "stdout": run.stdout.strip(), "stderr": run.stderr.strip()}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def base_environment() -> dict[str, object]:
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi": nvidia_smi(),
    }


def load_images(dataset_root: Path, dataset_audit: dict[str, object]) -> torch.Tensor:
    train_path = dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_path.open("rb") as stream:
        train = pickle.load(stream, encoding="bytes")
    ids = dataset_audit["fixed_gpu_smoke_train_sample_ids"]
    raw = np.asarray(train[b"data"], dtype=np.uint8)[ids].reshape(32, 3, 32, 32)
    value = torch.from_numpy(raw.copy()).to(torch.float32).div_(255.0)
    prep = dataset_audit["preprocessing_for_gpu_smoke"]
    mean = torch.tensor(prep["normalization_mean_train_only"]).reshape(1, 3, 1, 1)
    std = torch.tensor(prep["normalization_std_train_only_population"]).reshape(1, 3, 1, 1)
    return (value - mean) / std


def run_gate(dataset_root: Path, dataset_audit_path: Path, result: dict[str, object]) -> None:
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    torch.cuda.synchronize(device)
    free_before, total = torch.cuda.mem_get_info(device)
    result["gpu"] = {
        "name": props.name,
        "compute_capability": [props.major, props.minor],
        "total_memory_bytes_properties": props.total_memory,
        "total_memory_bytes_mem_get_info": total,
        "free_memory_before_bytes": free_before,
    }
    dataset_audit = json.loads(dataset_audit_path.read_text(encoding="utf-8"))
    if dataset_audit.get("status") != "DATA_IDENTITY_PASS":
        raise RuntimeError("Dataset identity is not PASS")
    images_cpu = load_images(dataset_root, dataset_audit)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = build_cifar100_model().eval().to(device)
    modules = dict(model.named_modules())
    layer = modules[LAYER_NAME]
    if list(layer.weight.shape) != [512, 512, 3, 3] or layer.weight[0].numel() != 4608:
        raise RuntimeError(f"Unexpected gate layer shape: {tuple(layer.weight.shape)}")
    captured: list[torch.Tensor] = []
    hook = layer.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach()))
    with torch.no_grad():
        prediction = model(images_cpu.to(device))
    hook.remove()
    if len(captured) != 1 or prediction.shape != (32, 100):
        raise RuntimeError("Activation capture failed")
    activation = captured[0]
    patches = grouped_patch_matrices(layer, activation)[0]
    padded = F.pad(activation, (1, 1, 1, 1))
    direct_first = padded[0, :, 0:3, 0:3].reshape(-1)
    patch_order_exact = torch.equal(patches[:, 0], direct_first)
    result["calibration"] = {
        "split": "train", "sample_ids": dataset_audit["fixed_gpu_smoke_train_sample_ids"],
        "sample_count": 32, "augmentation": "none", "input_shape": list(images_cpu.shape),
        "activation_shape": list(activation.shape), "patch_matrix_shape": list(patches.shape),
        "patch_order_direct_reference_exact": patch_order_exact,
    }
    result["model"] = {
        "model_id": CIFAR100_MODEL_ID, "randomly_initialized_not_trained": True,
        "initialization_seed": SEED, "layer_name": LAYER_NAME, "layer_weight_shape": list(layer.weight.shape),
        "k": layer.weight[0].numel(), "weight_sha256_before": sha256_tensor(layer.weight),
    }
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    h_started = time.perf_counter()
    hessian = 2.0 * (patches @ patches.T) / patches.shape[1]
    torch.cuda.synchronize(device)
    hessian_seconds = time.perf_counter() - h_started
    reference_indices = torch.linspace(0, hessian.shape[0] - 1, 64, device=device).round().long().unique()
    selected = patches.index_select(0, reference_indices).cpu().to(torch.float64)
    reference = 2.0 * (selected @ selected.T) / selected.shape[1]
    gpu_submatrix = hessian.index_select(0, reference_indices).index_select(1, reference_indices).cpu().to(torch.float64)
    reference_max_abs = float((gpu_submatrix - reference).abs().max())
    reference_relative_fro = float(torch.linalg.vector_norm(gpu_submatrix - reference) / torch.linalg.vector_norm(reference))
    matrix = weight_to_matrix(layer.weight)
    total_started = time.perf_counter()
    sparse = sparsegpt_matrix(
        matrix, hessian, keep_n=2, m=4, blocksize=BLOCKSIZE,
        percdamp=PERCDAMP, compute_dtype=torch.float32,
    )
    torch.cuda.synchronize(device)
    prune_seconds = time.perf_counter() - total_started
    with torch.no_grad():
        matrix.copy_(sparse.weight)
    validate_keep_mask(sparse.keep_mask, 2, 4)
    complete = sparse.keep_mask.reshape(512, -1).reshape(512, -1, 4).sum(dim=2)
    finite = bool(torch.isfinite(hessian).all() and torch.isfinite(layer.weight).all())
    # A controlled 6-column case exercises the dense-tail path with the same GPU FP32 implementation.
    tail_probe = sparsegpt_matrix(
        torch.tensor([[6., 5., 4., 3., 2., 1.]], device=device), torch.eye(6, device=device),
        keep_n=2, m=4, blocksize=4, percdamp=PERCDAMP, compute_dtype=torch.float32,
    )
    tail_dense = bool(tail_probe.keep_mask[:, 4:].all())
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    result["numerical"] = {
        "hessian_shape": list(hessian.shape), "hessian_dtype": str(hessian.dtype),
        "hessian_reference_dimension": len(reference_indices), "hessian_reference_cpu_dtype": "torch.float64",
        "hessian_reference_max_abs_error": reference_max_abs,
        "hessian_reference_relative_fro_error": reference_relative_fro,
        "relative_damping_fraction": PERCDAMP, "applied_damping": sparse.damp,
        "factorization": "Cholesky -> cholesky_inverse -> upper Cholesky compensation factor",
        "dynamic_sequential_mask": True, "compensated_weight_written_to_original_oihw_coordinates": True,
        "all_complete_groups_exactly_2_of_4": bool((complete == 2).all()),
        "dense_tail_probe": tail_dense, "keep_mask_dtype": str(sparse.keep_mask.dtype),
        "finite_outputs": finite, "oom": False, "cholesky_failure": False,
    }
    result["timing_seconds"] = {
        "hessian_build": hessian_seconds,
        "decomposition": sparse.decomposition_seconds,
        "compensation": sparse.compensation_seconds,
        "decomposition_and_compensation_observed_total": prune_seconds,
        "total_hessian_plus_prune": hessian_seconds + prune_seconds,
    }
    result["memory"] = {"peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved}
    result["model"]["weight_sha256_after"] = sha256_tensor(layer.weight)
    numerical_pass = all([
        patch_order_exact, reference_relative_fro <= 5e-5, finite,
        bool((complete == 2).all()), tail_dense, sparse.keep_mask.dtype == torch.bool,
    ])
    del tail_probe, sparse, matrix, hessian, patches, activation, captured, prediction, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    result["gpu"]["free_memory_after_release_bytes"] = free_after
    result["memory"]["memory_released_check"] = free_after >= free_before - 64 * 1024 * 1024
    result["statuses"]["SPARSEGPT_GPU_NUMERICAL_PASS"] = numerical_pass
    result["statuses"]["SPARSEGPT_GPU_RESOURCE_PASS"] = bool(result["memory"]["memory_released_check"])
    result["status"] = "PASS" if numerical_pass and result["statuses"]["SPARSEGPT_GPU_RESOURCE_PASS"] else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--model-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite gate result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "STARTED",
        "environment": base_environment(),
        "statuses": {
            "DATA_IDENTITY_PASS": None,
            "CIFAR100_MODEL_IDENTITY_PASS": None,
            "SPARSEGPT_GPU_NUMERICAL_PASS": False,
            "SPARSEGPT_GPU_RESOURCE_PASS": False,
            "FULL_EXPERIMENT_NOT_RUN": True,
        },
        "scope": "one internal 3x3 Conv2d only; no training, full-model pruning, recovery, or seed queue",
        "fixed_algorithm": {"keep_n": 2, "m": 4, "blocksize": BLOCKSIZE, "percdamp": PERCDAMP, "compute_dtype": "torch.float32", "retry_or_auto_tuning": False},
    }
    try:
        dataset = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
        model_audit = json.loads(args.model_audit.read_text(encoding="utf-8"))
        result["statuses"]["DATA_IDENTITY_PASS"] = dataset.get("status") == "DATA_IDENTITY_PASS"
        result["statuses"]["CIFAR100_MODEL_IDENTITY_PASS"] = model_audit.get("status") == "CIFAR100_MODEL_IDENTITY_PASS"
        result["dataset_identity"] = {
            "audit_path": str(args.dataset_audit.resolve()),
            "archive_sha256": dataset.get("archive", {}).get("sha256"),
            "fixed_gpu_smoke_train_sample_ids": dataset.get("fixed_gpu_smoke_train_sample_ids"),
        }
        result["model_identity"] = {
            "audit_path": str(args.model_audit.resolve()),
            "model_id": model_audit.get("model_id"),
            "architecture_sha256": model_audit.get("architecture_sha256"),
            "initialized_state_sha256": model_audit.get("initialized_state_sha256"),
        }
        if not torch.cuda.is_available():
            result["status"] = "BLOCKED_NO_CUDA"
            result["blocked_reason"] = "torch.cuda.is_available() is false; CPU execution is not accepted as a GPU gate"
            exit_code = 2
        else:
            run_gate(args.dataset_root, args.dataset_audit, result)
            exit_code = 0 if result["status"] == "PASS" else 1
    except BaseException as exc:
        result["status"] = "FAIL"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            result["failure"]["oom"] = True
        exit_code = 1
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
