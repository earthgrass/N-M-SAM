"""Phase-0 CUDA smokes for training, full-model pruning, BN and recovery.

All models are randomly initialized.  This script measures implementation and
resource paths only and never reports accuracy as an experiment result.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.rose import ROSEAdapter
from baseline_adapters.sparsegpt import SparseGPTAdapter
from baseline_adapters.wanda import WandaAdapter
from benchmark.nm import eligible_layers, magnitude_mask, project, validate_masks

SEED = 20_261_026


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_map_sha(values: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        item = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(item.dtype).encode())
        h.update(str(tuple(item.shape)).encode())
        h.update(memoryview(item.numpy()))
    return h.hexdigest()


def load_train(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    with (dataset_root / "extracted" / "cifar-100-python" / "train").open("rb") as stream:
        value = pickle.load(stream, encoding="bytes")
    return np.asarray(value[b"data"], dtype=np.uint8), np.asarray(value[b"fine_labels"], dtype=np.int64)


def batch(data: np.ndarray, labels: np.ndarray, ids: list[int], mean: list[float], std: list[float], augment: bool) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(data[ids].copy().reshape(-1, 3, 32, 32)).float().div_(255.0)
    if augment:
        augmented = torch.empty_like(x)
        for position, sample_id in enumerate(ids):
            generator = torch.Generator().manual_seed(SEED * 1_000_003 + int(sample_id))
            top, left = torch.randint(0, 9, (2,), generator=generator).tolist()
            item = F.pad(x[position], (4, 4, 4, 4))[:, top : top + 32, left : left + 32]
            if torch.rand((), generator=generator).item() < 0.5:
                item = item.flip(2)
            augmented[position] = item
        x = augmented
    mu = torch.tensor(mean).reshape(1, 3, 1, 1)
    sigma = torch.tensor(std).reshape(1, 3, 1, 1)
    return (x - mu) / sigma, torch.from_numpy(labels[ids].copy()).long()


def cuda_memory(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": free, "total_bytes": total}


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def training_smoke(method: str, initial: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor, device: torch.device, vpr_src: Path) -> dict[str, object]:
    if str(vpr_src) not in sys.path:
        sys.path.insert(0, str(vpr_src))
    from vpr_sam.optim import Stepper

    model = build_cifar100_model().to(device)
    model.load_state_dict(initial)
    config = {
        "seed": 0,
        "optimizer": {"method": method, "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": False, "rho": 0.05},
        "geometry": {"gamma": 5.482145661475941e-8, "dtype": "float64", "warmup_steps": 0, "volume_atol": 1e-10, "volume_rtol": 1e-12, "static_std": 0.1},
    }
    before = tensor_map_sha(dict(model.named_parameters()))
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = cuda_memory(device)
    stepper = Stepper(model, config)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    info = stepper.step(x.to(device), y.to(device))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    after = tensor_map_sha(dict(model.named_parameters()))
    result = {
        "status": "PASS",
        "method": "dense_sam" if method == "sam" else "apbr_sam_vpr",
        "single_batch_only": True,
        "batch_size": len(y),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "memory_before": memory_before,
        "finite": all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
        "weights_changed": before != after,
        "center_loss": info["center_loss"],
        "adversarial_loss": info["adversarial_loss"],
        "epsilon_norm": info["eps_norm"],
        "ellipsoid_norm": info["ellipsoid_norm"],
        "geometry_updates": info["geometry_updates"],
        "output_classes": 100,
    }
    del stepper, model
    clear_cuda()
    result["memory_after_cleanup"] = cuda_memory(device)
    return result


def prune_smoke(method: str, initial: dict[str, torch.Tensor], x: torch.Tensor, device: torch.device) -> dict[str, object]:
    model = build_cifar100_model().to(device).eval()
    model.load_state_dict(initial)
    layers = eligible_layers(model)
    adapters: dict[str, object] = {}
    hooks = []
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = cuda_memory(device)
    collect_started = time.perf_counter()
    if method != "magnitude":
        for name, layer in layers.items():
            if method == "wanda":
                adapter = WandaAdapter(layer)
            elif method == "sparsegpt":
                adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
            elif method == "rose":
                adapter = ROSEAdapter(layer, compute_dtype=torch.float32)
            else:
                raise ValueError(method)
            adapters[name] = adapter
            hooks.append(layer.register_forward_pre_hook(lambda _module, inputs, item=adapter: item.add_batch(inputs[0])))
        with torch.no_grad():
            logits_before = model(x.to(device))
        for hook in hooks:
            hook.remove()
    else:
        with torch.no_grad():
            logits_before = model(x.to(device))
    torch.cuda.synchronize(device)
    collect_seconds = time.perf_counter() - collect_started
    masks: dict[str, torch.Tensor] = {}
    layer_records = []
    prune_started = time.perf_counter()
    for name, layer in layers.items():
        layer_started = time.perf_counter()
        if method == "magnitude":
            keep = magnitude_mask(layer.weight, n=2, m=4)
            with torch.no_grad():
                layer.weight.mul_(keep)
            extra = {}
        elif method == "wanda":
            keep = adapters[name].prune(keep_n=2, m=4)
            extra = {}
        elif method == "sparsegpt":
            keep = adapters[name].prune(keep_n=2, m=4, blocksize=128, percdamp=0.01)
            extra = {}
        else:
            keep, orders = adapters[name].prune(keep_n=2, m=4, blocksize=128, percdamp=0.01, threshold=0.5)
            extra = {"reordered_groups": sum(order.reordered for order in orders), "max_relative_range": max((order.relative_range for order in orders), default=0.0)}
        masks[name + ".weight"] = keep.detach().cpu()
        torch.cuda.synchronize(device)
        layer_records.append({"name": name, "weight_shape": list(layer.weight.shape), "k": layer.weight[0].numel(), "seconds": time.perf_counter() - layer_started, **extra})
        if method != "magnitude":
            del adapters[name]
        clear_cuda()
    torch.cuda.synchronize(device)
    prune_seconds = time.perf_counter() - prune_started
    validate_masks(model, masks, n=2, m=4, complete=True)
    with torch.no_grad():
        logits_after = model(x.to(device))
    finite = bool(torch.isfinite(logits_before).all() and torch.isfinite(logits_after).all())
    result = {
        "status": "PASS" if finite else "FAIL",
        "method": method,
        "randomly_initialized_not_accuracy": True,
        "calibration_samples": len(x),
        "calibration_augmentation": "none",
        "eligible_layers": len(layers),
        "collect_seconds": collect_seconds,
        "prune_seconds": prune_seconds,
        "total_seconds": collect_seconds + prune_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "memory_before": memory_before,
        "finite": finite,
        "mask_sha256": tensor_map_sha(masks),
        "pruned_state_sha256": tensor_map_sha(model.state_dict()),
        "layers": layer_records,
        "oom": False,
    }
    del adapters, hooks, masks, logits_before, logits_after, model
    clear_cuda()
    result["memory_after_cleanup"] = cuda_memory(device)
    return result


def recovery_and_bn_smoke(initial: dict[str, torch.Tensor], train_x: torch.Tensor, train_y: torch.Tensor, calibration_x: torch.Tensor, device: torch.device) -> dict[str, object]:
    model = build_cifar100_model().to(device)
    model.load_state_dict(initial)
    masks = {name + ".weight": magnitude_mask(layer.weight, 2, 4).cpu() for name, layer in eligible_layers(model).items()}
    project(model, masks)
    validate_masks(model, masks)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=False)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = cuda_memory(device)
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(train_x.to(device)), train_y.to(device))
    loss.backward()
    project(model, masks, optimizer)
    optimizer.step()
    project(model, masks, optimizer)
    validate_masks(model, masks)
    momentum_zero = True
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        parameter = parameters[name]
        buffer = optimizer.state[parameter].get("momentum_buffer")
        if buffer is not None and torch.count_nonzero(buffer[~mask.to(device)]):
            momentum_zero = False
    torch.cuda.synchronize(device)
    recovery_seconds = time.perf_counter() - started
    parameters_before_bn = {name: value.detach().clone() for name, value in model.named_parameters()}
    buffers_before_bn = {name: value.detach().clone() for name, value in model.named_buffers()}
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
    model.train()
    with torch.no_grad():
        model(calibration_x.to(device))
    parameters_unchanged = all(torch.equal(parameters_before_bn[name], value) for name, value in model.named_parameters())
    changed_buffers = [name for name, value in model.named_buffers() if not torch.equal(buffers_before_bn[name], value)]
    allowed = all(any(token in name for token in ("running_mean", "running_var", "num_batches_tracked")) for name in changed_buffers)
    result = {
        "status": "PASS" if momentum_zero and parameters_unchanged and allowed and torch.isfinite(loss) else "FAIL",
        "recovery_single_step": True,
        "loss": float(loss.detach()),
        "masked_weights_zero": True,
        "masked_momentum_zero": momentum_zero,
        "mask_unchanged": True,
        "parameters_finite": all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
        "recovery_seconds": recovery_seconds,
        "bn_only_single_batch": True,
        "bn_parameters_unchanged": parameters_unchanged,
        "bn_changed_buffers": changed_buffers,
        "bn_only_changed_allowed_buffers": allowed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "memory_before": memory_before,
    }
    del optimizer, model, masks, parameters_before_bn, buffers_before_bn
    clear_cuda()
    result["memory_after_cleanup"] = cuda_memory(device)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--vpr-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-0 GPU smoke requires CUDA; CPU fallback forbidden")
    device = torch.device("cuda:0")
    data, labels = load_train(args.dataset_root)
    manifest = json.loads((args.split_dir / "CIFAR100_DATA_SPLIT_MANIFEST.json").read_text())
    train_ids = json.loads((args.split_dir / "indices" / "train_45000.json").read_text())
    calibration_ids = json.loads((args.split_dir / "indices" / "calibration_1024.json").read_text())
    mean = manifest["normalization"]["population_mean"]
    std = manifest["normalization"]["population_std"]
    train_x, train_y = batch(data, labels, train_ids[:128], mean, std, augment=True)
    prune_x, _ = batch(data, labels, calibration_ids[:32], mean, std, augment=False)
    bn_x, _ = batch(data, labels, train_ids[128:256], mean, std, augment=False)
    torch.manual_seed(SEED)
    initial_model = build_cifar100_model()
    initial = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    del initial_model
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "STARTED",
        "scope": "Phase-0 random-initialization GPU implementation/resource smoke; no accuracy claim",
        "seed": SEED,
        "gpu": {"name": torch.cuda.get_device_name(0), "properties": str(torch.cuda.get_device_properties(0))},
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "split_manifest_sha256": file_sha(args.split_dir / "CIFAR100_DATA_SPLIT_MANIFEST.json"),
        "source_sha256": {
            "rose_adapter": file_sha(ROOT / "baseline_adapters" / "rose" / "adapter.py"),
            "sparsegpt_adapter": file_sha(ROOT / "baseline_adapters" / "sparsegpt" / "adapter.py"),
            "wanda_adapter": file_sha(ROOT / "baseline_adapters" / "wanda" / "adapter.py"),
            "phase0_script": file_sha(Path(__file__)),
        },
        "training": {}, "pruning": {}, "recovery_bn": None,
    }
    exit_code = 0
    for method in ("sam", "vpr"):
        try:
            payload["training"][method] = training_smoke(method, initial, train_x, train_y, device, args.vpr_src)
        except BaseException as exc:
            payload["training"][method] = {"status": "FAIL", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            exit_code = 1
    for method in ("magnitude", "wanda", "sparsegpt", "rose"):
        try:
            payload["pruning"][method] = prune_smoke(method, initial, prune_x, device)
        except torch.cuda.OutOfMemoryError as exc:
            payload["pruning"][method] = {"status": "FAIL", "oom": True, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            clear_cuda(); exit_code = 1
        except BaseException as exc:
            payload["pruning"][method] = {"status": "FAIL", "oom": False, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            clear_cuda(); exit_code = 1
    try:
        payload["recovery_bn"] = recovery_and_bn_smoke(initial, train_x, train_y, bn_x, device)
        exit_code |= payload["recovery_bn"]["status"] != "PASS"
    except BaseException as exc:
        payload["recovery_bn"] = {"status": "FAIL", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        exit_code = 1
    statuses = [item["status"] for item in payload["training"].values()] + [item["status"] for item in payload["pruning"].values()] + [payload["recovery_bn"]["status"]]
    payload["status"] = "PASS" if all(value == "PASS" for value in statuses) else "FAIL"
    payload["full_experiment_not_run"] = True
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "training": {k:v["status"] for k,v in payload["training"].items()}, "pruning": {k:v["status"] for k,v in payload["pruning"].items()}, "recovery_bn": payload["recovery_bn"]["status"]}))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
