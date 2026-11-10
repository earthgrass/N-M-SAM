"""Train-only HQ calibration with explicit state isolation and CUDA timings.

One layer's Gram is collected and factorized before visiting the next layer.
Calibration batches are streamed through the fixed model for each layer. No Test loader,
automatic damping fallback, output-channel averaging, or BN update occurs.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from collections import OrderedDict
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .math_core import GeometrySpec, historical_geometry_update, target_from_inverse_diagonals


def synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_digest(mapping: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        value = mapping[name].detach().cpu().contiguous()
        # Canonical framing shared with training.tensor_sha checkpoints.
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def eligible_layers(model: nn.Module) -> OrderedDict[str, nn.Conv2d]:
    candidates = [(name, layer) for name, layer in model.named_modules()
                  if isinstance(layer, (nn.Conv2d, nn.Linear))]
    selected = candidates[1:-1]
    if not selected or any(not isinstance(layer, nn.Conv2d) for _, layer in selected):
        raise ValueError("Expected internal Conv2d eligibility between stem and head")
    return OrderedDict(selected)


def _rng_snapshot(loader) -> dict:
    generator = getattr(loader, "generator", None)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_generator": generator.get_state() if generator is not None else None}


def _rng_restore(snapshot: dict, loader) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])
    if snapshot["loader_generator"] is not None:
        loader.generator.set_state(snapshot["loader_generator"])


def _memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "current_allocated_bytes": torch.cuda.memory_allocated(device),
            "current_reserved_bytes": torch.cuda.memory_reserved(device),
            "memory_scope": "since most recent caller reset_peak_memory_stats"}


@torch.no_grad()
def collect_grams(model: nn.Module, loader, config: dict, device: torch.device | str, *, layer_names=None):
    """Return parameter-name -> [groups,K,K] FP32 Gram and audit report.

    H is XX^T / total_patch_columns, all calibration samples and spatial
    positions are retained. Default one GEMM per batch matches historical
    accumulation boundaries; an explicit positive gram_chunk_columns freezes
    alternative rounding boundaries without subsampling columns.
    """
    device = torch.device(device)
    synchronize(device)
    started = time.perf_counter()
    all_layers = eligible_layers(model)
    expected_layers = int(config.get("expected_layers", 19))
    expected_coordinates = int(config.get("expected_coordinates", 11_157_504))
    coordinates = sum(layer.weight.numel() for layer in all_layers.values())
    if len(all_layers) != expected_layers or coordinates != expected_coordinates:
        raise ValueError(f"Eligibility mismatch: {len(all_layers)} layers, {coordinates} coordinates")
    chosen = list(all_layers) if layer_names is None else [name.removesuffix(".weight") for name in layer_names]
    if len(chosen) != 1 or chosen[0] not in all_layers:
        raise ValueError("Exactly one eligible layer must be streamed and released at a time")
    layers = OrderedDict((name, all_layers[name]) for name in chosen)
    gram_bytes = sum(layer.groups * layer.weight[0].numel() ** 2 * 4 for layer in layers.values())
    if gram_bytes > int(config.get("max_gram_bytes", 512 * 2**20)):
        raise MemoryError(f"Gram allocation {gram_bytes} exceeds preregistered budget")
    chunk_columns = int(config.get("gram_chunk_columns", 0))
    if chunk_columns < 0:
        raise ValueError("gram_chunk_columns must be nonnegative")
    before_parameters = tensor_digest(dict(model.named_parameters()))
    before_buffers = tensor_digest(dict(model.named_buffers()))
    modes = {module: module.training for module in model.modules()}
    rng = _rng_snapshot(loader)
    accumulators, rows, hooks = {}, {}, []
    ids_digest = hashlib.sha256()
    ids_available, samples, batches = True, 0, 0
    try:
        model.eval()
        for name, layer in layers.items():
            if layer.padding_mode != "zeros":
                raise ValueError("Only zero-padding Conv2d is frozen for this calibration")
            key, features = name + ".weight", layer.weight[0].numel()
            if layer.weight.device != device:
                # CUDA aliases such as cuda vs cuda:0 are compared canonically.
                if layer.weight.device.type != device.type or (device.index is not None and layer.weight.device.index != device.index):
                    raise ValueError("Model and calibration device differ")
            accumulators[key] = torch.zeros((layer.groups, features, features), device=device, dtype=torch.float32)
            rows[key] = {"weight_shape": list(layer.weight.shape), "groups": layer.groups,
                         "features": features, "patch_columns": 0,
                         "unfold_seconds": 0.0, "gram_seconds": 0.0}

            def hook(_module, inputs, *, key=key, layer=layer, features=features):
                synchronize(device)
                t = time.perf_counter()
                inputs = inputs[0].detach()
                if inputs.ndim != 4 or inputs.shape[1] != layer.in_channels:
                    raise ValueError("Conv input mapping mismatch")
                patches = F.unfold(inputs, layer.kernel_size, dilation=layer.dilation,
                                   padding=layer.padding, stride=layer.stride)
                count, _, locations = patches.shape
                patches = patches.reshape(count, layer.groups, features, locations)
                matrices = [patches[:, group].permute(1, 0, 2).reshape(features, count * locations).float()
                            for group in range(layer.groups)]
                synchronize(device)
                rows[key]["unfold_seconds"] += time.perf_counter() - t
                t = time.perf_counter()
                for group, x in enumerate(matrices):
                    if not bool(torch.isfinite(x).all()):
                        raise FloatingPointError(f"Nonfinite calibration input: {key}")
                    if chunk_columns:
                        for start in range(0, x.shape[1], chunk_columns):
                            part = x[:, start:start + chunk_columns]
                            accumulators[key][group].addmm_(part, part.T)
                    else:
                        accumulators[key][group].addmm_(x, x.T)
                synchronize(device)
                rows[key]["gram_seconds"] += time.perf_counter() - t
                if config.get("reference_checks", False) and "reference_checks" not in rows[key]:
                    rows[key]["reference_checks"] = [bounded_reference_check(x, float(config.get("percdamp", .01)))
                                                     for x in matrices]
                rows[key]["patch_columns"] += count * locations

            hooks.append(layer.register_forward_pre_hook(hook))
        for batch in loader:
            inputs = batch[0].to(device, non_blocking=True)
            output = model(inputs)
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("Nonfinite calibration model output")
            samples += len(inputs)
            batches += 1
            if len(batch) >= 3:
                ids = torch.as_tensor(batch[2]).cpu().numpy().astype("<i8", copy=False)
                ids_digest.update(ids.tobytes())
            else:
                ids_available = False
        expected_samples = int(config.get("expected_samples", 1024))
        if samples != expected_samples:
            raise ValueError(f"Calibration count {samples} != frozen {expected_samples}")
        for key, gram in accumulators.items():
            vectors = rows[key]["patch_columns"]
            if not vectors:
                raise ValueError(f"Layer never observed: {key}")
            gram.div_(vectors)
            if not bool(torch.isfinite(gram).all()):
                raise FloatingPointError(f"Nonfinite Gram: {key}")
    finally:
        for hook_handle in hooks:
            hook_handle.remove()
        for module, mode in modes.items():
            module.training = mode
        _rng_restore(rng, loader)
    after_parameters = tensor_digest(dict(model.named_parameters()))
    after_buffers = tensor_digest(dict(model.named_buffers()))
    if after_parameters != before_parameters or after_buffers != before_buffers:
        raise RuntimeError("Calibration mutated model parameters or buffers")
    synchronize(device)
    report = {"status": "CALIBRATION_COLLECTION_PASS", "test_accessed": False,
              "hessian_definition": "XXT/n_all_patch_columns", "gram_dtype": "float32",
              "gram_chunk_columns": chunk_columns, "gram_storage_bytes": gram_bytes,
              "collection_policy": "single_layer_stream",
              "layer_count": len(layers), "eligible_coordinates": coordinates,
              "model_eligible_layer_count": len(all_layers),
              "calibration_samples": samples, "calibration_batches": batches,
              "calibration_ids_sha256": ids_digest.hexdigest() if ids_available else None,
              "collection_seconds": time.perf_counter() - started,
              "parameters_sha256_before": before_parameters, "parameters_sha256_after": after_parameters,
              "buffers_sha256_before": before_buffers, "buffers_sha256_after": after_buffers,
              "bn_unchanged": True, "parameters_unchanged": True, "rng_restored": True,
              "layers": rows, **_memory(device)}
    return accumulators, report


def collect_layer_gram(model, loader, layer_name, config, device):
    """Collect exactly one layer while retaining no other layer's Gram."""
    key = layer_name if layer_name.endswith(".weight") else layer_name + ".weight"
    grams, report = collect_grams(model, loader, config, device, layer_names=[key])
    return grams.pop(key), report


def inverse_diagonal_timed(gram: Tensor, percdamp: float):
    """Same Cholesky/triangular-solve mathematics as the CPU reference.

    Dead inputs receive the ordinary relative damping only. No unit diagonal
    replacement is applied to HQ H=XXT/n, unlike historical native SparseGPT.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.dtype not in (torch.float32, torch.float64):
        raise ValueError("Invalid Gram shape or dtype")
    if not math.isfinite(percdamp) or percdamp <= 0:
        raise ValueError("Require positive frozen relative damping")
    if not bool(torch.isfinite(gram).all()) or not torch.allclose(gram, gram.T, rtol=1e-6, atol=1e-8):
        raise ValueError("Invalid or materially asymmetric Gram")
    h = .5 * gram + .5 * gram.T
    correction = float((h - gram).abs().max())
    dead = int((h.diag() == 0).sum())
    delta = percdamp * h.diag().mean()
    if not bool(torch.isfinite(delta) and delta > 0):
        raise FloatingPointError("Invalid damping; no automatic fallback")
    h.diagonal().add_(delta)
    synchronize(h.device)
    start = time.perf_counter()
    lower = torch.linalg.cholesky(h)
    synchronize(h.device)
    cholesky_seconds = time.perf_counter() - start
    start = time.perf_counter()
    identity = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    inverse_lower = torch.linalg.solve_triangular(lower, identity, upper=False)
    q = inverse_lower.square().sum(dim=0)
    synchronize(h.device)
    inverse_seconds = time.perf_counter() - start
    if not bool(torch.isfinite(q).all() and (q > 0).all()):
        raise FloatingPointError("Nonfinite or nonpositive inverse diagonal")
    hdiag = h.diag()
    row = {"dead_coordinates": dead, "dead_policy": "ordinary_relative_damping_only",
           "damping": float(delta), "symmetry_correction_max_abs": correction,
           "cholesky_status": "PASS", "cholesky_seconds": cholesky_seconds,
           "inverse_diagonal_seconds": inverse_seconds,
           "damped_diagonal_ratio_proxy": float(hdiag.max() / hdiag.min()),
           "q_min": float(q.min()), "q_max": float(q.max()), "q_mean": float(q.double().mean()),
           "q_std": float(q.double().std(unbiased=False))}
    return q, row


def bounded_reference_check(matrix: Tensor, percdamp: float) -> dict:
    """Independent FP64 reference on actual bounded input, never full K=4608."""
    synchronize(matrix.device)
    started = time.perf_counter()
    sample = matrix[:min(32, matrix.shape[0]), :min(256, matrix.shape[1])].contiguous().float()
    if not sample.numel():
        raise ValueError("Empty reference patch sample")
    h32 = sample @ sample.T / sample.shape[1]
    cpu64 = sample.detach().cpu().double()
    h64 = cpu64 @ cpu64.T / cpu64.shape[1]
    denominator = max(float(h64.norm()), torch.finfo(torch.float64).tiny)
    gram_error = float((h32.detach().cpu().double() - h64).norm()) / denominator
    q32, _ = inverse_diagonal_timed(h32, percdamp)
    symmetric = .5 * h64 + .5 * h64.T
    damped64 = symmetric + percdamp * symmetric.diag().mean() * torch.eye(len(symmetric), dtype=torch.float64)
    # The tiny independent reference deliberately uses a direct inverse; the
    # production path exclusively uses Cholesky and triangular solves.
    q64 = torch.linalg.inv(damped64).diag()
    q_error = float((q32.detach().cpu().double() - q64).norm()) / max(float(q64.norm()), torch.finfo(torch.float64).tiny)
    if not math.isfinite(gram_error) or not math.isfinite(q_error) or gram_error > 5e-5 or q_error > 5e-4:
        raise FloatingPointError(f"Bounded FP64 reference failed: Gram={gram_error}, inverse_diag={q_error}")
    synchronize(matrix.device)
    return {"status": "PASS", "sample_shape": list(sample.shape),
            "sample_rule": "first_min32_features_first_min256_patch_columns_first_batch",
            "execution_device": str(matrix.device), "compute_dtype": "float32", "reference_dtype": "cpu_float64",
            "gram_relative_frobenius_error": gram_error, "gram_tolerance": 5e-5,
            "inverse_diagonal_relative_l2_error": q_error, "inverse_diagonal_tolerance": 5e-4,
            "reference_seconds": time.perf_counter() - started}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    sorted_ranks = np.repeat((starts + ends - 1) / 2, ends - starts)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def spearman(a: Tensor, b: Tensor) -> float | None:
    if a.numel() != b.numel():
        raise ValueError("Spearman shape mismatch")
    x = _average_ranks(a.detach().double().cpu().numpy().ravel())
    y = _average_ranks(b.detach().double().cpu().numpy().ravel())
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else None


@torch.no_grad()
def refresh(model, loader, geometry, config, device, previous_q=None):
    device = torch.device(device)
    synchronize(device)
    start = time.perf_counter()
    layers = eligible_layers(model)
    expected_keys = {name + ".weight" for name in layers}
    if set(geometry) != expected_keys:
        raise ValueError("Geometry keys do not match eligible parameters")
    specs, q_diags, collection_reports = [], {}, []
    report = {"layers": {}}
    for name, layer in layers.items():
        key = name + ".weight"
        group_q, diagnostics = [], []
        gram, collected = collect_layer_gram(model, loader, name, config, device)
        collection_reports.append(collected)
        report["layers"][key] = collected["layers"][key]
        for group in range(layer.groups):
            q, detail = inverse_diagonal_timed(gram[group], float(config.get("percdamp", .01)))
            group_q.append(q)
            diagnostics.append(detail)
        q_diags[key] = torch.stack(group_q)
        report["layers"][key]["groups_diagnostics"] = diagnostics
        report["layers"][key]["cholesky_seconds"] = sum(d["cholesky_seconds"] for d in diagnostics)
        report["layers"][key]["inverse_diagonal_seconds"] = sum(d["inverse_diagonal_seconds"] for d in diagnostics)
        report["layers"][key]["q_previous_spearman"] = (
            spearman(q_diags[key], previous_q[key]) if previous_q is not None else None)
        specs.append(GeometrySpec(key, tuple(layer.weight.shape), layer.groups, q_diags[key]))
        del gram
        synchronize(device)
    first = collection_reports[0]
    if any(row["calibration_ids_sha256"] != first["calibration_ids_sha256"]
           or row["parameters_sha256_before"] != first["parameters_sha256_before"]
           or row["buffers_sha256_before"] != first["buffers_sha256_before"] for row in collection_reports):
        raise RuntimeError("Layerwise calibration passes used different samples or model state")
    report.update({key: first[key] for key in ("test_accessed", "hessian_definition", "gram_dtype",
                  "gram_chunk_columns", "collection_policy", "eligible_coordinates", "calibration_samples",
                  "calibration_batches", "calibration_ids_sha256", "parameters_sha256_before",
                  "parameters_sha256_after", "buffers_sha256_before", "buffers_sha256_after",
                  "bn_unchanged", "parameters_unchanged", "rng_restored")})
    report.update(layer_count=len(layers), model_forward_passes=len(layers),
                  total_forwarded_samples=sum(row["calibration_samples"] for row in collection_reports),
                  gram_storage_bytes=max(row["gram_storage_bytes"] for row in collection_reports),
                  collection_seconds=sum(row["collection_seconds"] for row in collection_reports))
    target, center, coordinates = target_from_inverse_diagonals(specs)
    new_geometry = historical_geometry_update(geometry, target, float(config.get("eta", .1)))
    differences = [new_geometry[k] - geometry[k] for k in new_geometry]
    rms_drift = float((sum(value.square().sum() for value in differences) / coordinates).sqrt())
    max_drift = max(float(value.abs().max()) for value in differences)
    center_sum = float(sum(value.sum() for value in new_geometry.values()))
    absolute_sum = float(sum(value.abs().sum() for value in new_geometry.values()))
    if abs(center_sum) > 1e-12 * max(1., absolute_sum):
        raise FloatingPointError("New geometry violates zero-sum invariant")
    for name, layer in layers.items():
        key = name + ".weight"
        features = layer.weight[0].numel()
        old_features = geometry[key].reshape(layer.groups, -1, features)[:, 0]
        report["layers"][key]["target_previous_geometry_spearman"] = spearman(q_diags[key], old_features)
        report["layers"][key]["geometry_min"] = float(new_geometry[key].min())
        report["layers"][key]["geometry_max"] = float(new_geometry[key].max())
    report.update(status="HQ_CALIBRATION_REFRESH_PASS", eta=float(config.get("eta", .1)),
                  log_target_global_center=float(center), geometry_center_sum=center_sum,
                  log_geometry_drift={"rms": rms_drift, "max_abs": max_drift},
                  current_geometry_hash=tensor_digest(new_geometry), q_diagonal_hash=tensor_digest(q_diags),
                  q_diagonal_statistics={key: {"min": float(q.min()), "max": float(q.max()),
                    "mean": float(q.double().mean()), "std": float(q.double().std(unbiased=False))}
                    for key, q in q_diags.items()},
                  spearman_stability={key: row["q_previous_spearman"] for key, row in report["layers"].items()},
                  skipped_calibration_epochs=[], **_memory(device))
    synchronize(device)
    report["calibration_duration"] = time.perf_counter() - start
    return new_geometry, q_diags, report
