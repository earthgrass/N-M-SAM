"""Fixed-final HQ endpoints: static Final-Q, fixed-mask OBS, and native GPT.

The Test set is opened only after all four final model files have been frozen.
Every pruning branch starts at the original dense parent. No recovery path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
from torch import nn

from .data import (calibration_ids, calibration_loader, digest_file, evaluation_loader,
                   load_test, load_train)
from .math_core import final_obs_keep_mask, inverse_diagonal_cholesky
from .vendor.fixedmask_obs import fixedmask_obs_matrix
from .vendor.model import build_model, eligible_layers
from .vendor.nm import validate_keep_mask
from .vendor.sparsegpt import sparsegpt_matrix


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    restored = torch.load(temporary, map_location="cpu", weights_only=False)
    if tensor_hash(restored["model"]) != value["model_sha256"]:
        raise ValueError("Checkpoint save verification failed")
    temporary.replace(path)
    return digest_file(path)


def reconstruction_mse(before, after, hessian):
    """Physical output MSE, averaged over patches and output channels.

    H is XX^T/n, without damping. The product uses FP32 and its reduction FP64.
    This differs intentionally from the damped OBS optimization objective.
    """
    delta = after.float() - before.float()
    value = ((delta @ hessian.float()) * delta).double().sum() / before.shape[0]
    if not bool(torch.isfinite(value)) or float(value) < -1e-5:
        raise FloatingPointError("Invalid reconstruction MSE")
    return max(0., float(value))


def prepare_layer_endpoints(weight, hessian, *, percdamp=.01, blocksize=128):
    """Compute all three pruning outputs from one unmodified dense row matrix."""
    if weight.ndim != 2 or hessian.ndim != 2:
        raise ValueError("Expected one connection group's row matrix and Gram")
    synchronize(weight.device)
    q_started = time.perf_counter()
    q, _, delta = inverse_diagonal_cholesky(hessian, percdamp=percdamp, dead_policy="reject")
    synchronize(weight.device)
    q_seconds = time.perf_counter() - q_started
    mask_started = time.perf_counter()
    mask = final_obs_keep_mask(weight, q)
    raw = weight.detach() * mask
    synchronize(weight.device)
    mask_seconds = time.perf_counter() - mask_started
    # Canonical historical adapters use H_SG=2XX^T/n, including dead diag=1.
    # HQ static Final-Q uses H=XX^T/n without that dead-coordinate replacement.
    # Both conventions share actual activations; dead-feature counts are logged.
    fixed = fixedmask_obs_matrix(weight, 2 * hessian, mask, blocksize=blocksize,
                                 percdamp=percdamp, compute_dtype=torch.float32)
    native = sparsegpt_matrix(weight, 2 * hessian, blocksize=blocksize,
                              percdamp=percdamp, compute_dtype=torch.float32)
    if not torch.equal(mask, fixed.keep_mask):
        raise RuntimeError("Fixed-mask OBS changed Final-Q support")
    results = {"final_q_mask_only": (raw, mask),
               "final_q_fixed_obs": (fixed.weight, fixed.keep_mask),
               "sparsegpt_native": (native.weight, native.keep_mask)}
    for output, keep in results.values():
        validate_keep_mask(keep, 2, 4)
        if not bool(torch.isfinite(output).all()) or torch.count_nonzero(output[~keep]):
            raise FloatingPointError("Invalid final pruned weights")
    diagnostics = {
        "hq_absolute_damping": delta,
        "final_q_inverse_diagonal_seconds": q_seconds,
        "final_q_mask_projection_seconds": mask_seconds,
        "dead_input_features": int((hessian.diag() == 0).sum()),
        "hq_dead_policy": "no replacement; relative damping only",
        "obs_dead_policy": "historical H_SG=2H; replace dead diagonal by 1 and dead weights by 0",
        "final_q_inverse_diagonal_min": float(q.min()),
        "final_q_inverse_diagonal_max": float(q.max()),
        "layer_reconstruction_mse": {name: reconstruction_mse(weight, value[0], hessian)
                                     for name, value in results.items()},
        "fixed_obs": {"decomposition_seconds": fixed.decomposition_seconds,
                      "compensation_seconds": fixed.compensation_seconds,
                      "survivor_changed_count": fixed.survivor_changed_count},
        "sparsegpt_native": {"decomposition_seconds": native.decomposition_seconds,
                             "compensation_seconds": native.compensation_seconds},
        "static_native_mask_difference_count": int((mask != native.keep_mask).sum()),
    }
    return results, diagnostics


@torch.no_grad()
def prepare_endpoints(model, loader, config, output_dir, device):
    from .calibration import collect_layer_gram
    output_dir = Path(output_dir)
    dense = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    names = ("dense", "final_q_mask_only", "final_q_fixed_obs", "sparsegpt_native")
    states = {name: {key: value.clone() for key, value in dense.items()} for name in names}
    masks = {name: {} for name in names}
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11_157_504:
        raise ValueError("Final evaluation model scope mismatch")
    all_rows = []
    for name, layer in layers.items():
        key = name + ".weight"
        if layer.groups != 1:
            raise ValueError("Frozen CIFAR-100 evaluation expects groups=1")
        group_grams, calibration = collect_layer_gram(model, loader, name, config, device)
        if group_grams.shape != (1, layer.weight[0].numel(), layer.weight[0].numel()):
            raise ValueError("Final calibration mapping mismatch")
        h = group_grams[0]
        before = layer.weight.detach().reshape(layer.out_channels, -1)
        synchronize(before.device)
        started = time.perf_counter()
        results, row = prepare_layer_endpoints(before, h, percdamp=config.get("percdamp", .01),
                                               blocksize=config.get("blocksize", 128))
        synchronize(before.device)
        row.update(layer=name, shape=list(layer.weight.shape), total_seconds=time.perf_counter() - started,
                   calibration=calibration)
        for endpoint, (weight, keep) in results.items():
            states[endpoint][key] = weight.reshape_as(layer.weight).cpu().clone()
            masks[endpoint][key] = keep.reshape_as(layer.weight).cpu().clone()
        all_rows.append(row)
        with (output_dir / "pruning_layers.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        del group_grams, h, results
    for name in names[1:]:
        if sum(int((~keep).sum()) for keep in masks[name].values()) != 5_578_752:
            raise ValueError("Final deletion count mismatch")
        for key, before in dense.items():
            if key not in masks[name] and not torch.equal(before, states[name][key]):
                raise ValueError("Noneligible or BN state was modified")
    return states, masks, all_rows


@torch.no_grad()
def test_metrics(model, loader, device):
    model.eval()
    count = correct = 0
    loss = 0.
    synchronize(device)
    started = time.perf_counter()
    for x, y, _ids in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite final logits")
        loss += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    synchronize(device)
    if count != 10000:
        raise ValueError("Final Test sample count mismatch")
    return {"samples": count, "correct": correct, "accuracy": correct / count,
            "accuracy_percent": 100 * correct / count, "ce": loss / count,
            "evaluation_seconds": time.perf_counter() - started, "test_accessed": True}


def evaluate_final(parent_path, archive, config_path, output_dir, device="cuda:0"):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Refusing to repeat or overwrite final evaluation")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("calibration_samples") != 1024 or config.get("calibration_seed") != 20261027:
        raise ValueError("Final evaluation requires frozen historical calibration IDs")
    config_sha = digest_file(Path(config_path))
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("completed_epochs") != 200 or parent.get("epoch_index") != 199:
        raise ValueError("Final parent must be fixed Epoch index 199, completed 200")
    if parent.get("test_accessed") is not False or parent.get("seed", 0) != 0:
        raise ValueError("Parent isolation or seed mismatch")
    if parent.get("config_sha256", parent.get("identity", {}).get("config_sha256")) != config_sha:
        raise ValueError("Parent configuration hash mismatch")
    parent_sha = digest_file(Path(parent_path))
    model = build_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    data, labels, identity = load_train(archive)
    loader = calibration_loader(data, labels, batch_size=config.get("calibration_batch_size", 128),
                                workers=config.get("workers", 2))
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    states, masks, rows = prepare_endpoints(model, loader, config, output, device)
    atomic_json(output / "final_calibration.json", {
        "mode": "one_layer_collect_prune_release_then_next", "test_accessed": False,
        "calibration_ids": calibration_ids(),
        "layers": {row["layer"]: row["calibration"] for row in rows},
    })
    frozen = {"status": "ALL_ENDPOINTS_FROZEN", "test_accessed": False, "seed": 0,
              "parent_sha256": parent_sha, "config_sha256": config_sha,
              "source_identity": parent.get("source_identity", parent.get("identity")),
              "calibration_ids": calibration_ids(), "dataset": identity,
              "endpoints": {}, "recovery_epochs": 0}
    for endpoint, state in states.items():
        item = {"endpoint": endpoint, "model": state, "model_sha256": tensor_hash(state),
                "masks": masks[endpoint], "mask_sha256": tensor_hash(masks[endpoint]),
                "parent_sha256": parent_sha, "config_sha256": config_sha,
                "seed": 0, "completed_epochs": 200, "epoch_index": 199,
                "test_accessed": False, "recovery_epochs": 0}
        path = output / f"{endpoint}.pt"
        sha = atomic_checkpoint(path, item)
        frozen["endpoints"][endpoint] = {"checkpoint_path": str(path), "checkpoint_sha256": sha,
                                         "model_sha256": item["model_sha256"], "mask_sha256": item["mask_sha256"]}
    atomic_json(output / "all_endpoints_frozen.json", frozen)
    del states, masks, data, labels
    # Marker is written before the first Test read. A failure never retries.
    atomic_json(output / "test_started.json", {"test_accessed": True, "automatic_retry": False,
                                               "frozen_manifest_sha256": digest_file(output / "all_endpoints_frozen.json")})
    test_data, test_labels, test_identity = load_test(archive, frozen_manifest=output / "all_endpoints_frozen.json")
    test_loader = evaluation_loader(test_data, test_labels, workers=config.get("workers", 2))
    results = {}
    for endpoint, item in frozen["endpoints"].items():
        checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics = test_metrics(model, test_loader, device)
        metrics.update(endpoint=endpoint, seed=0, **item, parent_sha256=parent_sha, config_sha256=config_sha)
        atomic_json(output / f"{endpoint}_evaluation.json", metrics)
        results[endpoint] = metrics
    report = {"status": "FINAL_EVALUATION_PASS", "test_accessed": True, "results": results,
              "dataset": test_identity, "recovery_epochs": 0,
              "historical_comparison": "Reference only until dataset/model/calibration/damping/environment protocols are audited"}
    if torch.device(device).type == "cuda":
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    atomic_json(output / "evaluation_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    can_record_failure = not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir()))
    try:
        evaluate_final(args.parent, args.archive, args.config, args.output, args.device)
    except Exception:
        if can_record_failure and args.output.is_dir() and not (args.output / "failure.json").exists():
            atomic_json(args.output / "failure.json", {"status": "FINAL_EVALUATION_FAIL",
                       "automatic_retry": False, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
