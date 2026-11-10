"""Run APBR-radius-guided SparseGPT on one frozen APBR dense parent.

The only algorithmic difference from the frozen Original SparseGPT endpoint is
the dynamic within-group ranking metric ``C_gpt * exp(-s)``.  The APBR
log-radius is fixed; Hessian construction, damping, factorization, sequential
OBS compensation, grouping, exclusions, and dense-tail semantics are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

import run_apbr_weights_original_sparsegpt_one as original_run
import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.radius_guided_sparsegpt import radius_guided_sparsegpt_matrix
from baseline_adapters.sparsegpt import SparseGPTAdapter, sparsegpt_matrix
from benchmark.nm import eligible_layers, validate_masks

PROTOCOL = "cifar100_50k10k_final_v1"
EXPERIMENT = "apbr_radius_guided_sparsegpt_seed012"
FORMAL_LAMBDA = 1.0


def _geometry_logs(parent: dict, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    try:
        geometry = parent["stepper"]["geometry"]
    except KeyError as exc:
        raise ValueError("APBR parent has no geometry state") from exc
    if geometry.get("method") != "vpr" or geometry.get("version") != 1:
        raise ValueError("Unexpected APBR geometry method/version")
    expected = [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    saved = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    if saved != expected or len(geometry["logs"]) != len(expected):
        raise ValueError("Geometry mapping/order does not exactly match model parameters")
    logs = {}
    for (name, shape), value in zip(saved, geometry["logs"]):
        if value.dtype != torch.float64 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid APBR log-radius tensor: {name}")
        logs[name] = value.detach().clone()
    return logs


def _block_spearman(left: torch.Tensor, right: torch.Tensor, m: int = 4) -> float:
    complete = left.shape[1] // m * m
    a = left[:, :complete].reshape(-1, m)
    b = right[:, :complete].reshape(-1, m)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise FloatingPointError("Non-finite score in rank correlation")
    order_a = a.argsort(dim=1, stable=True)
    order_b = b.argsort(dim=1, stable=True)
    rank_a = torch.empty_like(order_a)
    rank_b = torch.empty_like(order_b)
    ranks = torch.arange(m, device=a.device).expand_as(order_a)
    rank_a.scatter_(1, order_a, ranks)
    rank_b.scatter_(1, order_b, ranks)
    x = rank_a.float().flatten()
    y = rank_b.float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()))


def _quadratic_error(delta: torch.Tensor, hessian: torch.Tensor) -> float:
    # Per-output-row local reconstruction energy under the frozen layer Hessian.
    return float(torch.einsum("ri,ij,rj->", delta.float(), hessian.float(), delta.float()) / delta.shape[0])


def _load_original_control(run_dir: Path, parent_sha: str, calibration_sha: str) -> dict[str, torch.Tensor]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "sparsegpt_final.pt", map_location="cpu", weights_only=False)
    masks = torch.load(run_dir / "mask.pt", map_location="cpu", weights_only=False)
    if (status.get("status") != "completed" or status.get("parent_sha256") != parent_sha
            or checkpoint.get("parent", {}).get("sha256") != parent_sha
            or checkpoint.get("calibration_ids_sha256") != calibration_sha
            or checkpoint.get("algorithm", {}).get("apbr_geometry_used") is not False
            or checkpoint.get("algorithm", {}).get("dynamic_mask") is not True
            or checkpoint.get("algorithm", {}).get("percdamp") != original_run.PERCDAMP
            or checkpoint.get("algorithm", {}).get("blocksize") != original_run.BLOCKSIZE):
        raise ValueError("Original SparseGPT control is not configuration/parent identical")
    if train.tensor_sha(masks) != checkpoint.get("mask_sha256"):
        raise ValueError("Original SparseGPT control mask hash mismatch")
    return masks


@torch.no_grad()
def _compare_layer(
    name: str,
    adapter: SparseGPTAdapter,
    log_radius_weight: torch.Tensor,
    saved_original_mask: torch.Tensor,
    *,
    commit_hybrid: bool,
) -> tuple[torch.Tensor, dict, dict, dict]:
    layer = adapter.layer
    matrix = weight_to_matrix(layer.weight)
    before = matrix.detach().clone()
    log_radius = weight_to_matrix(log_radius_weight.to(device=matrix.device))
    if log_radius.shape != matrix.shape:
        raise ValueError(f"Weight/radius layout mismatch: {name}")
    if not bool(torch.isfinite(log_radius).all()):
        raise FloatingPointError(f"Non-finite log-radius: {name}")
    radius = torch.exp(log_radius)
    if not bool(torch.isfinite(radius).all() and (radius > 0).all()):
        raise FloatingPointError(f"Invalid positive radius: {name}")

    rows_per_group = matrix.shape[0] // adapter.groups
    hybrid_keep = torch.empty_like(matrix, dtype=torch.bool)
    original_keep = torch.empty_like(matrix, dtype=torch.bool)
    layer_rows = []
    score_parts_original, score_parts_hybrid = [], []
    for group, hessian in enumerate(adapter.hessians()):
        rows = slice(group * rows_per_group, (group + 1) * rows_per_group)
        original = sparsegpt_matrix(
            before[rows], hessian, keep_n=2, m=4, blocksize=original_run.BLOCKSIZE,
            percdamp=original_run.PERCDAMP, compute_dtype=torch.float32,
        )
        hybrid = radius_guided_sparsegpt_matrix(
            before[rows], hessian, log_radius[rows], keep_n=2, m=4,
            blocksize=original_run.BLOCKSIZE, percdamp=original_run.PERCDAMP,
            compute_dtype=torch.float32, radius_lambda=FORMAL_LAMBDA,
        )
        original_keep[rows] = original.keep_mask
        hybrid_keep[rows] = hybrid.keep_mask
        valid = torch.isfinite(hybrid.original_metric)
        score_parts_original.append(hybrid.original_metric[valid].detach())
        score_parts_hybrid.append(hybrid.hybrid_metric[valid].detach())
        original_delta = original.weight - before[rows]
        hybrid_delta = hybrid.weight - before[rows]
        layer_rows.append({
            "layer": name,
            "group": group,
            "hessian_dimension": int(hessian.shape[0]),
            "activation_vectors": int(adapter.vectors),
            "damping_original": original.damp,
            "damping_hybrid": hybrid.damp,
            "hessian_identical": bool(torch.equal(original.hessian, hybrid.hessian)),
            "original_reconstruction_energy": _quadratic_error(original_delta, hessian),
            "hybrid_reconstruction_energy": _quadratic_error(hybrid_delta, hessian),
            "original_compensation_rms": float(original_delta[original.keep_mask].float().square().mean().sqrt()),
            "hybrid_compensation_rms": float(hybrid_delta[hybrid.keep_mask].float().square().mean().sqrt()),
            "decomposition_seconds_hybrid": hybrid.decomposition_seconds,
            "compensation_seconds_hybrid": hybrid.compensation_seconds,
            "deleted_weights_exact_zero": bool((hybrid.weight[~hybrid.keep_mask] == 0).all()),
            "finite": bool(torch.isfinite(hybrid.weight).all()),
        })
        if commit_hybrid:
            matrix[rows].copy_(hybrid.weight)

    expected_original = weight_to_matrix(saved_original_mask.to(device=matrix.device))
    if not torch.equal(original_keep, expected_original):
        raise ValueError(f"Recomputed Original SparseGPT mask differs from frozen control: {name}")
    complete = matrix.shape[1] // 4 * 4
    overlap = original_keep[:, :complete] == hybrid_keep[:, :complete]
    removed_intersection = ((~original_keep[:, :complete]) & (~hybrid_keep[:, :complete])).sum()
    removed_union = ((~original_keep[:, :complete]) | (~hybrid_keep[:, :complete])).sum()
    original_scores = torch.cat(score_parts_original)
    hybrid_scores = torch.cat(score_parts_hybrid)
    score_row = {
        "layer": name,
        "gpt_hybrid_score_spearman_within_2x4": _block_spearman(
            torch.where(torch.isfinite(hybrid.original_metric), hybrid.original_metric, torch.zeros_like(hybrid.original_metric)),
            torch.where(torch.isfinite(hybrid.hybrid_metric), hybrid.hybrid_metric, torch.zeros_like(hybrid.hybrid_metric)),
        ),
        "original_score_min": float(original_scores.min()),
        "original_score_mean": float(original_scores.mean()),
        "original_score_max": float(original_scores.max()),
        "hybrid_score_min": float(hybrid_scores.min()),
        "hybrid_score_mean": float(hybrid_scores.mean()),
        "hybrid_score_max": float(hybrid_scores.max()),
    }
    overlap_row = {
        "layer": name,
        "positions": int(overlap.numel()),
        "same_keep_delete_positions": int(overlap.sum()),
        "position_overlap_rate": float(overlap.float().mean()),
        "mask_flip_rate": float((~overlap).float().mean()),
        "removed_jaccard": float(removed_intersection / removed_union),
    }
    radius_row = {
        "layer": name,
        "coordinates": int(radius.numel()),
        "log_radius_mean": float(log_radius.mean()),
        "log_radius_std": float(log_radius.std(unbiased=False)),
        "log_radius_min": float(log_radius.min()),
        "log_radius_max": float(log_radius.max()),
        "radius_mean": float(radius.mean()),
        "radius_std": float(radius.std(unbiased=False)),
        "radius_min": float(radius.min()),
        "radius_max": float(radius.max()),
    }
    return hybrid_keep.reshape_as(layer.weight), radius_row, score_row, overlap_row, layer_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--original-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("diagnostic", "smoke", "formal"), required=True)
    parser.add_argument("--smoke-layer", default="layer4.0.conv2")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback forbidden")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    formal = config["formal_calibration"]
    expected = {"samples": 1024, "selection_seed": 20261027, "batch_size": 128, "augmentation": "none"}
    if config["protocol_id"] != PROTOCOL or any(formal[k] != v for k, v in expected.items()):
        raise ValueError("Frozen protocol/calibration mismatch")

    parent_sha = original_run.sha_file(args.parent)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    identity = parent.get("identity", {})
    if (parent.get("protocol_id") != PROTOCOL or parent.get("epoch") != 200
            or parent.get("steps") != 78200 or parent.get("test_accessed") is not False
            or identity.get("method") != "apbr_sam" or identity.get("seed") != args.seed):
        raise ValueError("APBR dense parent identity mismatch")

    train_file = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    import pickle
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    calibration_ids = np.random.default_rng(20261027).permutation(50_000)[:1024].tolist()
    ids_sha = hashlib.sha256(np.asarray(calibration_ids, dtype="<i8").tobytes()).hexdigest()
    original_masks = _load_original_control(args.original_run_dir, parent_sha, ids_sha)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]
    )])
    batches = DataLoader(
        original_run.CalibrationSubset(data, labels, calibration_ids, transform), batch_size=128,
        shuffle=False, drop_last=False, num_workers=2, pin_memory=True,
    )
    device = torch.device("cuda:0")
    train.seed_all(args.seed)
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    geometry_logs = _geometry_logs(parent, model)
    all_layers = eligible_layers(model)
    selected_layers = ({args.smoke_layer: all_layers[args.smoke_layer]}
                       if args.mode == "smoke" else all_layers)
    adapters, hooks = {}, []
    for name, layer in selected_layers.items():
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        adapters[name] = adapter
        hooks.append(layer.register_forward_pre_hook(lambda _m, inputs, item=adapter: item.add_batch(inputs[0])))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    seen = []
    with torch.inference_mode():
        for inputs, _targets, sample_ids in batches:
            model(inputs.to(device, non_blocking=True))
            seen.extend(map(int, sample_ids.tolist()))
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()
    if seen != calibration_ids:
        raise ValueError("Calibration order mismatch")

    masks, radius_rows, score_rows, overlap_rows, layer_rows = {}, [], [], [], []
    for name in selected_layers:
        key = name + ".weight"
        result = _compare_layer(
            name, adapters.pop(name), geometry_logs[key].to(device), original_masks[key],
            commit_hybrid=args.mode == "formal",
        )
        mask, radius_row, score_row, overlap_row, diagnostics = result
        masks[key] = mask.detach().cpu()
        radius_rows.append(radius_row)
        score_rows.append(score_row)
        overlap_rows.append(overlap_row)
        layer_rows.extend(diagnostics)

    original_run.write_csv(output / "RADIUS_DISTRIBUTION.csv", radius_rows)
    original_run.write_csv(output / "SCORE_CORRELATION.csv", score_rows)
    original_run.write_csv(output / "MASK_OVERLAP.csv", overlap_rows)
    original_run.write_csv(output / "layer_diagnostics.csv", layer_rows)
    status = {
        "status": "completed", "protocol_id": PROTOCOL, "experiment_id": EXPERIMENT,
        "mode": args.mode, "seed": args.seed, "parent_sha256": parent_sha,
        "calibration_ids_sha256": ids_sha, "layers": len(selected_layers),
        "radius_definition": "r=exp(s)", "hybrid_score": "C_gpt*exp(-s)",
        "radius_lambda": FORMAL_LAMBDA, "test_accessed": False,
        "recovery_epochs": 0, "automatic_retry": False,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if args.mode == "formal":
        validate_masks(model, masks, n=2, m=4, complete=True)
        eligible_positions = sum(x.numel() for x in masks.values())
        removed_positions = sum(int((~x).sum()) for x in masks.values())
        if eligible_positions != 11_157_504 or removed_positions != 5_578_752:
            raise ValueError("Formal 2:4 count mismatch")
        payload = {
            "schema": 1, "status": "healthy", "protocol_id": PROTOCOL,
            "experiment_id": EXPERIMENT, "stage": "raw_hard",
            "method": "apbr_radius_guided_sparsegpt", "seed": args.seed,
            "parent_method": "apbr_sam",
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "masks": masks, "mask_sha256": train.tensor_sha(masks),
            "parent": {"path": str(args.parent), "sha256": parent_sha},
            "calibration_ids": calibration_ids, "calibration_ids_sha256": ids_sha,
            "algorithm": {
                "dynamic_mask": True, "external_mask": False, "apbr_geometry_used": True,
                "radius_definition": "r=exp(s)", "score": "C_gpt/r", "radius_lambda": 1.0,
                "keep_n": 2, "m": 4, "blocksize": 128, "percdamp": 0.01,
                "compute_dtype": "float32", "obs_compensation": "original_unchanged",
                "recovery_epochs": 0, "bn_calibration": False,
            },
            "test_accessed": False,
        }
        status["checkpoint_sha256"] = original_run.atomic_torch(output / "sparsegpt_final.pt", payload)
        status["mask_file_sha256"] = original_run.atomic_torch(output / "mask.pt", masks)
        status["mask_tensor_sha256"] = payload["mask_sha256"]
        status["eligible_positions"] = eligible_positions
        status["removed_positions"] = removed_positions
    original_run.atomic_json(output / "status.json", status)
    print(json.dumps(status, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            original_run.atomic_json(target / "failure.json", {
                "status": "failed", "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False,
            })
        raise
