"""Single-layer fixed-mask quadratic compensation pilot for APBR geometry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

import run_apbr_weights_original_sparsegpt_one as common
import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers

PROTOCOL = "cifar100_50k10k_final_v1"
EXPERIMENT = "apbr_sparsegpt_compensation_attribution_geometry_20261031"
LAYER = "layer2.0.conv2"
ALPHAS = (0.0, 1e-3, 1e-2, 1e-1)
PERCDAMP = 0.01
CALIBRATION_SEED = 20261027
DIAGNOSTIC_SEED = 20261031
SAMPLES = 1024
SHUFFLE_SEED = 20261031


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometry_logs(parent: dict, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    geometry = parent["stepper"]["geometry"]
    mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    expected = [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    if geometry.get("method") != "vpr" or mapping != expected or len(geometry["logs"]) != len(mapping):
        raise ValueError("APBR geometry identity/mapping mismatch")
    result = {}
    for (name, shape), value in zip(mapping, geometry["logs"]):
        if value.dtype != torch.float64 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid geometry coordinate tensor: {name}")
        result[name] = value
    return result


def local_energy(delta: torch.Tensor, hessian: torch.Tensor) -> float:
    # H=2XX^T/n, so 0.5*delta H delta is mean squared local output error.
    return float(0.5 * torch.einsum("ri,ij,rj->", delta, hessian, delta) / delta.shape[0])


@torch.no_grad()
def solve_variant(
    weight: torch.Tensor,
    keep: torch.Tensor,
    hessian: torch.Tensor,
    log_radius: torch.Tensor,
    *,
    alpha: float,
    penalty: str,
) -> tuple[torch.Tensor, dict]:
    if penalty not in {"standard", "isotropic", "geometry", "geometry_shuffled"}:
        raise ValueError("Unknown penalty")
    device = weight.device
    damp = PERCDAMP * torch.diag(hessian).mean()
    hd = hessian.clone()
    hd.diagonal().add_(damp)
    output = weight.clone()
    output[~keep] = 0
    max_residual = 0.0
    sum_residual = 0.0
    max_lambda = 0.0
    geometry_cost = 0.0
    generator = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for row in range(weight.shape[0]):
        k = torch.where(keep[row])[0]
        d = torch.where(~keep[row])[0]
        if not len(k) or not len(d):
            raise ValueError("Pilot requires nonempty retained/deleted sets per row")
        hkk = hd.index_select(0, k).index_select(1, k)
        hkd = hd.index_select(0, k).index_select(1, d)
        rhs = hkd @ weight[row, d]
        scale = torch.diag(hkk).mean()
        lambda_abs = float(alpha) * scale
        if penalty == "standard":
            if alpha != 0:
                raise ValueError("Standard solver is defined only at alpha=0")
            penalty_diag = torch.zeros_like(k, dtype=weight.dtype)
        elif penalty == "isotropic":
            penalty_diag = torch.ones_like(k, dtype=weight.dtype)
        else:
            centered = log_radius[row, k] - log_radius[row, k].mean()
            penalty_diag = torch.exp(-2 * centered)
            if penalty == "geometry_shuffled":
                permutation = torch.randperm(len(k), generator=generator).to(device)
                penalty_diag = penalty_diag[permutation]
        system = hkk.clone()
        if alpha:
            system.diagonal().add_(lambda_abs * penalty_diag)
        chol, info = torch.linalg.cholesky_ex(system)
        if int(info) != 0:
            raise RuntimeError(f"Cholesky failed at output row {row}, penalty={penalty}, alpha={alpha}")
        delta_k = torch.cholesky_solve(rhs[:, None], chol).squeeze(1)
        residual = torch.linalg.vector_norm(system @ delta_k - rhs) / (
            torch.linalg.vector_norm(rhs) + torch.finfo(weight.dtype).eps
        )
        if not bool(torch.isfinite(delta_k).all() and torch.isfinite(residual)):
            raise FloatingPointError("Non-finite row solution")
        output[row, k] = weight[row, k] + delta_k
        output[row, d] = 0
        max_residual = max(max_residual, float(residual))
        sum_residual += float(residual)
        max_lambda = max(max_lambda, float(lambda_abs))
        geometry_cost += float((delta_k.square() * torch.exp(-2 * (
            log_radius[row, k] - log_radius[row, k].mean()
        ))).sum())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if torch.count_nonzero(output[~keep]) or not bool(torch.isfinite(output).all()):
        raise FloatingPointError("Fixed-mask solution violates zero/finite invariant")
    return output, {
        "runtime_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "max_relative_residual": max_residual,
        "mean_relative_residual": sum_residual / weight.shape[0],
        "max_lambda_absolute": max_lambda,
        "geometry_cost": geometry_cost / weight.shape[0],
        "damping": float(damp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL:
        raise ValueError("Protocol mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    if (parent.get("protocol_id") != PROTOCOL or parent.get("epoch") != 200
            or parent.get("steps") != 78200 or parent.get("test_accessed") is not False
            or parent.get("identity", {}).get("method") != "apbr_sam"
            or parent.get("identity", {}).get("seed") != 0):
        raise ValueError("Pilot is preregistered for APBR Seed 0 Dense Final")
    masks = torch.load(args.mask, map_location="cpu", weights_only=False)

    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    layer = eligible_layers(model)[LAYER]
    logs = geometry_logs(parent, model)
    log_radius = weight_to_matrix(logs[LAYER + ".weight"].to(device=device, dtype=torch.float32))
    weight = weight_to_matrix(layer.weight).detach().float().clone()
    keep = weight_to_matrix(masks[LAYER + ".weight"].to(device)).bool()
    if weight.shape != log_radius.shape or weight.shape != keep.shape:
        raise ValueError("Pilot weight/mask/geometry layout mismatch")

    train_file = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    calibration_ids = np.random.default_rng(CALIBRATION_SEED).permutation(50_000)[:SAMPLES].tolist()
    diagnostic_candidates = np.random.default_rng(DIAGNOSTIC_SEED).permutation(50_000).tolist()
    calibration_set = set(calibration_ids)
    diagnostic_ids = [int(i) for i in diagnostic_candidates if int(i) not in calibration_set][:SAMPLES]
    if set(calibration_ids) & set(diagnostic_ids):
        raise ValueError("Calibration/diagnostic Train split overlap")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]
    )])

    def collect(ids: list[int]) -> tuple[torch.Tensor, int]:
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        hook = layer.register_forward_pre_hook(lambda _m, inputs: adapter.add_batch(inputs[0]))
        loader = DataLoader(common.CalibrationSubset(data, labels, ids, transform), batch_size=128,
                            shuffle=False, drop_last=False, num_workers=2, pin_memory=True)
        seen = []
        with torch.inference_mode():
            for inputs, _targets, sample_ids in loader:
                model(inputs.to(device, non_blocking=True))
                seen.extend(map(int, sample_ids.tolist()))
        hook.remove()
        if seen != ids:
            raise ValueError("Train diagnostic order mismatch")
        return adapter.hessians()[0].detach().clone(), adapter.vectors

    h_cal, vectors_cal = collect(calibration_ids)
    h_diag, vectors_diag = collect(diagnostic_ids)
    rows = []
    solutions = {}
    mask_only = weight.clone()
    mask_only[~keep] = 0
    delta = mask_only - weight
    rows.append({
        "method": "no_compensation", "alpha": 0.0,
        "calibration_reconstruction_mse": local_energy(delta, h_cal),
        "diagnostic_reconstruction_mse": local_energy(delta, h_diag),
        "geometry_cost": 0.0, "runtime_seconds": 0.0,
        "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
        "max_relative_residual": 0.0, "mean_relative_residual": 0.0,
        "damping": PERCDAMP * float(torch.diag(h_cal).mean()),
    })

    standard, stats = solve_variant(weight, keep, h_cal, log_radius, alpha=0.0, penalty="standard")
    solutions["standard"] = standard.cpu()
    delta = standard - weight
    rows.append({
        "method": "standard_fixed_mask", "alpha": 0.0,
        "calibration_reconstruction_mse": local_energy(delta, h_cal),
        "diagnostic_reconstruction_mse": local_energy(delta, h_diag), **stats,
    })
    for alpha in ALPHAS:
        for method in ("isotropic", "geometry", "geometry_shuffled"):
            solved, stats = solve_variant(weight, keep, h_cal, log_radius, alpha=alpha, penalty=method)
            if alpha == 0.0:
                if not torch.equal(solved, standard):
                    raise ValueError(f"lambda=0 did not exactly degenerate for {method}")
            solutions[f"{method}_alpha_{alpha:g}"] = solved.cpu()
            delta = solved - weight
            rows.append({
                "method": method, "alpha": alpha,
                "calibration_reconstruction_mse": local_energy(delta, h_cal),
                "diagnostic_reconstruction_mse": local_energy(delta, h_diag), **stats,
            })
    if max(float(row["max_relative_residual"]) for row in rows) > 5e-4:
        raise FloatingPointError("Solver residual gate failed")
    write_csv(output_dir / "layer_pilot_results.csv", rows)
    write_csv(output_dir / "geometry_regularization_diagnostics.csv", rows)
    write_csv(output_dir / "runtime_memory_profile.csv", [{
        "method": row["method"], "alpha": row["alpha"],
        "runtime_seconds": row["runtime_seconds"],
        "peak_allocated_bytes": row["peak_allocated_bytes"],
        "peak_reserved_bytes": row["peak_reserved_bytes"],
    } for row in rows])
    solution_sha = common.atomic_torch(output_dir / "pilot_solutions.pt", {
        "layer": LAYER, "dense_weight": weight.cpu(), "keep_mask": keep.cpu(),
        "log_radius": log_radius.cpu(), "solutions": solutions,
    })
    status = {
        "status": "completed", "protocol_id": PROTOCOL, "experiment_id": EXPERIMENT,
        "seed": 0, "layer": LAYER, "weight_shape": list(layer.weight.shape),
        "matrix_shape": list(weight.shape), "calibration_samples": SAMPLES,
        "diagnostic_samples": SAMPLES, "calibration_vectors": vectors_cal,
        "diagnostic_vectors": vectors_diag,
        "calibration_ids_sha256": hashlib.sha256(np.asarray(calibration_ids, dtype="<i8").tobytes()).hexdigest(),
        "diagnostic_ids_sha256": hashlib.sha256(np.asarray(diagnostic_ids, dtype="<i8").tobytes()).hexdigest(),
        "alphas": list(ALPHAS), "lambda_scale": "alpha*mean(diag(H_KK_damped)) per row",
        "geometry_normalization": "s_K-centered per output row; R_tilde=exp(s_K-mean(s_K))",
        "damping": PERCDAMP, "test_accessed": False, "full_model_run": False,
        "solution_sha256": solution_sha,
    }
    common.atomic_json(output_dir / "status.json", status)
    print(json.dumps(status, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            common.atomic_json(target / "failure.json", {
                "status": "failed", "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False,
            })
        raise
