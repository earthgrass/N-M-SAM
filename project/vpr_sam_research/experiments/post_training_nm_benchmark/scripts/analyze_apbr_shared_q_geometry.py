"""Read-only Shared-Q geometry diagnostics on the frozen calibration set.

This script never constructs a Test loader and never writes model weights.  It
quantifies how much coordinate information survives output-row aggregation and
how the resulting vector relates to the input Hessian diagonal.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
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
from baseline_adapters.apbr_gpt import normalized_inverse_square
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers


def _rank_average(value: torch.Tensor) -> torch.Tensor:
    """Average ranks with deterministic tie handling (CPU float64)."""
    x = value.detach().double().cpu().numpy()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return torch.from_numpy(ranks)


def _corr(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().double().cpu().flatten()
    y = right.detach().double().cpu().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y) / denom) if float(denom) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Hessian construction")

    cfg = yaml.safe_load(args.config.read_text())
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    if parent.get("epoch") != 200 or parent.get("identity", {}).get("method") != "apbr_sam":
        raise ValueError("Unexpected APBR parent")

    device = torch.device("cuda")
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"])
    model.eval()
    layers = eligible_layers(model)

    geometry = parent["stepper"]["geometry"]
    mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    expected = [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    if geometry["method"] != "vpr" or mapping != expected:
        raise ValueError("Geometry/parameter identity mismatch")
    logs = {name: value for (name, _), value in zip(mapping, geometry["logs"])}
    qmap, global_raw_q_mean = normalized_inverse_square(
        {name + ".weight": logs[name + ".weight"] for name in layers}
    )

    train_file = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    calibration_ids = np.random.default_rng(20261027).permutation(50000)[:1024].tolist()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["dataset"]["normalize_mean"], cfg["dataset"]["normalize_std"]),
    ])
    loader = DataLoader(
        common.CalibrationSubset(data, labels, calibration_ids, transform),
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    adapters: dict[str, SparseGPTAdapter] = {}
    hooks = []
    for name, layer in layers.items():
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        adapters[name] = adapter
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, values, item=adapter: item.add_batch(values[0])
        ))
    with torch.inference_mode():
        for inputs, _targets, _ids in loader:
            model(inputs.to(device, non_blocking=True))
    for hook in hooks:
        hook.remove()

    rows = []
    for name, layer in layers.items():
        hessian = adapters[name].hessians()[0]
        hdiag = torch.diag(hessian).detach().double().cpu()
        q_exact = weight_to_matrix(qmap[name + ".weight"].float()).detach().double().cpu()
        q_shared = q_exact.mean(dim=0)
        shared_expanded = q_shared.unsqueeze(0).expand_as(q_exact)
        aggregation_delta = q_exact - shared_expanded
        ones = torch.ones_like(q_shared)
        rows.append({
            "layer": name,
            "output_rows": q_exact.shape[0],
            "input_coordinates": q_exact.shape[1],
            "q_shared_mean": float(q_shared.mean()),
            "q_shared_std": float(q_shared.std(unbiased=False)),
            "q_shared_cv": float(q_shared.std(unbiased=False) / q_shared.mean()),
            "q_shared_min": float(q_shared.min()),
            "q_shared_max": float(q_shared.max()),
            "q_shared_max_over_min": float(q_shared.max() / q_shared.min()),
            "q_shared_isotropic_relative_l2": float((q_shared - ones).norm() / ones.norm()),
            "exact_to_shared_relative_rms": float(
                aggregation_delta.square().mean().sqrt() / q_exact.square().mean().sqrt()
            ),
            "exact_to_shared_max_abs": float(aggregation_delta.abs().max()),
            "q_hdiag_pearson": _corr(q_shared, hdiag),
            "q_hdiag_spearman": _corr(_rank_average(q_shared), _rank_average(hdiag)),
            "q_log_hdiag_log_pearson": _corr(q_shared.log(), hdiag.clamp_min(1e-30).log()),
            "hdiag_mean": float(hdiag.mean()),
            "hdiag_std": float(hdiag.std(unbiased=False)),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed",
        "layers": len(rows),
        "calibration_samples": len(calibration_ids),
        "calibration_ids_sha256": __import__("hashlib").sha256(
            np.asarray(calibration_ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "global_raw_q_mean": global_raw_q_mean,
        "test_accessed": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
