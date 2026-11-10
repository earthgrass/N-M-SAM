"""Real CIFAR-100 one-layer GPU timing and numerical diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_geo_obs_seed0 as common
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.geo_dynamic_obs import dynamic_group_obs_prune
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent-kind", choices=tuple(common.PARENTS), required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    expected_sha, protocol, method = common.PARENTS[args.parent_kind]
    if common.digest(args.parent) != expected_sha:
        raise ValueError("Parent hash mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    if (parent["protocol_id"] != protocol or parent["epoch"] != 200 or parent["steps"] != 78200
            or parent["identity"]["method"] != method or parent["test_accessed"] is not False):
        raise ValueError("Parent contract mismatch")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != protocol:
        raise ValueError("Config mismatch")
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169001437 or common.digest(archive) != common.ARCHIVE_SHA256
            or common.digest(archive, "md5") != common.ARCHIVE_MD5):
        raise ValueError("Official archive mismatch")
    data, labels = common.read_cifar(args.dataset_root / "extracted/cifar-100-python/train", 50000)
    ids = np.random.default_rng(common.CALIBRATION_SEED).permutation(50000)[:1024].tolist()
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    loader = DataLoader(common.CifarSubset(data, labels, ids, transform), batch_size=128,
                        shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    layer_name = "layer2.0.conv1"
    layer = eligible_layers(model)[layer_name]
    geometry = parent["stepper"]["geometry"]
    mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    if mapping != [(name, tuple(p.shape)) for name, p in model.named_parameters()]:
        raise ValueError("Geometry mismatch")
    logs = {name: value for (name, _), value in zip(mapping, geometry["logs"])}
    s = weight_to_matrix(logs[layer_name + ".weight"].to(device))
    w = weight_to_matrix(layer.weight.detach())
    adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
    hook = layer.register_forward_pre_hook(lambda _m, values: adapter.add_batch(values[0]))
    seen = []
    with torch.inference_mode():
        for inputs, _targets, sample_ids in loader:
            model(inputs.to(device, non_blocking=True))
            seen.extend(map(int, sample_ids.tolist()))
    hook.remove()
    if seen != ids:
        raise ValueError("Calibration order mismatch")
    h = adapter.hessians()[0]
    methods = ("std_group_obs", "geo_select_std_comp", "geo_full")
    results = []
    torch.cuda.reset_peak_memory_stats(device)
    for mode in methods:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = dynamic_group_obs_prune(w, h, log_scale=s, mode=mode, blocksize=128)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        delta = result.weight - w
        a_inv = torch.exp(-s.float())
        physical = common.quadratic_cost(delta, h)
        baseline_energy = common.quadratic_cost(w, h)
        geometry_cost = common.quadratic_cost(delta * a_inv, h)
        results.append({"mode": mode, "wall_seconds": elapsed,
                        "decomposition_seconds": result.decomposition_seconds,
                        "selection_compensation_seconds": result.selection_compensation_seconds,
                        "physical_quadratic_cost": physical,
                        "physical_relative_reconstruction_error": physical / baseline_energy,
                        "geometry_quadratic_cost": geometry_cost,
                        "deleted": int((~result.keep_mask).sum()),
                        "survivor_change_rms": float(delta[result.keep_mask].float().square().mean().sqrt()),
                        "finite": bool(torch.isfinite(result.weight).all()),
                        "strict_2of4": True})
    status = "PASS" if all(r["wall_seconds"] < .5 for r in results) else "BLOCKED_TIMING"
    common.atomic_json(args.output, {"status": status, "parent_kind": args.parent_kind,
                                     "layer": layer_name, "K": w.shape[1], "calibration_ids_sha256": ids_sha,
                                     "parent_sha256": expected_sha, "source_sha256": common.digest(ROOT / "baseline_adapters/geo_dynamic_obs/adapter.py"),
                                     "test_accessed": False, "results": results,
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    print(json.dumps({"status": status, "results": results}), flush=True)


if __name__ == "__main__":
    main()
