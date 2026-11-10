"""One-shot dynamic Group-OBS and APBR geometry ablations on one dense parent."""
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
import run_geo_obs_seed0 as common
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import matrix_to_weight, weight_to_matrix
from baseline_adapters.geo_dynamic_obs import dynamic_group_obs_prune
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers, validate_masks

METHODS = {
    "dynamic_group_obs": "std_group_obs",
    "geo_select_std_comp": "geo_select_std_comp",
    "geo_full": "geo_full",
    "shuffled_geo_select": "geo_select_std_comp",
    "shuffled_geo_full": "geo_full",
}
SHUFFLE_SEED = 20261102


def shuffle_within_quartets(s: torch.Tensor, seed: int) -> torch.Tensor:
    if s.ndim != 2 or s.shape[1] % 4:
        raise ValueError("Geometry shape must be [rows, multiple of 4]")
    rows, width = s.shape
    groups = width // 4
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.rand((rows, groups, 4), generator=generator, dtype=torch.float64)
    permutation = keys.argsort(dim=-1, stable=True).to(s.device)
    return s.reshape(rows, groups, 4).gather(-1, permutation).reshape_as(s)


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device required")
    expected_sha, protocol, identity_method = common.PARENTS[args.parent_kind]
    if common.digest(args.parent) != expected_sha:
        raise ValueError("Dense parent SHA256 mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    identity = parent.get("identity", {})
    if (parent.get("protocol_id") != protocol or parent.get("epoch") != 200
            or parent.get("steps") != 78200 or parent.get("test_accessed") is not False
            or identity.get("method") != identity_method or identity.get("seed") != 0):
        raise ValueError("Dense parent identity mismatch")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != protocol:
        raise ValueError("Config mismatch")
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169001437 or common.digest(archive) != common.ARCHIVE_SHA256
            or common.digest(archive, "md5") != common.ARCHIVE_MD5):
        raise ValueError("Official archive mismatch")
    train_path = args.dataset_root / "extracted/cifar-100-python/train"
    data, labels = common.read_cifar(train_path, 50000)
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
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise ValueError("Eligible layer contract mismatch")
    geometry = parent["stepper"]["geometry"]
    mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    if geometry.get("method") != args.parent_kind or mapping != [(name, tuple(p.shape)) for name, p in model.named_parameters()]:
        raise ValueError("Geometry identity mismatch")
    logs = {name: value for (name, _), value in zip(mapping, geometry["logs"])}
    for name, layer in layers.items():
        if (logs[name + ".weight"].dtype != torch.float64
                or logs[name + ".weight"].shape != layer.weight.shape):
            raise ValueError("Geometry tensor mismatch")
    source = {
        "runner": common.digest(Path(__file__)),
        "adapter": common.digest(ROOT / "baseline_adapters/geo_dynamic_obs/adapter.py"),
        "hessian": common.digest(ROOT / "baseline_adapters/sparsegpt/adapter.py"),
        "model": common.digest(ROOT / "baseline_adapters/cifar100_model.py"),
        "config": common.digest(args.config),
    }
    common.atomic_json(out / "preflight.json", {
        "status": "PASS", "parent_kind": args.parent_kind, "parent_sha256": expected_sha,
        "protocol_id": protocol, "archive_sha256": common.ARCHIVE_SHA256,
        "train_pickle_sha256": common.digest(train_path), "calibration_ids_sha256": ids_sha,
        "calibration_ids": ids, "eligible_layers": list(layers),
        "eligible_positions": 11157504, "expected_deletions": 5578752,
        "source_sha256": source, "gpu": torch.cuda.get_device_name(0),
        "test_accessed": False,
    })
    del parent
    adapters, hooks = {}, []
    for name, layer in layers.items():
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        adapters[name] = adapter
        hooks.append(layer.register_forward_pre_hook(lambda _m, values, item=adapter: item.add_batch(values[0])))
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    calibration_started = time.perf_counter()
    seen = []
    with torch.inference_mode():
        for inputs, _targets, sample_ids in loader:
            model(inputs.to(device, non_blocking=True))
            seen.extend(map(int, sample_ids.tolist()))
    torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - calibration_started
    for hook in hooks:
        hook.remove()
    if seen != ids:
        raise ValueError("Calibration order mismatch")
    hessians = {name: adapter.hessians()[0] for name, adapter in adapters.items()}
    del adapters
    states = {method: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
              for method in METHODS}
    masks = {method: {} for method in METHODS}
    rows = []
    peak_allocated = peak_reserved = 0
    for index, (name, layer) in enumerate(layers.items(), 1):
        key = name + ".weight"
        w = weight_to_matrix(layer.weight.detach()).float()
        s = weight_to_matrix(logs[key].to(device))
        shuffled = shuffle_within_quartets(s, SHUFFLE_SEED + index)
        h = hessians[name]
        torch.cuda.reset_peak_memory_stats(device)
        for method, mode in METHODS.items():
            current_s = shuffled if method.startswith("shuffled") else s
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = dynamic_group_obs_prune(w, h, log_scale=current_s, mode=mode, blocksize=128)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            keep = result.keep_mask
            final = result.weight
            if not bool(torch.isfinite(final).all()) or int(torch.count_nonzero(final[~keep])):
                raise FloatingPointError(f"Invalid output: {name}/{method}")
            if int((~keep).sum()) != w.numel() // 2:
                raise ValueError("2:4 deletion count mismatch")
            masks[method][key] = keep.reshape_as(layer.weight).detach().cpu()
            states[method][key] = matrix_to_weight(final.detach().cpu(), layer.weight.shape)
            delta = final - w
            physical = common.quadratic_cost(delta, h)
            geometry_cost = common.quadratic_cost(delta * torch.exp(-s.float()), h)
            row = {"layer_index": index, "layer": name, "method": method,
                   "rows": w.shape[0], "K": w.shape[1], "deleted": int((~keep).sum()),
                   "physical_quadratic_cost": physical,
                   "physical_relative_reconstruction_error": physical / common.quadratic_cost(w, h),
                   "geometry_quadratic_cost": geometry_cost,
                   "survivor_change_rms": float(delta[keep].square().mean().sqrt()),
                   "decomposition_seconds": result.decomposition_seconds,
                   "selection_compensation_seconds": result.selection_compensation_seconds,
                   "wall_seconds": seconds, "finite": True, "strict_2of4": True,
                   "mask_flip_count_vs_dynamic_group": None,
                   "mask_flip_rate_vs_dynamic_group": None}
            rows.append(row)
            common.atomic_torch(out / "layers" / f"{index:02d}_{name.replace('.', '_')}_{method}.pt",
                                {"layer": name, "method": method, "weight": states[method][key],
                                 "mask": masks[method][key], "chosen_pairs": result.chosen_pairs.detach().cpu(),
                                 "candidate_costs": result.candidate_costs.detach().cpu(), "diagnostic": row})
        base = masks["dynamic_group_obs"][key]
        for row in rows[-len(METHODS):]:
            other = masks[row["method"]][key]
            flips = int((base != other).sum())
            row["mask_flip_count_vs_dynamic_group"] = flips
            row["mask_flip_rate_vs_dynamic_group"] = flips / base.numel()
        common.atomic_json(out / "progress.json", {"status": "healthy_partial",
                                                   "completed_layers": index, "last_layer": name,
                                                   "test_accessed": False})
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        print(json.dumps({"event": "layer_complete", "index": index, "layer": name,
                          "peak_allocated_bytes": peak_allocated, "test_accessed": False}), flush=True)
        torch.cuda.empty_cache()
    for method in METHODS:
        check = build_cifar100_model()
        check.load_state_dict(states[method], strict=True)
        validate_masks(check, masks[method], n=2, m=4, complete=True)
        if sum(int((~mask).sum()) for mask in masks[method].values()) != 5578752:
            raise ValueError("Full model deletion count mismatch")
    common.atomic_csv(out / "per_layer.csv", rows)
    frozen = {}
    for method in METHODS:
        method_dir = out / "methods" / method
        mask_sha = common.atomic_torch(method_dir / "masks.pt", masks[method])
        payload = {"schema": 1, "status": "healthy", "protocol_id": protocol,
                   "experiment_id": "geo_dynamic_seed0_20261102_v1", "stage": "post_obs_raw_hard",
                   "parent_kind": args.parent_kind, "method": method, "seed": 0,
                   "recovery_epochs": 0, "model": states[method], "masks": masks[method],
                   "parent_sha256": expected_sha, "calibration_ids_sha256": ids_sha,
                   "source_sha256": source, "test_accessed": False,
                   "gradient_training": False, "bn_calibration": False}
        checkpoint_sha = common.atomic_torch(method_dir / "checkpoint.pt", payload)
        frozen[method] = {"checkpoint_sha256": checkpoint_sha, "mask_sha256": mask_sha}
    common.atomic_json(out / "frozen_pretest.json", {"status": "ALL_CHECKPOINTS_FROZEN",
                                                  "methods": frozen, "test_accessed": False})
    # Official Test data are opened only after all five new checkpoints freeze.
    test_path = args.dataset_root / "extracted/cifar-100-python/test"
    test_data, test_labels = common.read_cifar(test_path, 10000)
    test_loader = DataLoader(common.CifarSubset(test_data, test_labels, range(10000), transform),
                             batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    results = []
    for method in METHODS:
        if common.digest(out / "methods" / method / "checkpoint.pt") != frozen[method]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed after freeze")
        started = time.perf_counter()
        accuracy, ce, count = common.evaluate_once(states[method], test_loader, device)
        result = {"method": method, "accuracy_percent": accuracy, "ce": ce,
                  "test_samples": count, "test_evaluations": 1,
                  "checkpoint_sha256": frozen[method]["checkpoint_sha256"],
                  "mask_sha256": frozen[method]["mask_sha256"],
                  "pruning_compensation_seconds": sum(r["wall_seconds"] for r in rows if r["method"] == method),
                  "test_seconds": time.perf_counter() - started}
        results.append(result)
        print(json.dumps({"event": "final_test", **result}), flush=True)
    common.atomic_csv(out / "results.csv", results)
    common.atomic_json(out / "status.json", {"status": "COMPLETE", "parent_kind": args.parent_kind,
                                          "protocol_id": protocol, "test_accessed_only_after_freeze": True,
                                          "calibration_seconds": calibration_seconds,
                                          "recovery_epochs": 0, "results": results,
                                          "peak_allocated_bytes": peak_allocated,
                                          "peak_reserved_bytes": peak_reserved})
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("".join(f"{common.digest(p)}  {p.relative_to(out).as_posix()}\n" for p in files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-kind", choices=tuple(common.PARENTS), required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException:
        if args.output.exists():
            common.atomic_json(args.output / "failure.json", {"status": "FAILED",
                                                              "traceback": traceback.format_exc()})
        raise
