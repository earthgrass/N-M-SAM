"""Geometric six-pair mask selection followed by physical fixed-support OBS.

This exploratory postprocessing is separate from APBR training and native
SparseGPT. All 19 layer masks and checkpoints are frozen before Test is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import matrix_to_weight, weight_to_matrix
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.fixedmask_obs import fixedmask_obs_matrix
from baseline_adapters.group_joint_obs import DELETION_PAIRS, prepare_obs_system, static_group_obs_costs
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers, validate_masks

ARCHIVE_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
PARENTS = {
    "fixed_l2": ("d68fbffc5f862be07656c5b45b8dcde2a5411865c0e5e03f2f45c2f97926b742", "cifar100_50k10k_apbr_fixed_l2_seed0_20261102", "apbr_fixed_l2"),
    "vpr": ("4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82", "cifar100_50k10k_final_v1", "apbr_sam"),
}
CALIBRATION_SEED = 20261027
CALIBRATION_COUNT = 1024
METHODS = ("group_obs", "geo_obs")


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_torch(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    return digest(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


class CifarSubset(Dataset):
    def __init__(self, data, labels, ids, transform):
        self.data, self.labels, self.ids, self.transform = data, labels, list(ids), transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = int(self.ids[index])
        image = self.data[sample_id].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(image), int(self.labels[sample_id]), sample_id


def read_cifar(path: Path, count: int):
    with path.open("rb") as f:
        raw = pickle.load(f, encoding="bytes")
    data = np.asarray(raw[b"data"])
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    per_class = count // 100
    if (data.dtype != np.uint8 or data.shape != (count, 3072)
            or labels.shape != (count,) or not np.array_equal(np.bincount(labels, minlength=100), np.full(100, per_class))):
        raise ValueError("Official CIFAR-100 data identity mismatch")
    return data, labels


def six_pair_keep(costs: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose minimum local quadratic cost with lexicographic stable ties."""
    if costs.ndim != 3 or costs.shape[-1] != 6 or costs.shape[1] != width // 4:
        raise ValueError("Six-pair cost shape mismatch")
    if not bool(torch.isfinite(costs).all()):
        raise ValueError("Nonfinite candidate costs")
    chosen = costs.argmin(dim=-1)
    rows, groups = chosen.shape
    keep = torch.ones((rows, width), dtype=torch.bool, device=costs.device)
    if groups:
        pairs = torch.tensor(DELETION_PAIRS, device=costs.device)
        selected = pairs[chosen]
        blocks = keep[:, :groups * 4].reshape(rows, groups, 4)
        blocks.scatter_(2, selected, False)
    validate_keep_mask(keep, 2, 4)
    return keep, chosen


def geometry_matrix(log_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if log_tensor.dtype != torch.float64 or log_tensor.shape != weight.shape:
        raise ValueError("Geometry coordinate/dtype mismatch")
    log = weight_to_matrix(log_tensor).to(device=weight.device, dtype=torch.float64)
    if not bool(torch.isfinite(log).all()):
        raise ValueError("Nonfinite geometry")
    radius = log.exp().to(torch.float32)
    if not bool(torch.isfinite(radius).all() and (radius > 0).all()):
        raise ValueError("Invalid geometry scale")
    return radius


def quadratic_cost(delta: torch.Tensor, h: torch.Tensor) -> float:
    delta = delta.float()
    value = 0.5 * torch.einsum("ri,ij,rj->", delta, h.float(), delta) / delta.shape[0]
    return float(value)


def evaluate_once(state: dict, loader: DataLoader, device: torch.device):
    model = build_cifar100_model().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    correct = count = 0
    ce_sum = 0.0
    with torch.inference_mode():
        for inputs, targets, _ in loader:
            logits = model(inputs.to(device, non_blocking=True))
            targets = targets.to(device, non_blocking=True)
            ce_sum += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            correct += int((logits.argmax(1) == targets).sum())
            count += len(targets)
    del model
    torch.cuda.empty_cache()
    return 100 * correct / count, ce_sum / count, count


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device required")
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if compute:
        raise RuntimeError(f"GPU occupied by compute process: {compute}")
    expected_sha, protocol, identity_method = PARENTS[args.parent_kind]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != protocol:
        raise ValueError("Protocol mismatch")
    if digest(args.parent) != expected_sha:
        raise ValueError("Parent checkpoint SHA256 mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    identity = parent.get("identity", {})
    if (parent.get("protocol_id") != protocol or parent.get("epoch") != 200
            or parent.get("steps") != 78200 or parent.get("test_accessed") is not False
            or identity.get("method") != identity_method or identity.get("seed") != 0):
        raise ValueError("Dense parent identity mismatch")
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169001437 or digest(archive, "md5") != ARCHIVE_MD5
            or digest(archive) != ARCHIVE_SHA256):
        raise ValueError("Official CIFAR-100 archive mismatch")
    train_path = args.dataset_root / "extracted/cifar-100-python/train"
    train_data, train_labels = read_cifar(train_path, 50000)
    ids = np.random.default_rng(CALIBRATION_SEED).permutation(50000)[:CALIBRATION_COUNT].tolist()
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    batches = DataLoader(CifarSubset(train_data, train_labels, ids, transform), batch_size=128,
                         shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(x.weight.numel() for x in layers.values()) != 11157504:
        raise ValueError("Eligible 2:4 layer contract mismatch")
    geometry = parent["stepper"]["geometry"]
    mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    if mapping != [(name, tuple(p.shape)) for name, p in model.named_parameters()]:
        raise ValueError("Geometry mapping does not match model parameters")
    if geometry.get("method") != args.parent_kind:
        raise ValueError("Geometry method mismatch")
    logs = {name: value for (name, _), value in zip(mapping, geometry["logs"])}
    for name, layer in layers.items():
        geometry_matrix(logs[name + ".weight"], layer.weight)
    source = {name: digest(ROOT / path) for name, path in {
        "runner": "scripts/run_geo_obs_seed0.py",
        "group_cost": "baseline_adapters/group_joint_obs/adapter.py",
        "fixedmask_obs": "baseline_adapters/fixedmask_obs/adapter.py",
        "sparsegpt_hessian": "baseline_adapters/sparsegpt/adapter.py",
        "conv2d": "baseline_adapters/common/conv2d.py",
        "model": "baseline_adapters/cifar100_model.py",
        "nm": "benchmark/nm.py",
    }.items()}
    environment = {
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    preflight = {"status": "PASS", "parent_kind": args.parent_kind, "parent_sha256": expected_sha,
                 "config_sha256": digest(args.config), "source_sha256": source,
                 "archive_sha256": ARCHIVE_SHA256, "train_pickle_sha256": digest(train_path),
                 "calibration_ids_sha256": ids_sha, "calibration_ids": ids, "geometry_method": geometry["method"],
                 "geometry_scale_min": min(float(x.min()) for x in geometry["logs"]),
                 "geometry_scale_max": max(float(x.max()) for x in geometry["logs"]),
                 "eligible_layers": list(layers), "eligible_positions": 11157504,
                 "expected_deletions": 5578752, "environment": environment, "test_accessed": False}
    atomic_json(out / "preflight.json", preflight)
    del parent
    adapters, hooks = {}, []
    for name, layer in layers.items():
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        adapters[name] = adapter
        hooks.append(layer.register_forward_pre_hook(lambda _m, values, item=adapter: item.add_batch(values[0])))
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    started = calibration_started = time.perf_counter()
    seen = []
    with torch.inference_mode():
        for inputs, _targets, sample_ids in batches:
            model(inputs.to(device, non_blocking=True))
            seen.extend(map(int, sample_ids.tolist()))
    torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - calibration_started
    for hook in hooks:
        hook.remove()
    if seen != ids:
        raise ValueError("Calibration sample order mismatch")
    hessians = {name: adapter.hessians()[0] for name, adapter in adapters.items()}
    del adapters
    states = {method: {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
              for method in METHODS}
    masks = {method: {} for method in METHODS}
    rows = []
    time_by_method = {method: 0.0 for method in METHODS}
    peak_allocated = peak_reserved = 0
    for index, (name, layer) in enumerate(layers.items(), 1):
        key = name + ".weight"
        w = weight_to_matrix(layer.weight.detach()).float()
        a = geometry_matrix(logs[key], layer.weight)
        v = w / a
        h = hessians[name]
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        selection_started = time.perf_counter()
        system = prepare_obs_system(h, percdamp=0.01, compute_dtype=torch.float32)
        group_costs, group_cond = static_group_obs_costs(w, system.inverse_hessian)
        geo_costs, geo_cond = static_group_obs_costs(v, system.inverse_hessian)
        if not torch.equal(group_cond, geo_cond):
            raise RuntimeError("Candidate systems diverged")
        costs = {"group_obs": group_costs, "geo_obs": geo_costs}
        for method in METHODS:
            keep, chosen = six_pair_keep(costs[method], w.shape[1])
            torch.cuda.synchronize()
            selection_seconds = time.perf_counter() - selection_started
            obs_started = time.perf_counter()
            compensated = fixedmask_obs_matrix(w, h, keep, keep_n=2, m=4, blocksize=128,
                                               percdamp=0.01, compute_dtype=torch.float32)
            final = compensated.weight
            final[~keep] = 0
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - obs_started
            if not bool(torch.isfinite(final).all()) or int(torch.count_nonzero(final[~keep])):
                raise FloatingPointError(f"Invalid compensated output: {name}/{method}")
            if compensated.survivor_changed_count == 0:
                raise RuntimeError(f"OBS did not change surviving weights: {name}/{method}")
            masks[method][key] = keep.reshape_as(layer.weight).detach().cpu()
            states[method][key] = matrix_to_weight(final.detach().cpu(), layer.weight.shape)
            raw = w * keep
            row = {"layer_index": index, "layer": name, "method": method,
                   "rows": w.shape[0], "K": w.shape[1], "deleted": int((~keep).sum()),
                   "selected_cost_mean": float(costs[method].gather(-1, chosen[..., None]).mean()),
                   "candidate_condition_max": float(group_cond.max()),
                   "physical_mask_only_cost": quadratic_cost(raw - w, h),
                   "physical_post_obs_cost": quadratic_cost(final - w, h),
                   "surrogate_post_obs_cost": quadratic_cost((final / a) - v, h),
                   "survivor_update_rms": float((final - w)[keep].square().mean().sqrt()),
                   "mask_selection_seconds": selection_seconds,
                   "obs_decomposition_seconds": compensated.decomposition_seconds,
                   "obs_compensation_seconds": compensated.compensation_seconds,
                   "method_layer_seconds": selection_seconds + elapsed,
                   "finite": True, "deleted_zero": True, "strict_2of4": True}
            rows.append(row)
            time_by_method[method] += selection_seconds + elapsed
            atomic_torch(out / "layers" / f"{index:02d}_{name.replace('.', '_')}_{method}.pt",
                         {"layer": name, "method": method, "mask": masks[method][key],
                          "weight": states[method][key], "chosen_pair": chosen.detach().cpu(),
                          "candidate_costs": costs[method].detach().cpu(), "diagnostic": row})
        flips = int((masks["group_obs"][key] != masks["geo_obs"][key]).sum())
        rows[-1]["mask_flip_count_vs_group_obs"] = flips
        rows[-1]["mask_flip_rate_vs_group_obs"] = flips / w.numel()
        atomic_json(out / "progress.json", {"status": "healthy_partial", "completed_layers": index,
                                            "last_layer": name, "test_accessed": False})
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        print(json.dumps({"event": "layer_complete", "layer": name, "index": index,
                          "mask_flip_rate": flips / w.numel(), "test_accessed": False}), flush=True)
        del system, group_costs, geo_costs, group_cond, geo_cond
        torch.cuda.empty_cache()
    for method in METHODS:
        check = build_cifar100_model()
        check.load_state_dict(states[method], strict=True)
        validate_masks(check, masks[method], n=2, m=4, complete=True)
        if sum(int((~m).sum()) for m in masks[method].values()) != 5578752:
            raise ValueError("Full model deletion count mismatch")
    atomic_csv(out / "per_layer.csv", rows)
    frozen = {}
    for method in METHODS:
        payload = {"schema": 1, "status": "healthy", "protocol_id": protocol,
                   "experiment_id": "geo_obs_seed0_20261102_v1", "parent_kind": args.parent_kind,
                   "stage": "post_obs_raw_hard", "method": method, "seed": 0, "recovery_epochs": 0,
                   "model": states[method], "masks": masks[method], "parent_sha256": expected_sha,
                   "config_sha256": digest(args.config), "calibration_ids_sha256": ids_sha,
                   "source_sha256": source, "test_accessed": False, "gradient_training": False,
                   "bn_calibration": False}
        method_dir = out / "methods" / method
        mask_sha = atomic_torch(method_dir / "masks.pt", masks[method])
        checkpoint_sha = atomic_torch(method_dir / "checkpoint.pt", payload)
        frozen[method] = {"checkpoint_sha256": checkpoint_sha, "mask_sha256": mask_sha}
    atomic_json(out / "frozen_pretest.json", {"status": "ALL_CHECKPOINTS_FROZEN",
                                              "methods": frozen, "test_accessed": False})
    test_path = args.dataset_root / "extracted/cifar-100-python/test"
    test_data, test_labels = read_cifar(test_path, 10000)
    test_loader = DataLoader(CifarSubset(test_data, test_labels, range(10000), transform),
                             batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    results = []
    for method in METHODS:
        checkpoint_path = out / "methods" / method / "checkpoint.pt"
        if digest(checkpoint_path) != frozen[method]["checkpoint_sha256"]:
            raise RuntimeError("Frozen checkpoint changed before Test")
        test_started = time.perf_counter()
        accuracy, ce, count = evaluate_once(states[method], test_loader, device)
        results.append({"method": method, "accuracy_percent": accuracy, "ce": ce,
                        "test_samples": count, "test_evaluations": 1,
                        "checkpoint_sha256": frozen[method]["checkpoint_sha256"],
                        "mask_sha256": frozen[method]["mask_sha256"],
                        "pruning_compensation_seconds": time_by_method[method],
                        "test_seconds": time.perf_counter() - test_started})
        print(json.dumps({"event": "final_test", **results[-1]}), flush=True)
    atomic_csv(out / "results.csv", results)
    atomic_json(out / "status.json", {"status": "COMPLETE", "parent_kind": args.parent_kind,
                                      "test_accessed_only_after_freeze": True, "recovery_epochs": 0,
                                      "results": results, "calibration_seconds": calibration_seconds,
                                      "total_seconds": time.perf_counter() - started,
                                      "peak_allocated_bytes": peak_allocated,
                                      "peak_reserved_bytes": peak_reserved})
    all_files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("".join(f"{digest(p)}  {p.relative_to(out).as_posix()}\n" for p in all_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-kind", required=True, choices=tuple(PARENTS))
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        if args.output.exists():
            atomic_json(args.output / "failure.json", {"status": "FAILED", "traceback": traceback.format_exc(),
                                                       "test_accessed": (args.output / "frozen_pretest.json").exists()})
        raise
